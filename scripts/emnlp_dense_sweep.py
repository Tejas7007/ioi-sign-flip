"""
EMNLP Dense Transition Sweep
============================
Runs the S2 activation patching intervention at every available retrained
checkpoint from step 1000 to step 5000 (51 checkpoints), and measures
L8H9 attention to S2 at each step. Produces the data for the headline
transition figure: ΔLD crosses zero somewhere in 2000-3000, tracking the
L8H9 attention phase transition.

Per checkpoint, we measure (n=300 prompts, 10 templates x 30):
  1. Baseline IOI logit difference and accuracy.
  2. ΔLD from replacing S2 residual stream with control (layers 3-5).
  3. L8H9 attention to S2, averaged over IOI prompts.
  4. Bootstrap 95% CI on ΔLD (10,000 resamples).

Matches the experimental protocol of causal_intervention.py for the five
sparse checkpoints already in results/causal_intervention.json, so the
existing data points can be overlaid on the dense curve as a robustness
check across the original-Pythia vs retrained boundary.

Output: results/emnlp_dense_sweep.json
Log:    results/emnlp_dense_sweep_log.txt
Runtime: ~6 hours on a single A100.
"""

import os
import gc
import json
import time
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES


# ----------------------------- Config -----------------------------

DEVICE = "cuda"
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"   # architecture template
PATCH_LAYERS = [3, 4, 5]
L8H9_LAYER = 8
L8H9_HEAD = 9
TEMPLATES = ALL_TEMPLATES[:10]                  # 10 templates
PROMPTS_PER_TEMPLATE = 30                       # n = 300 total per checkpoint
SEED = 42
N_BOOTSTRAP = 10_000

# Dense schedule on the retrained model:
#   every 50 steps in [1000, 3000]   -> 41 points
#   every 200 steps in (3000, 5000]  -> 10 points
STEPS_DENSE = list(range(1000, 3001, 50)) + list(range(3200, 5001, 200))
# Length = 41 + 10 = 51. Sanity-checked by assertion below.

RESULTS_PATH = "results/emnlp_dense_sweep.json"
LOG_PATH = "results/emnlp_dense_sweep_log.txt"


# --------------------------- Utilities ----------------------------

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def load_retrained(step):
    """Load retrained-seed42 checkpoint into a HookedTransformer."""
    hf = AutoModelForCausalLM.from_pretrained(
        RETRAINED_REPO, subfolder=f"step_{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL,
        hf_model=hf,
        device=DEVICE,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf
    torch.cuda.empty_cache()
    return model


def find_s2_position(token_row, s_token_id):
    """Return the index of the second occurrence of s_token_id in token_row."""
    seen = 0
    for j in range(1, token_row.shape[0]):  # skip BOS
        if int(token_row[j].item()) == int(s_token_id):
            seen += 1
            if seen == 2:
                return j
    return -1


def build_control_dataset(model, ds):
    """For each IOI prompt, build a matched control prompt that replaces the
    S2 *token* with a third single-token name (no duplication). Returns
    (control_tokens, ioi_s2_positions, control_s2_positions).

    The control prompt has the same template, same IO name, same S1 name,
    but a different name at the S2 slot. Since the IOIDataset uses the same
    template per call, replacing the token at S2 position is enough to
    create a no-duplication variant.
    """
    rng = np.random.default_rng(SEED + 1)
    ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
    n = ioi_tokens.shape[0]

    # Find S2 positions in the IOI prompts (this is where we patch).
    s2_positions = []
    for i in range(n):
        s2 = find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i])
        s2_positions.append(s2)
    s2_positions = torch.tensor(s2_positions, dtype=torch.long, device=DEVICE)

    # Build controls by swapping S2 token to a non-IO, non-S name.
    # Use the model's tokenizer over the CANDIDATE_NAMES pool from the dataset.
    try:
        from circuitscaling.datasets import CANDIDATE_NAMES
    except ImportError:
        from src.circuitscaling.datasets import CANDIDATE_NAMES

    # Build a vocabulary of single-token name ids.
    name_token_ids = []
    for name in CANDIDATE_NAMES:
        toks = model.to_tokens(" " + name, prepend_bos=False)
        if toks.shape[1] == 1:
            name_token_ids.append(int(toks[0, 0].item()))
    name_token_ids = list(set(name_token_ids))

    control_tokens = ioi_tokens.clone()
    for i in range(n):
        io_id = int(ds.io_token_ids[i])
        s_id = int(ds.s_token_ids[i])
        pool = [t for t in name_token_ids if t != io_id and t != s_id]
        replacement = int(rng.choice(pool))
        s2_pos = int(s2_positions[i].item())
        if s2_pos >= 0:
            control_tokens[i, s2_pos] = replacement

    return ioi_tokens, control_tokens, s2_positions


def logit_diff_per_prompt(logits, io_ids, s_ids):
    """Per-prompt LD at the final token position."""
    last = logits[:, -1, :]
    idx = torch.arange(last.shape[0], device=last.device)
    return last[idx, io_ids] - last[idx, s_ids]


def s2_patch_hook_factory(donor_cache, s2_positions, layer):
    """Returns a hook that replaces the residual stream at S2 with the
    donor's residual stream at S2 for the corresponding row."""
    donor_act = donor_cache[f"blocks.{layer}.hook_resid_post"]  # [B, T, D]

    def hook_fn(value, hook):
        for i in range(value.shape[0]):
            p = int(s2_positions[i].item())
            if p >= 0:
                value[i, p, :] = donor_act[i, p, :]
        return value

    return hook_fn


def measure_l8h9_attn_to_s2(model, ioi_tokens, s2_positions):
    """Mean attention from the final token position back to the S2 token
    position, for head L8H9, averaged over the batch.

    The S-inhibition head reads from S2 at the END position, so this is
    the standard 'L8H9 attention to S2' quantity used in the paper.
    """
    cache = {}

    def cap(value, hook):
        cache["pattern"] = value.detach()
        return value

    model.run_with_hooks(
        ioi_tokens,
        fwd_hooks=[(f"blocks.{L8H9_LAYER}.attn.hook_pattern", cap)],
    )
    # pattern shape: [batch, head, query, key]
    pat = cache["pattern"][:, L8H9_HEAD, :, :]  # [B, Q, K]
    end_q = pat.shape[1] - 1  # final-token query position
    vals = []
    for i in range(pat.shape[0]):
        k = int(s2_positions[i].item())
        if k >= 0:
            vals.append(float(pat[i, end_q, k].item()))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI on the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


# --------------------------- Per-step driver ----------------------

def run_one_step(model, step):
    """Run the full set of measurements for a single checkpoint."""
    # Per-template, accumulate per-prompt arrays. Templates have different
    # lengths so we cannot stack tokens across them.
    base_lds, patched_lds, deltas = [], [], []
    l8h9_attns = []

    for tmpl in TEMPLATES:
        ds = IOIDataset(
            model=model, n_prompts=PROMPTS_PER_TEMPLATE,
            templates=[tmpl], symmetric=True, seed=SEED,
        )
        ioi_tokens, ctrl_tokens, s2_positions = build_control_dataset(model, ds)
        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

        # Cache control activations for the patch layers.
        names = [f"blocks.{L}.hook_resid_post" for L in PATCH_LAYERS]

        def make_cap(name, store):
            def fn(value, hook):
                store[name] = value.detach()
                return value
            return fn

        donor = {}
        model.run_with_hooks(
            ctrl_tokens,
            fwd_hooks=[(n, make_cap(n, donor)) for n in names],
        )

        # Baseline IOI logits.
        with torch.no_grad():
            base_logits = model(ioi_tokens)
        base_ld = logit_diff_per_prompt(base_logits, io_ids, s_ids).detach().cpu().numpy()

        # Patched IOI logits (S2 residual replaced at PATCH_LAYERS).
        hooks = [
            (f"blocks.{L}.hook_resid_post",
             s2_patch_hook_factory(donor, s2_positions, L))
            for L in PATCH_LAYERS
        ]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
        patched_ld = logit_diff_per_prompt(patched_logits, io_ids, s_ids).detach().cpu().numpy()

        delta = patched_ld - base_ld

        # L8H9 attention to S2 on unpatched IOI prompts.
        attn = measure_l8h9_attn_to_s2(model, ioi_tokens, s2_positions)

        base_lds.extend(base_ld.tolist())
        patched_lds.extend(patched_ld.tolist())
        deltas.extend(delta.tolist())
        l8h9_attns.append(attn)

        del ioi_tokens, ctrl_tokens, base_logits, patched_logits, donor
        torch.cuda.empty_cache()

    base_lds = np.asarray(base_lds)
    patched_lds = np.asarray(patched_lds)
    deltas = np.asarray(deltas)

    acc = float((base_lds > 0).mean())
    base_mean = float(base_lds.mean())
    patched_mean = float(patched_lds.mean())
    delta_mean = float(deltas.mean())
    delta_lo, delta_hi = bootstrap_ci(deltas)
    base_lo, base_hi = bootstrap_ci(base_lds)
    l8h9 = float(np.mean(l8h9_attns))

    return {
        "step": step,
        "n_prompts": int(len(base_lds)),
        "ioi_acc": acc,
        "base_ld_mean": base_mean,
        "base_ld_ci95": [base_lo, base_hi],
        "patched_ld_mean": patched_mean,
        "delta_ld_mean": delta_mean,
        "delta_ld_ci95": [delta_lo, delta_hi],
        "l8h9_attn_to_s2": l8h9,
    }


# ------------------------------ Main ------------------------------

def main():
    os.makedirs("results", exist_ok=True)
    assert len(STEPS_DENSE) == 51, f"Expected 51 steps, got {len(STEPS_DENSE)}"

    # Resume support: skip checkpoints already in the output file.
    results = {"config": {
        "model": RETRAINED_REPO,
        "patch_layers": PATCH_LAYERS,
        "templates": len(TEMPLATES),
        "prompts_per_template": PROMPTS_PER_TEMPLATE,
        "n_total": len(TEMPLATES) * PROMPTS_PER_TEMPLATE,
        "n_bootstrap": N_BOOTSTRAP,
        "steps": STEPS_DENSE,
    }, "by_step": {}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                prev = json.load(f)
            if "by_step" in prev:
                results["by_step"] = prev["by_step"]
                log(f"Resuming. {len(results['by_step'])} steps already complete.")
        except Exception as e:
            log(f"Could not resume from existing file: {e}")

    t0 = time.time()
    for i, step in enumerate(STEPS_DENSE):
        key = f"step_{step}"
        if key in results["by_step"]:
            log(f"[{i+1}/51] step={step}: cached, skipping.")
            continue

        log(f"[{i+1}/51] step={step}: loading retrained checkpoint")
        try:
            model = load_retrained(step)
        except Exception as e:
            log(f"  FAILED to load step {step}: {e}")
            continue

        try:
            row = run_one_step(model, step)
        except Exception as e:
            log(f"  FAILED at step {step}: {e}")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        results["by_step"][key] = row
        log(
            f"  step={step}  acc={row['ioi_acc']:.3f}  "
            f"base_LD={row['base_ld_mean']:+.4f}  "
            f"ΔLD={row['delta_ld_mean']:+.4f} "
            f"[{row['delta_ld_ci95'][0]:+.3f}, {row['delta_ld_ci95'][1]:+.3f}]  "
            f"L8H9->S2={row['l8h9_attn_to_s2']:.3f}"
        )

        # Save after every checkpoint so partial progress survives a crash.
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
