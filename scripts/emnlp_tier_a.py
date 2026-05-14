"""
EMNLP Tier A: Robustness Expansion
===================================

PHASE 1: Greater-than dense sweep + L4H0 attention-from-S2
  - Retrained Pythia-160M, 51 dense checkpoints (steps 1000-5000)
  - Per checkpoint: greater-than accuracy and P(>)-P(<=) with bootstrap CIs
  - Per checkpoint: L4H0 attention pattern at S2 query position
    (BOS, S1, S2-self, IO as key positions)
  - The L4H0 attention measurement is the fix for Phase 3 of the
    robustness battery, which measured the wrong direction (END query
    instead of S2 query).
  - The dense greater-than sweep mirrors the IOI dense sweep for
    direct side-by-side comparison.

PHASE 2: IOI sign flip on Pythia-410M
  - Original Pythia-410M-deduped, 5 sparse checkpoints
  - Layer range scaled to model depth: layers 6-10 (24 layers total)
  - Demonstrates scale invariance of the sign flip.

PHASE 3: IOI sign flip on Pythia-1B
  - Original Pythia-1B-deduped, 5 sparse checkpoints
  - Layer range scaled to model depth: layers 4-6 (16 layers total)
  - Stronger scale invariance argument.

Output: results/emnlp_tier_a.json
Log:    results/emnlp_tier_a_log.txt
Runtime: ~3-5 hours on A100.
"""

import os
import gc
import json
import time
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES


# ----------------------------- Config -----------------------------

DEVICE = "cuda"
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL_160M = "EleutherAI/pythia-160m-deduped"
BASE_MODEL_410M = "EleutherAI/pythia-410m-deduped"
BASE_MODEL_1B = "EleutherAI/pythia-1b-deduped"
SEED = 42
N_BOOTSTRAP = 10_000

# Phase 1
DENSE_STEPS = list(range(1000, 3001, 50)) + list(range(3200, 5001, 200))  # 51
L4H0_LAYER = 4
L4H0_HEAD = 0
GT_EVENTS = ["war", "battle", "dispute", "conflict", "argument", "siege"]
GT_VERBS = ["lasted", "ran", "extended", "continued", "stretched"]
GT_N_PROMPTS = 300
TRAJ_TEMPLATES = ALL_TEMPLATES[:10]
TRAJ_PPT = 30

# Phase 2 (410M): 24 layers, patch at 6-10 (~25-42% depth)
PYTHIA_410M_LAYERS = [6, 7, 8, 9, 10]
# Phase 3 (1B): 16 layers, patch at 4-6 (~25-44% depth)
PYTHIA_1B_LAYERS = [4, 5, 6]

# Sparse checkpoints for multi-scale sign flip: pre-dip / floor / mid-recovery / mature
# Based on existing pythia_410m_ioi_sweep.json and pythia_1b_ioi_sweep.json:
#   - 410M dip floor: step 2000 (acc=42%)
#   - 1B dip floor: step 1000 (acc=38%)
PYTHIA_410M_STEPS = [256, 1000, 2000, 3000, 5000, 143000]
PYTHIA_1B_STEPS = [256, 512, 1000, 3000, 5000, 143000]

# Common protocol for sign flip
PATCH_TEMPLATES = ALL_TEMPLATES[:10]
PATCH_PPT = 30

RESULTS_PATH = "results/emnlp_tier_a.json"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_retrained(step):
    hf = AutoModelForCausalLM.from_pretrained(
        RETRAINED_REPO, subfolder=f"step_{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL_160M, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
    )
    del hf
    torch.cuda.empty_cache()
    return model


def load_pythia_original(model_name, step):
    hf = AutoModelForCausalLM.from_pretrained(
        model_name, revision=f"step{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        model_name, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
    )
    del hf
    torch.cuda.empty_cache()
    return model


def find_token_positions(token_row, io_token_id, s_token_id):
    s_count = 0
    s1, s2, io_pos = -1, -1, -1
    for j in range(1, token_row.shape[0]):
        tok = int(token_row[j].item())
        if tok == int(s_token_id):
            s_count += 1
            if s_count == 1:
                s1 = j
            elif s_count == 2:
                s2 = j
        if tok == int(io_token_id) and io_pos == -1:
            io_pos = j
    return {
        "BOS": 0, "S1": s1, "S2": s2, "IO": io_pos,
        "END": token_row.shape[0] - 1,
    }


def get_single_token_names(tokenizer):
    ids = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    return ids


def get_two_digit_completion_tokens(tokenizer):
    """For greater-than: map 2-digit completion to single token id."""
    mapping = {}
    for d in range(100):
        s = f"{d:02d}"
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            mapping[d] = ids[0]
    return mapping


def make_greater_than_prompts(seed=SEED):
    rng = np.random.default_rng(seed)
    prompts = []
    for _ in range(GT_N_PROMPTS):
        event = rng.choice(GT_EVENTS)
        verb = rng.choice(GT_VERBS)
        y_low_2 = int(rng.integers(3, 97))
        year = 1700 + y_low_2
        prompts.append({
            "prompt": f"The {event} {verb} from the year {year} to the year 17",
            "start_yy": y_low_2,
        })
    return prompts


def eval_greater_than(model, prompts, digit_tokens):
    valid_digits = sorted(digit_tokens.keys())
    token_ids = torch.tensor(
        [digit_tokens[d] for d in valid_digits], device=DEVICE,
    )

    diffs, p_gt, p_le, correct = [], [], [], []
    for p in prompts:
        tokens = model.to_tokens(p["prompt"]).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        digit_probs = probs[token_ids].cpu().numpy()
        gmask = np.array([d > p["start_yy"] for d in valid_digits])
        lmask = np.array([d <= p["start_yy"] for d in valid_digits])
        pg = float(digit_probs[gmask].sum())
        pl = float(digit_probs[lmask].sum())
        p_gt.append(pg)
        p_le.append(pl)
        diffs.append(pg - pl)
        correct.append(1.0 if pg > pl else 0.0)
    return np.asarray(diffs), np.asarray(correct), np.asarray(p_gt), np.asarray(p_le)


def measure_l4h0_attention_from_s2(model):
    """Across TRAJ_TEMPLATES * TRAJ_PPT prompts, measure L4H0 attention
    pattern with S2 as the query position. Records attention TO each
    of BOS, S1, S2 (self), IO. Returns mean attention values.
    """
    attn_to = {"BOS": [], "S1": [], "S2_self": [], "IO": []}

    for tmpl in TRAJ_TEMPLATES:
        ds = IOIDataset(
            model=model, n_prompts=TRAJ_PPT, templates=[tmpl],
            symmetric=True, seed=SEED,
        )
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)

        pat_cache = {}

        def cap_pattern(value, hook):
            pat_cache["pattern"] = value.detach()
            return value

        with torch.no_grad():
            model.run_with_hooks(
                ioi_tokens,
                fwd_hooks=[
                    (f"blocks.{L4H0_LAYER}.attn.hook_pattern", cap_pattern),
                ],
            )

        pat = pat_cache["pattern"][:, L4H0_HEAD, :, :]  # [B, Q, K]

        for b in range(ioi_tokens.shape[0]):
            pos = find_token_positions(
                ioi_tokens[b].cpu(), ds.io_token_ids[b], ds.s_token_ids[b],
            )
            s2_q = pos["S2"]
            if s2_q < 0:
                continue
            attn_to["BOS"].append(float(pat[b, s2_q, pos["BOS"]].item()))
            if pos["S1"] >= 0:
                attn_to["S1"].append(float(pat[b, s2_q, pos["S1"]].item()))
            attn_to["S2_self"].append(float(pat[b, s2_q, s2_q].item()))
            if pos["IO"] >= 0:
                attn_to["IO"].append(float(pat[b, s2_q, pos["IO"]].item()))

        del pat_cache
        torch.cuda.empty_cache()

    return {k: float(np.mean(v)) if v else float("nan") for k, v in attn_to.items()}


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.shape[0], size=(n_resamples, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ====================================================================
# PHASE 1: Greater-than dense sweep + L4H0 attention from S2
# ====================================================================

def phase1():
    log("=" * 60)
    log("PHASE 1: Greater-than dense + L4H0 attention from S2")
    log("=" * 60)

    out = {"by_step": {}, "config": {
        "model": RETRAINED_REPO,
        "n_steps": len(DENSE_STEPS),
        "gt_n_prompts": GT_N_PROMPTS,
        "l4h0_n_prompts": len(TRAJ_TEMPLATES) * TRAJ_PPT,
        "l4h0_position": f"L{L4H0_LAYER}H{L4H0_HEAD}",
    }}

    gt_prompts = make_greater_than_prompts(SEED)
    t0 = time.time()

    for i, step in enumerate(DENSE_STEPS):
        try:
            model = load_retrained(step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        digit_tokens = get_two_digit_completion_tokens(model.tokenizer)
        gt_diffs, gt_correct, gt_pg, gt_pl = eval_greater_than(
            model, gt_prompts, digit_tokens,
        )
        diff_lo, diff_hi = bootstrap_ci(gt_diffs)

        attn = measure_l4h0_attention_from_s2(model)

        out["by_step"][f"step_{step}"] = {
            "gt_acc": float(gt_correct.mean()),
            "gt_mean_diff": float(gt_diffs.mean()),
            "gt_diff_ci95": [diff_lo, diff_hi],
            "gt_mean_p_greater": float(gt_pg.mean()),
            "gt_mean_p_lesseq": float(gt_pl.mean()),
            "l4h0_attn_from_s2": attn,
        }

        if (i + 1) % 5 == 0 or i == len(DENSE_STEPS) - 1:
            elapsed = time.time() - t0
            log(
                f"  [{i+1}/{len(DENSE_STEPS)}] step={step}  "
                f"gt_acc={gt_correct.mean()*100:.1f}%  "
                f"P(>)-P(<=)={gt_diffs.mean():+.3f}  "
                f"L4H0[S2->S1]={attn.get('S1', float('nan')):.3f}  "
                f"L4H0[S2->IO]={attn.get('IO', float('nan')):.3f}  "
                f"({elapsed/(i+1):.1f}s/step)"
            )

        # Save incrementally so partial progress survives a crash.
        save_results({"phase1_greater_than_and_l4h0": out, **
                      {k: v for k, v in _global_results.items() if k != "phase1_greater_than_and_l4h0"}})

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# Common IOI sign-flip helper (used by Phase 2 and 3)
# ====================================================================

def run_sign_flip_at_step(model, layer_range):
    """Causal intervention: replace IOI S2 residual with control S2
    residual at the given layer range. Same protocol as causal_intervention.py.
    """
    rng = np.random.default_rng(SEED + 1)
    single_name_ids = get_single_token_names(model.tokenizer)

    base_lds = []
    patched_lds = []

    for tmpl in PATCH_TEMPLATES:
        ds = IOIDataset(
            model=model, n_prompts=PATCH_PPT, templates=[tmpl],
            symmetric=True, seed=SEED,
        )
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]

        s2_positions = []
        for i in range(n):
            pos = find_token_positions(
                ioi_tokens[i].cpu(), ds.io_token_ids[i], ds.s_token_ids[i],
            )
            s2_positions.append(pos["S2"])
        s2_positions = torch.tensor(s2_positions, dtype=torch.long, device=DEVICE)

        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            io_id = int(ds.io_token_ids[i])
            s_id = int(ds.s_token_ids[i])
            pool = [t for t in single_name_ids if t != io_id and t != s_id]
            if pool and s2_positions[i] > 0:
                ctrl_tokens[i, s2_positions[i]] = int(rng.choice(pool))

        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

        names = [f"blocks.{L}.hook_resid_post" for L in layer_range]
        donor = {}

        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(
                ctrl_tokens,
                fwd_hooks=[(n, make_cap(n)) for n in names],
            )

        with torch.no_grad():
            base_logits = model(ioi_tokens)
        last = base_logits[:, -1, :]
        idx = torch.arange(n, device=DEVICE)
        base = (last[idx, io_ids] - last[idx, s_ids]).cpu().numpy()

        def make_patch(name):
            donor_act = donor[name]
            def fn(value, hook):
                for i in range(value.shape[0]):
                    p = int(s2_positions[i].item())
                    if p > 0:
                        value[i, p, :] = donor_act[i, p, :]
                return value
            return fn

        hooks = [(n, make_patch(n)) for n in names]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
        last = patched_logits[:, -1, :]
        patched = (last[idx, io_ids] - last[idx, s_ids]).cpu().numpy()

        base_lds.extend(base.tolist())
        patched_lds.extend(patched.tolist())

        del ioi_tokens, ctrl_tokens, base_logits, patched_logits, donor
        torch.cuda.empty_cache()

    base_arr = np.asarray(base_lds)
    patched_arr = np.asarray(patched_lds)
    deltas = patched_arr - base_arr
    delta_lo, delta_hi = bootstrap_ci(deltas)

    return {
        "n_prompts": int(base_arr.shape[0]),
        "ioi_acc": float((base_arr > 0).mean()),
        "base_ld_mean": float(base_arr.mean()),
        "patched_ld_mean": float(patched_arr.mean()),
        "delta_ld_mean": float(deltas.mean()),
        "delta_ld_ci95": [delta_lo, delta_hi],
    }


# ====================================================================
# PHASE 2: Pythia-410M sign flip
# ====================================================================

def phase2():
    log("=" * 60)
    log("PHASE 2: IOI sign flip on Pythia-410M")
    log("=" * 60)
    out = {"by_step": {}, "config": {
        "model": BASE_MODEL_410M,
        "n_layers": 24,
        "patch_layers": PYTHIA_410M_LAYERS,
        "templates": len(PATCH_TEMPLATES),
        "prompts_per_template": PATCH_PPT,
    }}

    for step in PYTHIA_410M_STEPS:
        try:
            log(f"  loading step {step}")
            model = load_pythia_original(BASE_MODEL_410M, step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        r = run_sign_flip_at_step(model, PYTHIA_410M_LAYERS)
        out["by_step"][f"step_{step}"] = r
        log(
            f"    step={step}  acc={r['ioi_acc']:.3f}  "
            f"base_LD={r['base_ld_mean']:+.4f}  "
            f"ΔLD={r['delta_ld_mean']:+.4f} "
            f"[{r['delta_ld_ci95'][0]:+.3f}, {r['delta_ld_ci95'][1]:+.3f}]"
        )

        save_results({**_global_results, "phase2_pythia_410m": out})

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# PHASE 3: Pythia-1B sign flip
# ====================================================================

def phase3():
    log("=" * 60)
    log("PHASE 3: IOI sign flip on Pythia-1B")
    log("=" * 60)
    out = {"by_step": {}, "config": {
        "model": BASE_MODEL_1B,
        "n_layers": 16,
        "patch_layers": PYTHIA_1B_LAYERS,
        "templates": len(PATCH_TEMPLATES),
        "prompts_per_template": PATCH_PPT,
    }}

    for step in PYTHIA_1B_STEPS:
        try:
            log(f"  loading step {step}")
            model = load_pythia_original(BASE_MODEL_1B, step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        r = run_sign_flip_at_step(model, PYTHIA_1B_LAYERS)
        out["by_step"][f"step_{step}"] = r
        log(
            f"    step={step}  acc={r['ioi_acc']:.3f}  "
            f"base_LD={r['base_ld_mean']:+.4f}  "
            f"ΔLD={r['delta_ld_mean']:+.4f} "
            f"[{r['delta_ld_ci95'][0]:+.3f}, {r['delta_ld_ci95'][1]:+.3f}]"
        )

        save_results({**_global_results, "phase3_pythia_1b": out})

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# MAIN
# ====================================================================

_global_results = {}


def main():
    global _global_results
    os.makedirs("results", exist_ok=True)

    _global_results = {"config": {
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
    }}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                _global_results = json.load(f)
            done = [k for k in _global_results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception as e:
            log(f"Could not resume: {e}")

    phases = [
        ("phase1_greater_than_and_l4h0", phase1),
        ("phase2_pythia_410m", phase2),
        ("phase3_pythia_1b", phase3),
    ]

    t0 = time.time()
    for key, fn in phases:
        cached = _global_results.get(key)
        if cached is not None and "error" not in cached:
            log(f"SKIP {key}: already done")
            continue
        if cached is not None and "error" in cached:
            log(f"RETRY {key}: previous run errored ({cached['error']})")
        log(f"START {key}")
        try:
            _global_results[key] = fn()
            save_results(_global_results)
        except Exception as e:
            log(f"FAILED {key}: {e}")
            traceback.print_exc()
            _global_results[key] = {"error": str(e)}
            save_results(_global_results)

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
