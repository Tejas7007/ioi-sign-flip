"""
EMNLP Gap Filler: Final Reviewer-Proofing Experiments
======================================================

Closes the last gaps a tough reviewer would find.

PHASE 1: Greater-than SIGN FLIP at maturity (step 143000)
  - Same causal intervention as task_generalization Phase 4
  - At maturity, model NEEDS first-year info for correct comparison
  - Expected: ΔLD strongly NEGATIVE (completing the +0.26 → negative flip)
  - Without this, the "sign flip" claim is only shown for IOI

PHASE 2: SVA sign flip at maturity (step 143000)
  - At maturity, model has robust syntactic parsing
  - Patching attractor should hurt slightly or be neutral
  - Completes the SVA dip story

PHASE 3: SVA intervention at step 1000 (stronger effect size)
  - Step 512 gave Δ=+0.008 (uncomfortably small, tiny absolute probs)
  - Step 1000 has acc=39%, model puts measurable mass on both verbs
  - Expect a larger, more convincing absolute effect

PHASE 4: Greater-than dip at Pythia-410M (multi-scale)
  - 7 sparse checkpoints, accuracy sweep
  - If GT dips at 410M too → "dips generalize across tasks AND scales"

PHASE 5: Greater-than dip at Pythia-1B (multi-scale)
  - Same protocol at 1B
  - Strongest scale-invariance evidence

Output: results/emnlp_gap_filler.json
Runtime: ~35-45 minutes on A100.
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
BASE_MODEL_160M = "EleutherAI/pythia-160m-deduped"
BASE_MODEL_410M = "EleutherAI/pythia-410m-deduped"
BASE_MODEL_1B = "EleutherAI/pythia-1b-deduped"
SEED = 42
N_BOOTSTRAP = 10_000

# Intervention config
GT_EVENTS = ["war", "battle", "dispute", "conflict", "argument", "siege"]
GT_VERBS = ["lasted", "ran", "extended", "continued", "stretched"]
GT_N_PROMPTS = 300
INTERVENTION_LAYERS_160M = [3, 4, 5]

# SVA config
SVA_SINGULAR_NOUNS = [
    "boy", "girl", "dog", "cat", "doctor", "writer",
    "teacher", "child", "athlete", "actor",
]
SVA_PLURAL_NOUNS = [
    "boys", "girls", "dogs", "cats", "doctors", "writers",
    "teachers", "children", "athletes", "actors",
]
SVA_PREPOSITIONS = ["near", "beside", "with", "behind"]
SVA_VERB_PAIRS = [("is", "are"), ("was", "were"), ("has", "have")]
SVA_N_PROMPTS = 200

# Multi-scale GT dip checkpoints
GT_MULTISCALE_STEPS = [0, 512, 1000, 2000, 3000, 5000, 10000, 143000]

RESULTS_PATH = "results/emnlp_gap_filler.json"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.shape[0], size=(n_resamples, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def get_two_digit_completion_tokens(tokenizer):
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


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ====================================================================
# PHASE 1: Greater-than sign flip at MATURITY
# ====================================================================

def phase1_gt_mature_intervention():
    """Run greater-than causal intervention at step 143000 (mature).
    At maturity, model needs first-year info for correct comparison.
    Removing it should HURT → negative ΔLD."""
    log("=" * 60)
    log("PHASE 1: Greater-than sign flip at step 143000 (mature)")
    log("=" * 60)

    model = load_pythia_original(BASE_MODEL_160M, 143000)
    digit_tokens = get_two_digit_completion_tokens(model.tokenizer)
    valid_digits = sorted(digit_tokens.keys())
    token_ids = torch.tensor([digit_tokens[d] for d in valid_digits], device=DEVICE)

    rng = np.random.default_rng(SEED + 100)
    pairs = []
    for _ in range(GT_N_PROMPTS):
        event = rng.choice(GT_EVENTS)
        verb = rng.choice(GT_VERBS)
        y_main = int(rng.integers(3, 97))
        y_ctrl = y_main
        while abs(y_ctrl - y_main) < 10:
            y_ctrl = int(rng.integers(3, 97))
        pairs.append({
            "main": f"The {event} {verb} from the year {1700 + y_main} to the year 17",
            "ctrl": f"The {event} {verb} from the year {1700 + y_ctrl} to the year 17",
            "start_yy": y_main,
        })

    names = [f"blocks.{L}.hook_resid_post" for L in INTERVENTION_LAYERS_160M]
    base_diffs, patched_diffs = [], []

    for pr in pairs:
        m_tok = model.to_tokens(pr["main"]).to(DEVICE)
        c_tok = model.to_tokens(pr["ctrl"]).to(DEVICE)
        if m_tok.shape[1] != c_tok.shape[1]:
            continue

        diff_pos = -1
        for j in range(1, m_tok.shape[1]):
            if int(m_tok[0, j].item()) != int(c_tok[0, j].item()):
                diff_pos = j
                break
        if diff_pos < 0:
            continue

        donor = {}

        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(c_tok, fwd_hooks=[(n, make_cap(n)) for n in names])

        with torch.no_grad():
            base_logits = model(m_tok)[0, -1, :]
        base_probs = F.softmax(base_logits.float(), dim=-1)[token_ids].cpu().numpy()

        def make_patch(name, pos):
            donor_act = donor[name]
            def fn(value, hook):
                value[:, pos, :] = donor_act[:, pos, :]
                return value
            return fn

        hooks = [(n, make_patch(n, diff_pos)) for n in names]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(m_tok, fwd_hooks=hooks)[0, -1, :]
        patched_probs = F.softmax(patched_logits.float(), dim=-1)[token_ids].cpu().numpy()

        gmask = np.array([d > pr["start_yy"] for d in valid_digits])
        lmask = np.array([d <= pr["start_yy"] for d in valid_digits])
        base_diffs.append(float(base_probs[gmask].sum() - base_probs[lmask].sum()))
        patched_diffs.append(float(patched_probs[gmask].sum() - patched_probs[lmask].sum()))
        del donor

    base_arr = np.asarray(base_diffs)
    patched_arr = np.asarray(patched_diffs)
    deltas = patched_arr - base_arr
    delta_lo, delta_hi = bootstrap_ci(deltas)

    out = {
        "step": 143000,
        "task": "greater_than",
        "n_pairs": int(len(deltas)),
        "base_diff_mean": float(base_arr.mean()),
        "patched_diff_mean": float(patched_arr.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [delta_lo, delta_hi],
    }
    log(
        f"  step=143000  base P(>)-P(<=)={base_arr.mean():+.4f}  "
        f"patched={patched_arr.mean():+.4f}  "
        f"Δ={deltas.mean():+.4f} [{delta_lo:+.3f}, {delta_hi:+.3f}]"
    )
    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# ====================================================================
# PHASE 2: SVA sign flip at MATURITY
# ====================================================================

def make_sva_intervention_pairs(model, seed=SEED):
    rng = np.random.default_rng(seed)
    pairs = list(zip(SVA_SINGULAR_NOUNS, SVA_PLURAL_NOUNS))
    verb_pairs_valid = []
    for sing, plur in SVA_VERB_PAIRS:
        s_ids = model.tokenizer.encode(" " + sing, add_special_tokens=False)
        p_ids = model.tokenizer.encode(" " + plur, add_special_tokens=False)
        if len(s_ids) == 1 and len(p_ids) == 1:
            verb_pairs_valid.append({
                "singular": (sing, s_ids[0]),
                "plural": (plur, p_ids[0]),
            })

    intervention_pairs = []
    for _ in range(SVA_N_PROMPTS):
        subject_singular = bool(rng.integers(0, 2))
        i = int(rng.integers(0, len(pairs)))
        j = int(rng.integers(0, len(pairs)))
        if subject_singular:
            subj = pairs[i][0]
            attr_main = pairs[j][1]
            attr_ctrl = pairs[j][0]
        else:
            subj = pairs[i][1]
            attr_main = pairs[j][0]
            attr_ctrl = pairs[j][1]
        prep = rng.choice(SVA_PREPOSITIONS)
        vp = verb_pairs_valid[int(rng.integers(0, len(verb_pairs_valid)))]
        correct_id = vp["singular" if subject_singular else "plural"][1]
        attractor_id = vp["plural" if subject_singular else "singular"][1]
        intervention_pairs.append({
            "main": f"The {subj} {prep} the {attr_main}",
            "ctrl": f"The {subj} {prep} the {attr_ctrl}",
            "correct_id": correct_id,
            "attractor_id": attractor_id,
        })
    return intervention_pairs


def run_sva_intervention(model, step_label):
    """Run SVA causal intervention at the given model."""
    intervention_pairs = make_sva_intervention_pairs(model, SEED + 200)
    names = [f"blocks.{L}.hook_resid_post" for L in INTERVENTION_LAYERS_160M]
    base_diffs, patched_diffs = [], []
    base_log_ratios, patched_log_ratios = [], []

    for pr in intervention_pairs:
        m_tok = model.to_tokens(pr["main"]).to(DEVICE)
        c_tok = model.to_tokens(pr["ctrl"]).to(DEVICE)
        if m_tok.shape[1] != c_tok.shape[1]:
            continue
        diff_pos = -1
        for j in range(1, m_tok.shape[1]):
            if int(m_tok[0, j].item()) != int(c_tok[0, j].item()):
                diff_pos = j
                break
        if diff_pos < 0:
            continue

        donor = {}

        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(c_tok, fwd_hooks=[(n, make_cap(n)) for n in names])

        with torch.no_grad():
            base_logits = model(m_tok)[0, -1, :]
        base_probs = F.softmax(base_logits.float(), dim=-1)
        pc = float(base_probs[pr["correct_id"]].item())
        pa = float(base_probs[pr["attractor_id"]].item())
        base_diffs.append(pc - pa)
        base_log_ratios.append(float(np.log(max(pc, 1e-12)) - np.log(max(pa, 1e-12))))

        def make_patch(name, pos):
            donor_act = donor[name]
            def fn(value, hook):
                value[:, pos, :] = donor_act[:, pos, :]
                return value
            return fn

        hooks = [(n, make_patch(n, diff_pos)) for n in names]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(m_tok, fwd_hooks=hooks)[0, -1, :]
        patched_probs = F.softmax(patched_logits.float(), dim=-1)
        pc2 = float(patched_probs[pr["correct_id"]].item())
        pa2 = float(patched_probs[pr["attractor_id"]].item())
        patched_diffs.append(pc2 - pa2)
        patched_log_ratios.append(float(np.log(max(pc2, 1e-12)) - np.log(max(pa2, 1e-12))))
        del donor

    base_arr = np.asarray(base_diffs)
    patched_arr = np.asarray(patched_diffs)
    deltas = patched_arr - base_arr
    delta_lo, delta_hi = bootstrap_ci(deltas)

    # Also report log-ratio delta for better SVA comparison.
    base_lr = np.asarray(base_log_ratios)
    patched_lr = np.asarray(patched_log_ratios)
    lr_deltas = patched_lr - base_lr
    lr_lo, lr_hi = bootstrap_ci(lr_deltas)

    out = {
        "step": step_label,
        "task": "sva",
        "n_pairs": int(len(deltas)),
        "base_diff_mean": float(base_arr.mean()),
        "patched_diff_mean": float(patched_arr.mean()),
        "delta_prob_mean": float(deltas.mean()),
        "delta_prob_ci95": [delta_lo, delta_hi],
        "base_log_ratio_mean": float(base_lr.mean()),
        "patched_log_ratio_mean": float(patched_lr.mean()),
        "delta_log_ratio_mean": float(lr_deltas.mean()),
        "delta_log_ratio_ci95": [lr_lo, lr_hi],
    }
    log(
        f"  step={step_label}  "
        f"Δ_prob={deltas.mean():+.4f} [{delta_lo:+.3f}, {delta_hi:+.3f}]  "
        f"Δ_logR={lr_deltas.mean():+.4f} [{lr_lo:+.3f}, {lr_hi:+.3f}]"
    )
    return out


def phase2_sva_mature():
    log("=" * 60)
    log("PHASE 2: SVA sign flip at step 143000 (mature)")
    log("=" * 60)
    model = load_pythia_original(BASE_MODEL_160M, 143000)
    out = run_sva_intervention(model, 143000)
    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# ====================================================================
# PHASE 3: SVA intervention at step 1000 (better effect size)
# ====================================================================

def phase3_sva_step1000():
    log("=" * 60)
    log("PHASE 3: SVA intervention at step 1000 (stronger effect)")
    log("=" * 60)
    model = load_pythia_original(BASE_MODEL_160M, 1000)
    out = run_sva_intervention(model, 1000)
    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# ====================================================================
# PHASE 4 & 5: Greater-than dip at Pythia-410M and 1B
# ====================================================================

def eval_greater_than_sweep(model_name, steps):
    """Run greater-than accuracy across sparse checkpoints."""
    out = {"model": model_name, "by_step": {}}
    gt_prompts = make_greater_than_prompts(SEED)

    for step in steps:
        try:
            model = load_pythia_original(model_name, step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        digit_tokens = get_two_digit_completion_tokens(model.tokenizer)
        valid_digits = sorted(digit_tokens.keys())
        token_ids = torch.tensor([digit_tokens[d] for d in valid_digits], device=DEVICE)

        diffs, correct = [], []
        for p in gt_prompts:
            tokens = model.to_tokens(p["prompt"]).to(DEVICE)
            with torch.no_grad():
                logits = model(tokens)[0, -1, :]
            probs = F.softmax(logits.float(), dim=-1)
            dp = probs[token_ids].cpu().numpy()
            gmask = np.array([d > p["start_yy"] for d in valid_digits])
            lmask = np.array([d <= p["start_yy"] for d in valid_digits])
            pg = float(dp[gmask].sum())
            pl = float(dp[lmask].sum())
            diffs.append(pg - pl)
            correct.append(1.0 if pg > pl else 0.0)

        diffs = np.asarray(diffs)
        correct = np.asarray(correct)
        lo, hi = bootstrap_ci(diffs)

        out["by_step"][f"step_{step}"] = {
            "acc": float(correct.mean()),
            "mean_diff": float(diffs.mean()),
            "diff_ci95": [lo, hi],
        }
        marker = "  <-- DIP" if correct.mean() < 0.5 else ""
        log(
            f"  step={step:>6}  acc={correct.mean()*100:.1f}%  "
            f"P(>)-P(<=)={diffs.mean():+.4f} [{lo:+.3f}, {hi:+.3f}]{marker}"
        )

        del model; torch.cuda.empty_cache(); gc.collect()

    return out


def phase4_gt_410m():
    log("=" * 60)
    log("PHASE 4: Greater-than dip at Pythia-410M")
    log("=" * 60)
    return eval_greater_than_sweep(BASE_MODEL_410M, GT_MULTISCALE_STEPS)


def phase5_gt_1b():
    log("=" * 60)
    log("PHASE 5: Greater-than dip at Pythia-1B")
    log("=" * 60)
    return eval_greater_than_sweep(BASE_MODEL_1B, GT_MULTISCALE_STEPS)


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    results = {"config": {"seed": SEED, "n_bootstrap": N_BOOTSTRAP}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done = [k for k in results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception as e:
            log(f"Could not resume: {e}")

    phases = [
        ("phase1_gt_mature_intervention", phase1_gt_mature_intervention),
        ("phase2_sva_mature_intervention", phase2_sva_mature),
        ("phase3_sva_step1000_intervention", phase3_sva_step1000),
        ("phase4_gt_410m_dip", phase4_gt_410m),
        ("phase5_gt_1b_dip", phase5_gt_1b),
    ]

    t0 = time.time()
    for key, fn in phases:
        cached = results.get(key)
        if cached is not None and "error" not in cached:
            log(f"SKIP {key}: already done")
            continue
        if cached is not None and "error" in cached:
            log(f"RETRY {key}: previous error ({cached['error'][:60]})")
        log(f"START {key}")
        try:
            results[key] = fn()
            save_results(results)
        except Exception as e:
            log(f"FAILED {key}: {e}")
            traceback.print_exc()
            results[key] = {"error": str(e)}
            save_results(results)

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
