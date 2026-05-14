"""
EMNLP Task Generalization Battery (Script B)
=============================================

PRE-REGISTERED HYPOTHESIS
-------------------------
A below-chance training dip occurs in tasks where:
  (a) the task requires >= 2 coordinated components,
  (b) at least one component (the "shortcut") is detectable by simple
      statistical features available early in training,
  (c) acting on the shortcut alone produces a SYSTEMATIC wrong answer
      (not random),
  (d) the "correcter" component requires longer to form.

Single-component tasks (e.g., next-token prediction on natural text)
should NOT show the dip.

PREDICTIONS (pre-registered before running)
-------------------------------------------
1. Greater-than (Hanna et al. 2023):
   Shortcut: echo first year's last two digits.
   Misuse:    predict completion <= start year (wrong by greater-than).
   Correcter: numerical comparison.
   ==> Predict YES dip (P(>) - P(<=) goes negative at some point).

2. Subject-verb agreement with attractor (Linzen-style):
   Shortcut: agree verb with most recent noun (the attractor).
   Misuse:    verb disagrees with the syntactic subject.
   Correcter: syntactic subject identification.
   ==> Predict YES dip (P(correct) - P(attractor) goes negative).

3. Natural sentence completion (negative control):
   No two-component structure; just next-token prediction.
   ==> Predict NO dip; mean NLL decreases monotonically.

If 2/2 positive predictions show dips, AND control shows no dip, the
dip framework is supported as a general principle of circuit formation,
not an IOI-specific phenomenon.

PHASES
------
PHASE 1: Greater-than accuracy across Pythia checkpoints (dip detection).
PHASE 2: SVA accuracy across same checkpoints.
PHASE 3: Natural sentence NLL across same checkpoints (negative control).
PHASE 4: If greater-than dip detected, causal intervention at dip floor.
PHASE 5: If SVA dip detected, causal intervention at dip floor.

Output: results/emnlp_task_generalization.json
Log:    results/emnlp_task_generalization_log.txt
Runtime: ~30-45 minutes on A100.
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


# ----------------------------- Config -----------------------------

DEVICE = "cuda"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"
SEED = 42
N_BOOTSTRAP = 10_000

# Original Pythia revisions, span pre-training to end.
TASK_STEPS = [0, 256, 512, 1000, 2000, 3000, 5000, 8000, 10000, 50000, 143000]

# Greater-than config
GT_EVENTS = ["war", "battle", "dispute", "conflict", "argument", "siege"]
GT_VERBS = ["lasted", "ran", "extended", "continued", "stretched"]
GT_YEAR_LO = 1702
GT_YEAR_HI = 1798
GT_N_PROMPTS = 300

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
# Pairs (singular_verb, plural_verb). Both must be single tokens with
# leading space in Pythia tokenizer.
SVA_VERB_PAIRS = [("is", "are"), ("was", "were"), ("has", "have")]
SVA_N_PROMPTS = 200

# Negative control config — natural sentences whose next-token
# perplexity should decrease monotonically across training. Picked to
# be high-frequency factoids whose continuations are unambiguous.
LM_SENTENCES = [
    "The capital of France is Paris.",
    "The sun rises in the east and sets in the west.",
    "Water freezes at zero degrees Celsius.",
    "Shakespeare wrote Romeo and Juliet.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Mount Everest is the tallest mountain in the world.",
    "The human body has two hundred and six bones.",
    "Light travels faster than sound.",
    "The Great Wall of China is in China.",
    "A year on Earth has twelve months.",
]

# Causal intervention config
INTERVENTION_LAYERS = [3, 4, 5]
INTERVENTION_PPT = 30

RESULTS_PATH = "results/emnlp_task_generalization.json"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pythia_original(step):
    hf = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=f"step{step}", torch_dtype=torch.float32,
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


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.shape[0], size=(n_resamples, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ====================================================================
# PHASE 1: GREATER-THAN
# ====================================================================

def get_two_digit_completion_tokens(tokenizer):
    """Return dict {digit_value: token_id} for two-digit strings that
    tokenize to a single token in the Pythia BPE. Pythia's tokenizer
    is byte-level GPT-NeoX BPE; many 2-digit strings tokenize as one
    token but not all do. We filter to single-token completions for
    fair comparison.
    """
    mapping = {}
    for d in range(100):
        s = f"{d:02d}"
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            mapping[d] = ids[0]
    return mapping


def make_greater_than_prompts(seed=SEED):
    """Generate prompts: 'The {event} {verb} from the year 17{Y1Y2} to
    the year 17'. The model's continuation should be a 2-digit number
    Y > Y1Y2.
    """
    rng = np.random.default_rng(seed)
    prompts = []
    for _ in range(GT_N_PROMPTS):
        event = rng.choice(GT_EVENTS)
        verb = rng.choice(GT_VERBS)
        # Sample Y1Y2 such that both sides of the split are non-trivial.
        y_low_2 = int(rng.integers(GT_YEAR_LO % 100 + 1, GT_YEAR_HI % 100))
        year = 1700 + y_low_2
        prompt = f"The {event} {verb} from the year {year} to the year 17"
        prompts.append({"prompt": prompt, "start_yy": y_low_2})
    return prompts


def eval_greater_than(model, prompts, digit_tokens):
    """Per prompt, compute P(>) - P(<=) over single-token 2-digit
    completions. Returns numpy array of per-prompt scores plus the
    fraction-correct (P(>) > P(<=)) which is the dip-detection metric.
    """
    valid_digits = sorted(digit_tokens.keys())
    token_ids = torch.tensor([digit_tokens[d] for d in valid_digits], device=DEVICE)

    per_prompt_diff = []
    per_prompt_p_greater = []
    per_prompt_p_lesseq = []
    per_prompt_correct = []  # 1 if P(>) > P(<=)

    for p in prompts:
        tokens = model.to_tokens(p["prompt"]).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]  # final-position logits
        probs = F.softmax(logits.float(), dim=-1)
        digit_probs = probs[token_ids].cpu().numpy()

        # Sum probability over completions > start_yy and <= start_yy.
        greater_mask = np.array([d > p["start_yy"] for d in valid_digits])
        lesseq_mask = np.array([d <= p["start_yy"] for d in valid_digits])
        p_greater = float(digit_probs[greater_mask].sum())
        p_lesseq = float(digit_probs[lesseq_mask].sum())

        per_prompt_p_greater.append(p_greater)
        per_prompt_p_lesseq.append(p_lesseq)
        per_prompt_diff.append(p_greater - p_lesseq)
        per_prompt_correct.append(1.0 if p_greater > p_lesseq else 0.0)

    return {
        "per_prompt_diff": per_prompt_diff,
        "per_prompt_p_greater": per_prompt_p_greater,
        "per_prompt_p_lesseq": per_prompt_p_lesseq,
        "per_prompt_correct": per_prompt_correct,
    }


def phase1_greater_than():
    log("=" * 60)
    log("PHASE 1: Greater-than (Hanna et al. 2023 format)")
    log("=" * 60)
    out = {"by_step": {}, "n_prompts": GT_N_PROMPTS}

    prompts = make_greater_than_prompts(SEED)

    for step in TASK_STEPS:
        try:
            model = load_pythia_original(step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        digit_tokens = get_two_digit_completion_tokens(model.tokenizer)
        log(f"  step {step}: {len(digit_tokens)}/100 two-digit completions are single tokens")

        r = eval_greater_than(model, prompts, digit_tokens)
        diffs = np.asarray(r["per_prompt_diff"])
        correct = np.asarray(r["per_prompt_correct"])
        lo, hi = bootstrap_ci(diffs)

        out["by_step"][f"step_{step}"] = {
            "n": int(len(diffs)),
            "acc_p_greater_beats_lesseq": float(correct.mean()),
            "mean_diff": float(diffs.mean()),
            "diff_ci95": [lo, hi],
            "mean_p_greater": float(np.mean(r["per_prompt_p_greater"])),
            "mean_p_lesseq": float(np.mean(r["per_prompt_p_lesseq"])),
        }
        log(
            f"  step={step}  acc={correct.mean()*100:.1f}%  "
            f"P(>)-P(<=)={diffs.mean():+.4f}  CI=[{lo:+.3f}, {hi:+.3f}]"
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Identify dip step: lowest accuracy across training.
    accs = [(s, r["acc_p_greater_beats_lesseq"]) for s, r in out["by_step"].items()]
    accs.sort(key=lambda x: x[1])
    out["dip_step_key"] = accs[0][0] if accs else None
    out["dip_min_acc"] = accs[0][1] if accs else None
    log(f"  Lowest accuracy: {out['dip_step_key']} at {out['dip_min_acc']*100:.1f}%")
    return out


# ====================================================================
# PHASE 2: SUBJECT-VERB AGREEMENT
# ====================================================================

def make_sva_prompts(model, seed=SEED):
    """Generate SVA prompts with attractor. Each prompt has subject of
    one number (singular/plural) and an attractor of opposite number,
    plus a singular/plural verb pair to score."""
    rng = np.random.default_rng(seed)
    prompts = []

    # Verify each verb pair has both forms as single tokens with leading space.
    verb_pairs_valid = []
    for sing, plur in SVA_VERB_PAIRS:
        sing_ids = model.tokenizer.encode(" " + sing, add_special_tokens=False)
        plur_ids = model.tokenizer.encode(" " + plur, add_special_tokens=False)
        if len(sing_ids) == 1 and len(plur_ids) == 1:
            verb_pairs_valid.append({
                "singular": (sing, sing_ids[0]),
                "plural": (plur, plur_ids[0]),
            })
    if not verb_pairs_valid:
        raise RuntimeError("No valid SVA verb pairs found in tokenizer.")

    pairs = list(zip(SVA_SINGULAR_NOUNS, SVA_PLURAL_NOUNS))
    for _ in range(SVA_N_PROMPTS):
        # Half the time subject is singular, attractor plural; otherwise reversed.
        subject_singular = bool(rng.integers(0, 2))
        sing_idx = int(rng.integers(0, len(pairs)))
        attr_idx = int(rng.integers(0, len(pairs)))
        if subject_singular:
            subj = pairs[sing_idx][0]   # singular noun
            attr = pairs[attr_idx][1]   # plural noun (attractor)
        else:
            subj = pairs[sing_idx][1]   # plural noun (subject)
            attr = pairs[attr_idx][0]   # singular noun (attractor)

        prep = rng.choice(SVA_PREPOSITIONS)
        verb_pair = verb_pairs_valid[int(rng.integers(0, len(verb_pairs_valid)))]
        correct_form = "singular" if subject_singular else "plural"
        attractor_form = "plural" if subject_singular else "singular"
        correct_word, correct_id = verb_pair[correct_form]
        attractor_word, attractor_id = verb_pair[attractor_form]

        prompt = f"The {subj} {prep} the {attr}"
        prompts.append({
            "prompt": prompt,
            "subject_singular": subject_singular,
            "correct_token_id": correct_id,
            "attractor_token_id": attractor_id,
            "correct_word": correct_word,
            "attractor_word": attractor_word,
        })
    return prompts


def eval_sva(model, prompts):
    """Compute per-prompt P(correct verb) and P(attractor verb) at the
    final position."""
    per_prompt_diff = []
    per_prompt_log_ratio = []
    per_prompt_correct = []

    for p in prompts:
        tokens = model.to_tokens(p["prompt"]).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        p_correct = float(probs[p["correct_token_id"]].item())
        p_attractor = float(probs[p["attractor_token_id"]].item())

        per_prompt_diff.append(p_correct - p_attractor)
        # Log ratio is the standard SVA metric in the literature.
        ratio = np.log(max(p_correct, 1e-12)) - np.log(max(p_attractor, 1e-12))
        per_prompt_log_ratio.append(float(ratio))
        per_prompt_correct.append(1.0 if p_correct > p_attractor else 0.0)

    return {
        "per_prompt_diff": per_prompt_diff,
        "per_prompt_log_ratio": per_prompt_log_ratio,
        "per_prompt_correct": per_prompt_correct,
    }


def phase2_sva():
    log("=" * 60)
    log("PHASE 2: Subject-verb agreement with attractor")
    log("=" * 60)
    out = {"by_step": {}, "n_prompts": SVA_N_PROMPTS}

    # Need a tokenizer; pull one before main loop to construct prompts.
    bootstrap_model = load_pythia_original(0)
    prompts = make_sva_prompts(bootstrap_model, SEED)
    log(f"  Generated {len(prompts)} SVA prompts")
    del bootstrap_model
    torch.cuda.empty_cache()

    for step in TASK_STEPS:
        try:
            model = load_pythia_original(step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        r = eval_sva(model, prompts)
        diffs = np.asarray(r["per_prompt_diff"])
        log_ratio = np.asarray(r["per_prompt_log_ratio"])
        correct = np.asarray(r["per_prompt_correct"])
        diff_lo, diff_hi = bootstrap_ci(diffs)
        lr_lo, lr_hi = bootstrap_ci(log_ratio)

        out["by_step"][f"step_{step}"] = {
            "n": int(len(diffs)),
            "acc": float(correct.mean()),
            "mean_diff": float(diffs.mean()),
            "diff_ci95": [diff_lo, diff_hi],
            "mean_log_ratio": float(log_ratio.mean()),
            "log_ratio_ci95": [lr_lo, lr_hi],
        }
        log(
            f"  step={step}  acc={correct.mean()*100:.1f}%  "
            f"logP(correct)-logP(attractor)={log_ratio.mean():+.4f} "
            f"[{lr_lo:+.3f}, {lr_hi:+.3f}]"
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    accs = [(s, r["acc"]) for s, r in out["by_step"].items()]
    accs.sort(key=lambda x: x[1])
    out["dip_step_key"] = accs[0][0] if accs else None
    out["dip_min_acc"] = accs[0][1] if accs else None
    log(f"  Lowest accuracy: {out['dip_step_key']} at {out['dip_min_acc']*100:.1f}%")
    return out


# ====================================================================
# PHASE 3: NEGATIVE CONTROL (NATURAL SENTENCE NLL)
# ====================================================================

def eval_natural_nll(model, sentences):
    """Mean negative log-likelihood per token across the sentences,
    excluding the first token of each (which has no context)."""
    losses = []
    for s in sentences:
        tokens = model.to_tokens(s).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)
        # Standard LM loss: predict token[i+1] from position i.
        targets = tokens[0, 1:]
        log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
        nll = -log_probs[torch.arange(targets.shape[0]), targets]
        losses.extend(nll.cpu().numpy().tolist())
    return float(np.mean(losses)), losses


def phase3_natural_control():
    log("=" * 60)
    log("PHASE 3: Natural sentence NLL (negative control)")
    log("=" * 60)
    out = {"by_step": {}, "n_sentences": len(LM_SENTENCES)}

    for step in TASK_STEPS:
        try:
            model = load_pythia_original(step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        mean_nll, losses = eval_natural_nll(model, LM_SENTENCES)
        lo, hi = bootstrap_ci(losses)
        out["by_step"][f"step_{step}"] = {
            "n_tokens": int(len(losses)),
            "mean_nll": mean_nll,
            "nll_ci95": [lo, hi],
            "perplexity": float(np.exp(mean_nll)),
        }
        log(
            f"  step={step}  NLL={mean_nll:.4f}  "
            f"PPL={np.exp(mean_nll):.2f}  "
            f"CI=[{lo:.3f}, {hi:.3f}]"
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# PHASE 4 & 5: CONDITIONAL CAUSAL INTERVENTIONS
# ====================================================================

def find_token_position(tokens, target_substring, model):
    """Return the position of the first token in tokens that begins
    target_substring, or -1 if not found. Used for locating the start
    year in greater-than prompts."""
    # Decode each prefix and look for target_substring boundary.
    seq = tokens[0].cpu().tolist()
    for i in range(1, len(seq)):
        decoded = model.tokenizer.decode(seq[i:i+1])
        if target_substring in decoded:
            return i
    return -1


def phase4_greater_than_intervention(phase1_out):
    """If greater-than showed a dip, patch the first year's residual at
    layers 3-5 with a control prompt that has a different first year.
    Tests whether removing the first-year information at the dip floor
    recovers accuracy (analog of S2 patching for IOI)."""
    log("=" * 60)
    log("PHASE 4: Causal intervention on greater-than at dip floor")
    log("=" * 60)

    if phase1_out is None or "dip_step_key" not in phase1_out:
        log("  No greater-than data; skipping")
        return None
    dip_step_key = phase1_out["dip_step_key"]
    dip_step = int(dip_step_key.split("_")[1])
    dip_acc = phase1_out["dip_min_acc"]

    if dip_acc > 0.45:
        # No clear dip; intervention isn't motivated.
        log(f"  Dip min acc = {dip_acc*100:.1f}% > 45%; no dip, skipping intervention")
        return {"skipped_reason": "no dip detected", "dip_min_acc": dip_acc}

    log(f"  Dip detected at {dip_step_key} (acc={dip_acc*100:.1f}%)")
    log(f"  Loading model at dip floor (step {dip_step})")
    model = load_pythia_original(dip_step)
    digit_tokens = get_two_digit_completion_tokens(model.tokenizer)

    # Generate matched IOI-style pairs: a prompt and a control with a
    # different first year. The control answers DIFFERENTLY (since the
    # threshold for ">" is different), so we measure P(>) relative to
    # the IOI's start_yy.
    rng = np.random.default_rng(SEED + 100)
    pairs = []
    for _ in range(GT_N_PROMPTS):
        event = rng.choice(GT_EVENTS)
        verb = rng.choice(GT_VERBS)
        y_main = int(rng.integers(GT_YEAR_LO % 100 + 1, GT_YEAR_HI % 100))
        y_ctrl = y_main
        while abs(y_ctrl - y_main) < 10:
            y_ctrl = int(rng.integers(GT_YEAR_LO % 100 + 1, GT_YEAR_HI % 100))
        prompts = (
            f"The {event} {verb} from the year {1700 + y_main} to the year 17",
            f"The {event} {verb} from the year {1700 + y_ctrl} to the year 17",
        )
        pairs.append({"main": prompts[0], "ctrl": prompts[1], "start_yy": y_main})

    # We patch the residual stream at the YEAR TOKEN position of the
    # main prompt with the corresponding residual from the control.
    # The year token is the token corresponding to the first year's
    # tokenization. For "1732", we look for the position right after
    # the verb ends. Simplest approach: tokenize both prompts and find
    # the position where they first DIFFER.

    valid_digits = sorted(digit_tokens.keys())
    token_ids = torch.tensor([digit_tokens[d] for d in valid_digits], device=DEVICE)

    base_diffs = []
    patched_diffs = []

    for pr in pairs:
        m_tok = model.to_tokens(pr["main"]).to(DEVICE)
        c_tok = model.to_tokens(pr["ctrl"]).to(DEVICE)
        if m_tok.shape[1] != c_tok.shape[1]:
            # Token-length mismatch; skip this pair.
            continue
        # Locate the first differing position (this is the year token).
        diff_pos = -1
        for j in range(1, m_tok.shape[1]):
            if int(m_tok[0, j].item()) != int(c_tok[0, j].item()):
                diff_pos = j
                break
        if diff_pos < 0:
            continue

        # Cache control residuals at patch layers.
        names = [f"blocks.{L}.hook_resid_post" for L in INTERVENTION_LAYERS]
        donor = {}

        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(c_tok, fwd_hooks=[(n, make_cap(n)) for n in names])

        # Baseline.
        with torch.no_grad():
            base_logits = model(m_tok)[0, -1, :]
        base_probs = F.softmax(base_logits.float(), dim=-1)[token_ids].cpu().numpy()

        # Patched: replace residual at diff_pos for layers 3-5.
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

        greater_mask = np.array([d > pr["start_yy"] for d in valid_digits])
        lesseq_mask = np.array([d <= pr["start_yy"] for d in valid_digits])
        base_diffs.append(float(base_probs[greater_mask].sum() - base_probs[lesseq_mask].sum()))
        patched_diffs.append(float(patched_probs[greater_mask].sum() - patched_probs[lesseq_mask].sum()))

        del donor

    base_arr = np.asarray(base_diffs)
    patched_arr = np.asarray(patched_diffs)
    deltas = patched_arr - base_arr
    delta_lo, delta_hi = bootstrap_ci(deltas)

    out = {
        "step": dip_step,
        "n_pairs": int(len(deltas)),
        "base_diff_mean": float(base_arr.mean()),
        "patched_diff_mean": float(patched_arr.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [delta_lo, delta_hi],
    }
    log(
        f"  dip step={dip_step}  base P(>)-P(<=)={base_arr.mean():+.4f}  "
        f"patched={patched_arr.mean():+.4f}  Δ={deltas.mean():+.4f} "
        f"[{delta_lo:+.3f}, {delta_hi:+.3f}]"
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


def phase5_sva_intervention(phase2_out):
    """If SVA showed a dip, patch the ATTRACTOR noun's residual stream
    with a control where the attractor has the same number as the
    subject. Tests whether removing the attractor's wrong-number signal
    at the dip floor recovers accuracy."""
    log("=" * 60)
    log("PHASE 5: Causal intervention on SVA at dip floor")
    log("=" * 60)

    if phase2_out is None or "dip_step_key" not in phase2_out:
        log("  No SVA data; skipping")
        return None
    dip_step_key = phase2_out["dip_step_key"]
    dip_step = int(dip_step_key.split("_")[1])
    dip_acc = phase2_out["dip_min_acc"]

    if dip_acc > 0.45:
        log(f"  Dip min acc = {dip_acc*100:.1f}% > 45%; no dip, skipping intervention")
        return {"skipped_reason": "no dip detected", "dip_min_acc": dip_acc}

    log(f"  Dip detected at {dip_step_key} (acc={dip_acc*100:.1f}%)")
    log(f"  Loading model at dip floor (step {dip_step})")
    model = load_pythia_original(dip_step)

    # Generate matched pairs: main = "The {subj_sing} near the {attr_plur}"
    # ctrl = "The {subj_sing} near the {attr_sing}". Same subject, but
    # attractor matches subject's number. Patching the attractor noun
    # position in main with ctrl's representation removes the
    # wrong-number signal.
    rng = np.random.default_rng(SEED + 200)
    pairs = list(zip(SVA_SINGULAR_NOUNS, SVA_PLURAL_NOUNS))

    # Filter verb pairs to single-token forms.
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
            subj = pairs[i][0]                # singular subject
            attr_main = pairs[j][1]           # plural attractor (wrong number)
            attr_ctrl = pairs[j][0]           # singular attractor (matches subject)
        else:
            subj = pairs[i][1]                # plural subject
            attr_main = pairs[j][0]           # singular attractor (wrong number)
            attr_ctrl = pairs[j][1]           # plural attractor (matches subject)

        prep = rng.choice(SVA_PREPOSITIONS)
        verb_pair = verb_pairs_valid[int(rng.integers(0, len(verb_pairs_valid)))]
        correct_id = verb_pair["singular" if subject_singular else "plural"][1]
        attractor_id = verb_pair["plural" if subject_singular else "singular"][1]

        main_prompt = f"The {subj} {prep} the {attr_main}"
        ctrl_prompt = f"The {subj} {prep} the {attr_ctrl}"
        intervention_pairs.append({
            "main": main_prompt, "ctrl": ctrl_prompt,
            "correct_id": correct_id, "attractor_id": attractor_id,
        })

    base_diffs = []
    patched_diffs = []
    for pr in intervention_pairs:
        m_tok = model.to_tokens(pr["main"]).to(DEVICE)
        c_tok = model.to_tokens(pr["ctrl"]).to(DEVICE)
        if m_tok.shape[1] != c_tok.shape[1]:
            continue
        # Find attractor noun position: first differing token.
        diff_pos = -1
        for j in range(1, m_tok.shape[1]):
            if int(m_tok[0, j].item()) != int(c_tok[0, j].item()):
                diff_pos = j
                break
        if diff_pos < 0:
            continue

        names = [f"blocks.{L}.hook_resid_post" for L in INTERVENTION_LAYERS]
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

        base_diff = float(base_probs[pr["correct_id"]].item() - base_probs[pr["attractor_id"]].item())
        patched_diff = float(patched_probs[pr["correct_id"]].item() - patched_probs[pr["attractor_id"]].item())
        base_diffs.append(base_diff)
        patched_diffs.append(patched_diff)

        del donor

    base_arr = np.asarray(base_diffs)
    patched_arr = np.asarray(patched_diffs)
    deltas = patched_arr - base_arr
    delta_lo, delta_hi = bootstrap_ci(deltas)

    out = {
        "step": dip_step,
        "n_pairs": int(len(deltas)),
        "base_diff_mean": float(base_arr.mean()),
        "patched_diff_mean": float(patched_arr.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [delta_lo, delta_hi],
    }
    log(
        f"  dip step={dip_step}  base P(correct)-P(attractor)={base_arr.mean():+.4f}  "
        f"patched={patched_arr.mean():+.4f}  Δ={deltas.mean():+.4f} "
        f"[{delta_lo:+.3f}, {delta_hi:+.3f}]"
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)

    results = {"config": {
        "model": BASE_MODEL,
        "seed": SEED,
        "steps": TASK_STEPS,
        "n_bootstrap": N_BOOTSTRAP,
    }}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done = [k for k in results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception as e:
            log(f"Could not resume: {e}")

    t0 = time.time()

    # Phase 1.
    cached = results.get("phase1_greater_than")
    if cached is not None and "error" not in cached:
        log("SKIP phase1_greater_than: already done")
    else:
        log(f"START phase1_greater_than")
        try:
            results["phase1_greater_than"] = phase1_greater_than()
            save_results(results)
        except Exception as e:
            log(f"FAILED phase1: {e}")
            traceback.print_exc()
            results["phase1_greater_than"] = {"error": str(e)}
            save_results(results)

    # Phase 2.
    cached = results.get("phase2_sva")
    if cached is not None and "error" not in cached:
        log("SKIP phase2_sva: already done")
    else:
        log(f"START phase2_sva")
        try:
            results["phase2_sva"] = phase2_sva()
            save_results(results)
        except Exception as e:
            log(f"FAILED phase2: {e}")
            traceback.print_exc()
            results["phase2_sva"] = {"error": str(e)}
            save_results(results)

    # Phase 3.
    cached = results.get("phase3_natural_control")
    if cached is not None and "error" not in cached:
        log("SKIP phase3_natural_control: already done")
    else:
        log(f"START phase3_natural_control")
        try:
            results["phase3_natural_control"] = phase3_natural_control()
            save_results(results)
        except Exception as e:
            log(f"FAILED phase3: {e}")
            traceback.print_exc()
            results["phase3_natural_control"] = {"error": str(e)}
            save_results(results)

    # Phase 4 (conditional on phase 1 dip).
    cached = results.get("phase4_greater_than_intervention")
    if cached is not None and "error" not in cached:
        log("SKIP phase4_greater_than_intervention: already done")
    else:
        log(f"START phase4_greater_than_intervention")
        try:
            results["phase4_greater_than_intervention"] = phase4_greater_than_intervention(
                results.get("phase1_greater_than"),
            )
            save_results(results)
        except Exception as e:
            log(f"FAILED phase4: {e}")
            traceback.print_exc()
            results["phase4_greater_than_intervention"] = {"error": str(e)}
            save_results(results)

    # Phase 5 (conditional on phase 2 dip).
    cached = results.get("phase5_sva_intervention")
    if cached is not None and "error" not in cached:
        log("SKIP phase5_sva_intervention: already done")
    else:
        log(f"START phase5_sva_intervention")
        try:
            results["phase5_sva_intervention"] = phase5_sva_intervention(
                results.get("phase2_sva"),
            )
            save_results(results)
        except Exception as e:
            log(f"FAILED phase5: {e}")
            traceback.print_exc()
            results["phase5_sva_intervention"] = {"error": str(e)}
            save_results(results)

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
