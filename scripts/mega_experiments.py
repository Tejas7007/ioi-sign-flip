"""
MEGA EXPERIMENT SCRIPT - Answers all of Cole's questions.
Run overnight on Vast.ai with nohup.

Experiments:
  A: L0H10 output projection (what is it doing at step 1000 vs 3000?)
  B: Full circuit component tracking across training (all Wang et al. components)
  C: Ablate dominant head on Pile (does synthetic circuit = natural circuit?)
  D: Logit difference distribution at step 1000 (systematic S preference?)
  E: Attention patterns of early NMs (what are they attending to?)

Saves incrementally to results/mega_experiments.json
"""

import torch
import json
import os
import shutil
import time
import traceback
import numpy as np

from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except ImportError:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

# ============================================================
# CONFIG
# ============================================================

MODELS = [
    ("EleutherAI/pythia-160m-deduped", 160),
    ("EleutherAI/pythia-410m-deduped", 410),
    ("EleutherAI/pythia-1b-deduped", 1000),
]

CHECKPOINTS = [0, 512, 1000, 2000, 3000, 4000, 5000, 8000,
               10000, 16000, 33000, 66000, 143000]

# Dominant heads at step 143000 for each model
DOMINANT_HEADS = {
    "EleutherAI/pythia-160m-deduped": (8, 9),    # L8H9
    "EleutherAI/pythia-410m-deduped": (4, 6),     # L4H6
    "EleutherAI/pythia-1b-deduped": (11, 0),      # L11H0
}

# Early NM heads for projection analysis
EARLY_NMS_160M = [(0, 5), (0, 6), (0, 10)]

TEMPLATES = ALL_TEMPLATES[:15]
PPT = 20
SEED = 42
TAU = 0.02

RESULTS_FILE = "results/mega_experiments.json"

# ============================================================
# UTILITIES
# ============================================================

def clear_cache():
    cache_dir = "/workspace/.hf_home/hub"
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith("models--"):
                shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)


def load_model(model_name, step):
    clear_cache()
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
        device="cuda",
        checkpoint_value=step,
    )
    return model


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  [saved %s]" % RESULTS_FILE)


def get_ioi_data(model, n_templates=15, ppt=20, seed=42):
    """Generate IOI data and return tokens, positions, etc."""
    all_tokens = []
    all_io_ids = []
    all_s_ids = []
    all_io_positions = []
    all_s1_positions = []
    all_s2_positions = []

    for tmpl in TEMPLATES[:n_templates]:
        ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl],
                        symmetric=True, seed=seed)
        tokens = model.to_tokens(ds.prompts).cuda()
        io_ids = torch.tensor(ds.io_token_ids, device="cuda")
        s_ids = torch.tensor(ds.s_token_ids, device="cuda")

        # Find positions of IO and S tokens in each prompt
        for i in range(tokens.shape[0]):
            io_tok = io_ids[i].item()
            s_tok = s_ids[i].item()

            # Find positions (skip BOS at position 0)
            io_pos = -1
            s1_pos = -1
            s2_pos = -1
            s_count = 0

            for j in range(1, tokens.shape[1]):
                if tokens[i, j].item() == io_tok and io_pos == -1:
                    io_pos = j
                if tokens[i, j].item() == s_tok:
                    s_count += 1
                    if s_count == 1:
                        s1_pos = j
                    elif s_count == 2:
                        s2_pos = j

            all_io_positions.append(io_pos)
            all_s1_positions.append(s1_pos)
            all_s2_positions.append(s2_pos)

        all_tokens.append(tokens)
        all_io_ids.append(io_ids)
        all_s_ids.append(s_ids)

    return {
        "tokens_list": all_tokens,
        "io_ids_list": all_io_ids,
        "s_ids_list": all_s_ids,
        "io_positions": all_io_positions,
        "s1_positions": all_s1_positions,
        "s2_positions": all_s2_positions,
    }


# ============================================================
# EXPERIMENT A: L0H10 Output Projection
# What is L0H10 writing at step 1000 vs 3000?
# ============================================================

def run_experiment_a(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT A: L0H10 Output Projection")
    print("=" * 60)

    results["exp_a_output_projection"] = {}

    model_name = "EleutherAI/pythia-160m-deduped"

    for step in [1000, 2000, 3000, 143000]:
        print("\n--- Step %d ---" % step)
        try:
            model = load_model(model_name, step)
        except Exception as e:
            print("  FAILED to load: %s" % str(e))
            continue

        W_U = model.W_U  # [d_model, d_vocab]

        # Collect head outputs for early NM heads
        head_results = {}

        for layer, head in EARLY_NMS_160M:
            io_projections = []
            s_projections = []
            other_projections = []

            for tmpl in TEMPLATES[:10]:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).cuda()
                io_ids = torch.tensor(ds.io_token_ids, device="cuda")
                s_ids = torch.tensor(ds.s_token_ids, device="cuda")

                # Run with cache to get head output
                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                # Head output at final token position
                # hook_z shape: [batch, pos, n_heads, d_head]
                z = cache["blocks.%d.attn.hook_result" % layer]
                # z shape: [batch, pos, n_heads, d_model]
                head_output = z[:, -1, head, :]  # [batch, d_model]

                # Project onto IO and S directions in unembedding space
                for i in range(len(io_ids)):
                    io_tok = io_ids[i].item()
                    s_tok = s_ids[i].item()

                    io_dir = W_U[:, io_tok]  # [d_model]
                    s_dir = W_U[:, s_tok]    # [d_model]

                    io_proj = torch.dot(head_output[i], io_dir).item()
                    s_proj = torch.dot(head_output[i], s_dir).item()

                    io_projections.append(io_proj)
                    s_projections.append(s_proj)

                del cache
                torch.cuda.empty_cache()

            mean_io = float(np.mean(io_projections))
            mean_s = float(np.mean(s_projections))
            diff = mean_io - mean_s

            head_name = "L%dH%d" % (layer, head)
            head_results[head_name] = {
                "mean_io_projection": round(mean_io, 4),
                "mean_s_projection": round(mean_s, 4),
                "io_minus_s": round(diff, 4),
                "promotes": "IO" if diff > 0 else "S",
                "n_examples": len(io_projections),
            }
            print("  %s: IO=%.4f, S=%.4f, diff=%.4f -> promotes %s" % (
                head_name, mean_io, mean_s, diff,
                "IO" if diff > 0 else "S"))

        results["exp_a_output_projection"]["step_%d" % step] = head_results
        save_results(results)

        del model
        torch.cuda.empty_cache()

    print("\nEXPERIMENT A COMPLETE")


# ============================================================
# EXPERIMENT B: Full Circuit Component Tracking
# Track ALL Wang et al. components across training
# ============================================================

def classify_head_full(model, layer, head, tokens_list, io_ids_list, s_ids_list,
                       io_positions, s1_positions, s2_positions, tau=0.02):
    """
    Classify a single head using multiple metrics:
    - delta_ioi: effect on IO-S logit diff (name mover metric)
    - delta_s_logit: effect on S logit (S-inhibition metric)
    - attn_s2_to_s1: attention from S2 pos to S1 pos (duplicate token metric)
    - attn_prev_token: avg attention to pos-1 (previous token metric)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Compute baseline and ablated metrics
    base_io_logits = []
    base_s_logits = []
    abl_io_logits = []
    abl_s_logits = []

    # Attention pattern metrics
    attn_s2_to_s1_scores = []
    attn_prev_scores = []

    def hook_fn(value, hook, h=head):
        value[:, :, h, :] = 0.0
        return value

    example_idx = 0
    for t_idx in range(len(tokens_list)):
        tokens = tokens_list[t_idx]
        io_ids = io_ids_list[t_idx]
        s_ids = s_ids_list[t_idx]
        batch_size = tokens.shape[0]

        # Baseline run with attention cache
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        base_logits = model(tokens)
        base_last = base_logits[:, -1, :]

        for i in range(batch_size):
            base_io_logits.append(base_last[i, io_ids[i]].item())
            base_s_logits.append(base_last[i, s_ids[i]].item())

        # Get attention pattern for this head
        attn_key = "blocks.%d.attn.hook_pattern" % layer
        if attn_key in cache:
            attn = cache[attn_key]  # [batch, n_heads, dest, src]
            for i in range(batch_size):
                idx = example_idx + i
                if idx < len(s2_positions):
                    s2_pos = s2_positions[idx]
                    s1_pos = s1_positions[idx]
                    if s2_pos > 0 and s1_pos > 0 and s2_pos < attn.shape[2] and s1_pos < attn.shape[3]:
                        attn_s2_to_s1_scores.append(
                            attn[i, head, s2_pos, s1_pos].item())

                # Previous token attention (average across positions)
                seq_len = tokens.shape[1]
                prev_attn = 0.0
                count = 0
                for pos in range(1, min(seq_len, attn.shape[2])):
                    if pos < attn.shape[3]:
                        prev_attn += attn[i, head, pos, pos - 1].item()
                        count += 1
                if count > 0:
                    attn_prev_scores.append(prev_attn / count)

        del cache
        torch.cuda.empty_cache()

        # Ablated run
        hook = ("blocks.%d.attn.hook_z" % layer, hook_fn)
        abl_logits = model.run_with_hooks(tokens, fwd_hooks=[hook])
        abl_last = abl_logits[:, -1, :]

        for i in range(batch_size):
            abl_io_logits.append(abl_last[i, io_ids[i]].item())
            abl_s_logits.append(abl_last[i, s_ids[i]].item())

        example_idx += batch_size

    # Compute metrics
    base_io = np.mean(base_io_logits)
    base_s = np.mean(base_s_logits)
    abl_io = np.mean(abl_io_logits)
    abl_s = np.mean(abl_s_logits)

    base_ld = base_io - base_s
    abl_ld = abl_io - abl_s

    delta_ioi = abl_ld - base_ld          # negative = head helps IOI
    delta_s_logit = abl_s - base_s        # positive = head was suppressing S
    delta_io_logit = abl_io - base_io     # negative = head was promoting IO

    attn_s2_s1 = float(np.mean(attn_s2_to_s1_scores)) if attn_s2_to_s1_scores else 0.0
    attn_prev = float(np.mean(attn_prev_scores)) if attn_prev_scores else 0.0

    # Classification
    roles = []
    if delta_ioi < -tau:
        roles.append("name_mover")
    if delta_s_logit > tau:
        roles.append("s_inhibition")
    if attn_s2_s1 > 0.2:
        roles.append("duplicate_token")
    if attn_prev > 0.3:
        roles.append("previous_token")
    if delta_ioi > tau and delta_s_logit < -tau:
        roles.append("negative_name_mover")

    return {
        "delta_ioi": round(float(delta_ioi), 4),
        "delta_s_logit": round(float(delta_s_logit), 4),
        "delta_io_logit": round(float(delta_io_logit), 4),
        "attn_s2_s1": round(attn_s2_s1, 4),
        "attn_prev": round(attn_prev, 4),
        "roles": roles,
    }


def run_experiment_b(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT B: Full Circuit Component Tracking")
    print("=" * 60)

    if "exp_b_components" not in results:
        results["exp_b_components"] = {}

    # Use subset of checkpoints for speed (the interesting ones)
    key_checkpoints = [0, 512, 1000, 2000, 3000, 5000, 10000, 143000]

    for model_name, model_size in MODELS:
        m_key = "pythia_%dm" % model_size
        if m_key not in results["exp_b_components"]:
            results["exp_b_components"][m_key] = {}

        print("\n" + "=" * 40)
        print("  %s" % model_name)
        print("=" * 40)

        for step in key_checkpoints:
            step_key = "step_%d" % step

            # Skip if already done
            if step_key in results["exp_b_components"][m_key]:
                print("  Step %d already done, skipping" % step)
                continue

            print("\n--- Step %d ---" % step)
            try:
                model = load_model(model_name, step)
            except Exception as e:
                print("  FAILED: %s" % str(e))
                continue

            # Generate IOI data with position info
            data = get_ioi_data(model, n_templates=10, ppt=PPT, seed=SEED)

            n_layers = model.cfg.n_layers
            n_heads = model.cfg.n_heads

            # Count components
            counts = {
                "name_mover": 0,
                "s_inhibition": 0,
                "duplicate_token": 0,
                "previous_token": 0,
                "negative_name_mover": 0,
            }
            top_heads = {
                "name_mover": [],
                "s_inhibition": [],
                "duplicate_token": [],
                "previous_token": [],
            }

            for layer in range(n_layers):
                for head in range(n_heads):
                    try:
                        metrics = classify_head_full(
                            model, layer, head,
                            data["tokens_list"], data["io_ids_list"], data["s_ids_list"],
                            data["io_positions"], data["s1_positions"], data["s2_positions"],
                            tau=TAU,
                        )

                        for role in metrics["roles"]:
                            if role in counts:
                                counts[role] += 1

                        head_name = "L%dH%d" % (layer, head)

                        # Track top heads by relevant metric
                        if "name_mover" in metrics["roles"]:
                            top_heads["name_mover"].append(
                                (head_name, metrics["delta_ioi"]))
                        if "s_inhibition" in metrics["roles"]:
                            top_heads["s_inhibition"].append(
                                (head_name, metrics["delta_s_logit"]))
                        if "duplicate_token" in metrics["roles"]:
                            top_heads["duplicate_token"].append(
                                (head_name, metrics["attn_s2_s1"]))
                        if "previous_token" in metrics["roles"]:
                            top_heads["previous_token"].append(
                                (head_name, metrics["attn_prev"]))

                    except Exception as e:
                        pass  # skip problematic heads silently

                # Print progress every 4 layers
                if (layer + 1) % 4 == 0:
                    print("    Layer %d/%d done" % (layer + 1, n_layers))

            # Sort top heads
            for role in top_heads:
                if role in ["name_mover"]:
                    top_heads[role] = sorted(top_heads[role], key=lambda x: x[1])[:5]
                elif role in ["s_inhibition", "duplicate_token", "previous_token"]:
                    top_heads[role] = sorted(top_heads[role], key=lambda x: -x[1])[:5]

            # Convert to serializable format
            top_heads_ser = {}
            for role in top_heads:
                top_heads_ser[role] = [
                    {"head": h[0], "score": round(h[1], 4)} for h in top_heads[role]
                ]

            step_result = {
                "counts": counts,
                "top_heads": top_heads_ser,
            }

            results["exp_b_components"][m_key][step_key] = step_result
            save_results(results)

            print("  NM=%d, S-inhib=%d, DupTok=%d, PrevTok=%d, NegNM=%d" % (
                counts["name_mover"], counts["s_inhibition"],
                counts["duplicate_token"], counts["previous_token"],
                counts["negative_name_mover"]))

            del model
            torch.cuda.empty_cache()

    print("\nEXPERIMENT B COMPLETE")


# ============================================================
# EXPERIMENT C: Ablate Dominant Head on Pile
# Does the synthetic circuit handle natural IOI?
# ============================================================

def run_experiment_c(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT C: Ablate Dominant Head on Pile")
    print("=" * 60)

    results["exp_c_pile_ablation"] = {}

    with open("data/pile_ioi_natural.json") as f:
        pile_data = json.load(f)

    for model_name, model_size in MODELS:
        m_key = "pythia_%dm" % model_size
        print("\n--- %s ---" % model_name)

        try:
            model = load_model(model_name, 143000)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue

        dom_layer, dom_head = DOMINANT_HEADS[model_name]

        # Filter to single-token IO names
        valid = []
        for e in pile_data:
            toks = model.to_tokens(" " + e["io_name"])
            if toks.shape[1] == 2:
                valid.append(e)

        # Also filter single-token S names
        valid2 = []
        for e in valid:
            toks = model.to_tokens(" " + e["s_name"])
            if toks.shape[1] == 2:
                valid2.append(e)
        valid = valid2

        print("  Valid Pile examples: %d" % len(valid))

        # Baseline Pile accuracy
        correct_base = 0
        correct_abl = 0
        ld_base_list = []
        ld_abl_list = []

        def hook_fn(value, hook, h=dom_head):
            value[:, :, h, :] = 0.0
            return value

        hook = ("blocks.%d.attn.hook_z" % dom_layer, hook_fn)

        for e in valid:
            tokens = model.to_tokens(e["prompt"]).cuda()
            io_tok = model.to_tokens(" " + e["io_name"])[0, 1].item()
            s_tok = model.to_tokens(" " + e["s_name"])[0, 1].item()

            # Baseline
            logits = model(tokens)
            io_logit = logits[0, -1, io_tok].item()
            s_logit = logits[0, -1, s_tok].item()
            if io_logit > s_logit:
                correct_base += 1
            ld_base_list.append(io_logit - s_logit)

            # Ablated
            logits_abl = model.run_with_hooks(tokens, fwd_hooks=[hook])
            io_logit_a = logits_abl[0, -1, io_tok].item()
            s_logit_a = logits_abl[0, -1, s_tok].item()
            if io_logit_a > s_logit_a:
                correct_abl += 1
            ld_abl_list.append(io_logit_a - s_logit_a)

        n = len(valid)
        base_acc = correct_base / n
        abl_acc = correct_abl / n
        diff = abl_acc - base_acc

        results["exp_c_pile_ablation"][m_key] = {
            "n_examples": n,
            "dominant_head": "L%dH%d" % (dom_layer, dom_head),
            "pile_baseline_acc": round(base_acc, 4),
            "pile_ablated_acc": round(abl_acc, 4),
            "pile_ablation_diff": round(diff, 4),
            "pile_baseline_mean_ld": round(float(np.mean(ld_base_list)), 4),
            "pile_ablated_mean_ld": round(float(np.mean(ld_abl_list)), 4),
        }

        print("  Pile baseline: %.3f" % base_acc)
        print("  Pile ablate L%dH%d: %.3f (%+.3f)" % (dom_layer, dom_head, abl_acc, diff))

        save_results(results)

        del model
        torch.cuda.empty_cache()

    print("\nEXPERIMENT C COMPLETE")


# ============================================================
# EXPERIMENT D: Logit Difference Distribution at Step 1000
# Is the model systematically picking S?
# ============================================================

def run_experiment_d(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT D: Logit Diff Distribution at Step 1000")
    print("=" * 60)

    results["exp_d_logit_distribution"] = {}

    for model_name, model_size in MODELS:
        m_key = "pythia_%dm" % model_size
        print("\n--- %s ---" % model_name)

        try:
            model = load_model(model_name, 1000)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue

        all_lds = []
        all_io_ranks = []
        all_s_ranks = []

        for tmpl in TEMPLATES:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).cuda()
            io_ids = torch.tensor(ds.io_token_ids, device="cuda")
            s_ids = torch.tensor(ds.s_token_ids, device="cuda")

            logits = model(tokens)
            last = logits[:, -1, :]

            for i in range(len(io_ids)):
                io_logit = last[i, io_ids[i]].item()
                s_logit = last[i, s_ids[i]].item()
                ld = io_logit - s_logit
                all_lds.append(ld)

                # Rank of IO and S tokens
                sorted_idx = last[i].argsort(descending=True)
                io_rank = (sorted_idx == io_ids[i]).nonzero(as_tuple=True)[0].item()
                s_rank = (sorted_idx == s_ids[i]).nonzero(as_tuple=True)[0].item()
                all_io_ranks.append(io_rank)
                all_s_ranks.append(s_rank)

        lds = np.array(all_lds)
        pct_negative = float((lds < 0).mean())
        pct_positive = float((lds > 0).mean())

        results["exp_d_logit_distribution"][m_key] = {
            "n_examples": len(lds),
            "mean_ld": round(float(lds.mean()), 4),
            "std_ld": round(float(lds.std()), 4),
            "pct_negative": round(pct_negative, 4),
            "pct_positive": round(pct_positive, 4),
            "pct_strongly_negative": round(float((lds < -1.0).mean()), 4),
            "median_io_rank": float(np.median(all_io_ranks)),
            "median_s_rank": float(np.median(all_s_ranks)),
            "mean_io_rank": round(float(np.mean(all_io_ranks)), 1),
            "mean_s_rank": round(float(np.mean(all_s_ranks)), 1),
            "ld_percentiles": {
                "p10": round(float(np.percentile(lds, 10)), 4),
                "p25": round(float(np.percentile(lds, 25)), 4),
                "p50": round(float(np.percentile(lds, 50)), 4),
                "p75": round(float(np.percentile(lds, 75)), 4),
                "p90": round(float(np.percentile(lds, 90)), 4),
            }
        }

        print("  Mean LD: %.4f (std=%.4f)" % (lds.mean(), lds.std()))
        print("  Pct picking S (LD<0): %.1f%%" % (pct_negative * 100))
        print("  Pct picking IO (LD>0): %.1f%%" % (pct_positive * 100))
        print("  Median IO rank: %.0f, Median S rank: %.0f" % (
            np.median(all_io_ranks), np.median(all_s_ranks)))

        save_results(results)

        del model
        torch.cuda.empty_cache()

    print("\nEXPERIMENT D COMPLETE")


# ============================================================
# EXPERIMENT E: Attention Patterns of Early NMs
# Where are early NMs attending at step 1000 vs 3000?
# ============================================================

def run_experiment_e(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT E: Early NM Attention Patterns")
    print("=" * 60)

    results["exp_e_attention"] = {}

    model_name = "EleutherAI/pythia-160m-deduped"
    heads_to_check = [(0, 5), (0, 6), (0, 10), (8, 9)]

    for step in [1000, 3000, 143000]:
        print("\n--- Step %d ---" % step)
        step_key = "step_%d" % step

        try:
            model = load_model(model_name, step)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue

        head_attn_results = {}

        for layer, head in heads_to_check:
            if layer >= model.cfg.n_layers:
                continue

            attn_to_io = []
            attn_to_s1 = []
            attn_to_s2 = []
            attn_to_last_pos = []
            attn_to_prev = []

            for tmpl in TEMPLATES[:10]:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).cuda()
                io_ids = torch.tensor(ds.io_token_ids, device="cuda")
                s_ids = torch.tensor(ds.s_token_ids, device="cuda")

                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                attn = cache["blocks.%d.attn.hook_pattern" % layer]
                # attn shape: [batch, n_heads, dest, src]

                final_pos = tokens.shape[1] - 1

                for i in range(tokens.shape[0]):
                    io_tok = io_ids[i].item()
                    s_tok = s_ids[i].item()

                    # Find positions
                    io_pos = -1
                    s1_pos = -1
                    s2_pos = -1
                    s_count = 0

                    for j in range(1, tokens.shape[1]):
                        if tokens[i, j].item() == io_tok and io_pos == -1:
                            io_pos = j
                        if tokens[i, j].item() == s_tok:
                            s_count += 1
                            if s_count == 1:
                                s1_pos = j
                            elif s_count == 2:
                                s2_pos = j

                    # Attention FROM final position TO each key position
                    if io_pos > 0:
                        attn_to_io.append(attn[i, head, final_pos, io_pos].item())
                    if s1_pos > 0:
                        attn_to_s1.append(attn[i, head, final_pos, s1_pos].item())
                    if s2_pos > 0:
                        attn_to_s2.append(attn[i, head, final_pos, s2_pos].item())
                    if final_pos > 0:
                        attn_to_prev.append(attn[i, head, final_pos, final_pos - 1].item())

                del cache
                torch.cuda.empty_cache()

            head_name = "L%dH%d" % (layer, head)
            head_attn_results[head_name] = {
                "attn_to_IO": round(float(np.mean(attn_to_io)), 4) if attn_to_io else 0,
                "attn_to_S1": round(float(np.mean(attn_to_s1)), 4) if attn_to_s1 else 0,
                "attn_to_S2": round(float(np.mean(attn_to_s2)), 4) if attn_to_s2 else 0,
                "attn_to_prev": round(float(np.mean(attn_to_prev)), 4) if attn_to_prev else 0,
                "n_examples": len(attn_to_io),
            }

            r = head_attn_results[head_name]
            print("  %s: to_IO=%.4f, to_S1=%.4f, to_S2=%.4f, to_prev=%.4f" % (
                head_name, r["attn_to_IO"], r["attn_to_S1"],
                r["attn_to_S2"], r["attn_to_prev"]))

        results["exp_e_attention"][step_key] = head_attn_results
        save_results(results)

        del model
        torch.cuda.empty_cache()

    print("\nEXPERIMENT E COMPLETE")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  MEGA EXPERIMENT SCRIPT")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # Load existing results if any
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print("  Loaded existing results")
    else:
        results = {}

    t0 = time.time()

    # Run experiments in order of speed and importance
    try:
        run_experiment_a(results)  # ~20 min
    except Exception as e:
        print("EXPERIMENT A FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_experiment_d(results)  # ~10 min
    except Exception as e:
        print("EXPERIMENT D FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_experiment_e(results)  # ~30 min
    except Exception as e:
        print("EXPERIMENT E FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_experiment_c(results)  # ~20 min
    except Exception as e:
        print("EXPERIMENT C FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_experiment_b(results)  # ~4-6 hours (the big one)
    except Exception as e:
        print("EXPERIMENT B FAILED: %s" % str(e))
        traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("  Total time: %.0f seconds (%.1f hours)" % (elapsed, elapsed / 3600))
    print("=" * 60)


if __name__ == "__main__":
    main()
