import os
os.environ["HF_TOKEN"] = ""
import torch, json, time, numpy as np, sys, random
sys.path.insert(0, '/workspace/MLP-Paper-Cole/src')
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES

DEVICE = "cuda"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
TEMPLATES = ALL_TEMPLATES[:15]
PPT = 30
SEED = 42
RESULTS_FILE = "results/causal_intervention.json"

def load_model(step):
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step%d" % step)
    model = HookedTransformer.from_pretrained(MODEL_NAME, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf; return model

def get_s2_positions(tokens, s_ids):
    s2_positions = []
    for i in range(tokens.shape[0]):
        s_tok = s_ids[i].item(); s_count = 0; s2_pos = -1
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == s_tok:
                s_count += 1
                if s_count == 2: s2_pos = j; break
        s2_positions.append(s2_pos)
    return s2_positions

def get_single_token_names(tokenizer):
    names = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1: names.append(toks[0])
    return names

def run_experiment():
    print("=" * 60)
    print("  CAUSAL INTERVENTION EXPERIMENT")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    results = {}
    steps_to_test = [1000, 2000, 3000, 5000, 143000]

    for step in steps_to_test:
        print("\n=== Step %d ===" % step)
        model = load_model(step)
        single_names = get_single_token_names(model.tokenizer)
        random.seed(SEED)

        layer_ranges = [(0, 3), (3, 6), (4, 7), (0, 6), (0, 12)]
        # Accumulators
        all_baseline_ld = []
        all_ctrl_ld = []
        all_remove = {("layers_%d_%d" % (s, e-1)): [] for s, e in layer_ranges}
        all_add = {("layers_%d_%d" % (s, e-1)): [] for s, e in layer_ranges}

        for tmpl in TEMPLATES[:10]:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl], symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
                s2_pos = get_s2_positions(tokens, s_ids)
                n = tokens.shape[0]

                # Make control: replace S2 with third name
                ctrl = tokens.clone()
                for i in range(n):
                    if s2_pos[i] > 0:
                        cands = [t for t in single_names if t != s_ids[i].item() and t != io_ids[i].item()]
                        if cands: ctrl[i, s2_pos[i]] = random.choice(cands)

                # Baseline logit diffs
                with torch.no_grad():
                    logits_ioi = model(tokens)[:, -1, :]
                    logits_ctrl = model(ctrl)[:, -1, :]
                for i in range(n):
                    all_baseline_ld.append(logits_ioi[i, io_ids[i]].item() - logits_ioi[i, s_ids[i]].item())
                    all_ctrl_ld.append(logits_ctrl[i, io_ids[i]].item() - logits_ctrl[i, s_ids[i]].item())

                # Cache both
                _, cache_ioi = model.run_with_cache(tokens, remove_batch_dim=False)
                _, cache_ctrl = model.run_with_cache(ctrl, remove_batch_dim=False)

                for l_start, l_end in layer_ranges:
                    rname = "layers_%d_%d" % (l_start, l_end - 1)

                    # REMOVE duplication: replace IOI's S2 with control's S2
                    def make_rm_hook(li, ci=cache_ctrl, sp=s2_pos, nn=n):
                        def hook_fn(value, hook):
                            for i in range(nn):
                                if sp[i] > 0:
                                    value[i, sp[i], :] = ci["blocks.%d.hook_resid_post" % li][i, sp[i], :]
                            return value
                        return hook_fn

                    rm_hooks = [("blocks.%d.hook_resid_post" % l, make_rm_hook(l)) for l in range(l_start, l_end)]
                    with torch.no_grad():
                        logits_rm = model.run_with_hooks(tokens, fwd_hooks=rm_hooks)[:, -1, :]
                    for i in range(n):
                        all_remove[rname].append(logits_rm[i, io_ids[i]].item() - logits_rm[i, s_ids[i]].item())

                    # ADD duplication: replace control's S2 with IOI's S2
                    def make_add_hook(li, ci=cache_ioi, sp=s2_pos, nn=n):
                        def hook_fn(value, hook):
                            for i in range(nn):
                                if sp[i] > 0:
                                    value[i, sp[i], :] = ci["blocks.%d.hook_resid_post" % li][i, sp[i], :]
                            return value
                        return hook_fn

                    add_hooks = [("blocks.%d.hook_resid_post" % l, make_add_hook(l)) for l in range(l_start, l_end)]
                    with torch.no_grad():
                        logits_add = model.run_with_hooks(ctrl, fwd_hooks=add_hooks)[:, -1, :]
                    for i in range(n):
                        all_add[rname].append(logits_add[i, io_ids[i]].item() - logits_add[i, s_ids[i]].item())

                del cache_ioi, cache_ctrl
                torch.cuda.empty_cache()
            except Exception as e:
                print("  Template failed: %s" % str(e)[:60])
                continue

        # Compute summary
        bl_mean = float(np.mean(all_baseline_ld))
        bl_acc = float((np.array(all_baseline_ld) > 0).mean())
        ct_mean = float(np.mean(all_ctrl_ld))
        ct_acc = float((np.array(all_ctrl_ld) > 0).mean())
        print("  IOI baseline: LD=%.4f, acc=%.1f%% (n=%d)" % (bl_mean, bl_acc*100, len(all_baseline_ld)))
        print("  Ctrl baseline: LD=%.4f, acc=%.1f%%" % (ct_mean, ct_acc*100))

        rm_res = {}
        add_res = {}
        for rname in all_remove:
            if all_remove[rname]:
                rm_m = float(np.mean(all_remove[rname]))
                rm_a = float((np.array(all_remove[rname]) > 0).mean())
                rm_ch = rm_m - bl_mean
                rm_res[rname] = {"ld": round(rm_m, 4), "acc": round(rm_a, 4), "change": round(rm_ch, 4)}
                print("  REMOVE dup @ %s: LD=%.4f (%+.4f), acc=%.1f%%" % (rname, rm_m, rm_ch, rm_a*100))
            if all_add[rname]:
                ad_m = float(np.mean(all_add[rname]))
                ad_a = float((np.array(all_add[rname]) > 0).mean())
                ad_ch = ad_m - ct_mean
                add_res[rname] = {"ld": round(ad_m, 4), "acc": round(ad_a, 4), "change": round(ad_ch, 4)}
                print("  ADD dup @ %s: LD=%.4f (%+.4f), acc=%.1f%%" % (rname, ad_m, ad_ch, ad_a*100))

        results["step_%d" % step] = {
            "n": len(all_baseline_ld),
            "ioi_ld": round(bl_mean, 4), "ioi_acc": round(bl_acc, 4),
            "ctrl_ld": round(ct_mean, 4), "ctrl_acc": round(ct_acc, 4),
            "remove_dup": rm_res, "add_dup": add_res}

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("  If REMOVE change > 0: removing duplication reduces S-bias (SUPPORTS claim)")
    print("  If ADD change < 0: adding duplication creates S-bias (SUPPORTS claim)")
    print("=" * 60)
    for sk in sorted(results, key=lambda x: int(x.split('_')[1])):
        r = results[sk]
        rm35 = r["remove_dup"].get("layers_3_5", {})
        ad35 = r["add_dup"].get("layers_3_5", {})
        print("  %s: IOI_LD=%.4f | Remove L3-5: %+.4f | Add L3-5: %+.4f" % (
            sk, r["ioi_ld"],
            rm35.get("change", 0),
            ad35.get("change", 0)))
    print("\nResults: %s" % RESULTS_FILE)

if __name__ == "__main__":
    run_experiment()
