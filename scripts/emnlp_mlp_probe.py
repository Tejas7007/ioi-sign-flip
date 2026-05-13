"""
EMNLP Experiment 1: MLP Probe vs Logistic Regression
Compare 2-layer MLP probe with logistic regression on S2 activations.
Tests whether duplication is nonlinearly encoded at step 0.
"""
import os, sys, json, random, torch, numpy as np
os.environ["HF_TOKEN"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES

DEVICE = "cuda"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
SEED = 42
STEPS = [0, 1000, 2000, 3000, 5000, 10000, 143000]
N_TEMPLATES = 10
PPT = 30
RESULTS_FILE = "results/emnlp_mlp_probe.json"

def get_s2_positions(tokens, s_ids):
    s2_pos = []
    for i in range(tokens.shape[0]):
        s_tok = s_ids[i].item(); cnt = 0; pos = -1
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == s_tok:
                cnt += 1
                if cnt == 2: pos = j; break
        s2_pos.append(pos)
    return s2_pos

def load_model(step):
    rev = "step%d" % step
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=rev)
    model = HookedTransformer.from_pretrained(MODEL_NAME, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf
    return model

def collect_activations(model, n_templates=N_TEMPLATES, ppt=PPT):
    tokenizer = model.tokenizer
    single_names = [tokenizer.encode(" " + n, add_special_tokens=False)[0]
                    for n in CANDIDATE_NAMES if len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1]
    random.seed(SEED)
    acts = {layer: [] for layer in range(12)}
    labels = []
    for tmpl in ALL_TEMPLATES[:n_templates]:
        ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=SEED)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
        s2_pos = get_s2_positions(tokens, s_ids)
        n = tokens.shape[0]
        ctrl = tokens.clone()
        for i in range(n):
            if s2_pos[i] > 0:
                cands = [t for t in single_names if t != s_ids[i].item() and t != io_ids[i].item()]
                if cands: ctrl[i, s2_pos[i]] = random.choice(cands)
        with torch.no_grad():
            _, cache_ioi = model.run_with_cache(tokens, remove_batch_dim=False)
            _, cache_ctrl = model.run_with_cache(ctrl, remove_batch_dim=False)
        for layer in range(12):
            hook = "blocks.%d.hook_resid_post" % layer
            for i in range(n):
                if s2_pos[i] > 0:
                    acts[layer].append(cache_ioi[hook][i, s2_pos[i], :].detach().cpu().float().numpy())
                    acts[layer].append(cache_ctrl[hook][i, s2_pos[i], :].detach().cpu().float().numpy())
        for i in range(n):
            if s2_pos[i] > 0:
                labels.extend([1, 0])
        del cache_ioi, cache_ctrl
    torch.cuda.empty_cache()
    return acts, np.array(labels)

def run_probes(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr_accs = []
    mlp_accs = []
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(max_iter=2000, random_state=SEED, C=1.0)
        lr.fit(X[tr], y[tr])
        lr_accs.append(accuracy_score(y[te], lr.predict(X[te])))
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            max_iter=2000,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=1e-3
        )
        mlp.fit(X[tr], y[tr])
        mlp_accs.append(accuracy_score(y[te], mlp.predict(X[te])))
    return {
        "lr_mean": float(np.mean(lr_accs)),
        "lr_std": float(np.std(lr_accs)),
        "mlp_mean": float(np.mean(mlp_accs)),
        "mlp_std": float(np.std(mlp_accs)),
        "lr_folds": [float(a) for a in lr_accs],
        "mlp_folds": [float(a) for a in mlp_accs],
    }

def main():
    print("=" * 60)
    print("  MLP PROBE vs LOGISTIC REGRESSION")
    print("  Testing nonlinear encoding of duplication at S2")
    print("=" * 60)
    results = {}
    for step in STEPS:
        print("\n=== Step %d ===" % step)
        model = load_model(step)
        ioi_correct = 0; ioi_total = 0
        for tmpl in ALL_TEMPLATES[:N_TEMPLATES]:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl], symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).to(DEVICE)
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
            with torch.no_grad():
                logits = model(tokens)[:, -1, :]
            for i in range(tokens.shape[0]):
                if logits[i, io_ids[i]] > logits[i, s_ids[i]]:
                    ioi_correct += 1
                ioi_total += 1
        ioi_acc = ioi_correct / ioi_total
        print("  IOI accuracy: %.1f%%" % (ioi_acc * 100))
        acts, labels = collect_activations(model)
        print("  Collected %d examples" % len(labels))
        step_results = {"ioi_acc": round(ioi_acc, 4), "n_examples": len(labels), "layers": {}}
        for layer in range(12):
            X = np.array(acts[layer])
            probe_result = run_probes(X, labels)
            step_results["layers"]["layer_%d" % layer] = probe_result
            lr_acc = probe_result["lr_mean"] * 100
            mlp_acc = probe_result["mlp_mean"] * 100
            gap = mlp_acc - lr_acc
            marker = " ***" if abs(gap) > 2.0 else ""
            print("  Layer %2d: LR=%.1f%% MLP=%.1f%% (gap=%+.1f%%)%s" % (layer, lr_acc, mlp_acc, gap, marker))
        results["step_%d" % step] = step_results
        os.makedirs("results", exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("  SUMMARY")
    for step in STEPS:
        r = results["step_%d" % step]
        lr_best_layer = max(r["layers"].keys(), key=lambda l: r["layers"][l]["lr_mean"])
        mlp_best_layer = max(r["layers"].keys(), key=lambda l: r["layers"][l]["mlp_mean"])
        lr_best = r["layers"][lr_best_layer]["lr_mean"] * 100
        mlp_best = r["layers"][mlp_best_layer]["mlp_mean"] * 100
        print("  Step %6d: IOI=%.0f%% LR=%.1f%%(%s) MLP=%.1f%%(%s) gap=%+.1f%%" % (
            step, r["ioi_acc"]*100, lr_best, lr_best_layer, mlp_best, mlp_best_layer, mlp_best-lr_best))
    print("\nSaved: %s" % RESULTS_FILE)

if __name__ == "__main__":
    main()
