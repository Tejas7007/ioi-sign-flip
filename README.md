# Detection Without Suppression: A Sign Flip in IOI Circuit Formation

Code, results, and figures for the ICML 2026 Mechanistic Interpretability Workshop submission.

## Repository Structure

```
scripts/       Experiment scripts (runnable on a single GPU)
src/           IOI dataset generation code (bundled, no external dependencies)
results/       Raw JSON outputs from all experiments
figures/       All figures used in the paper (main text + appendix)
```

## Requirements

```bash
pip install torch transformers transformer-lens scikit-learn matplotlib numpy
export PYTHONPATH=src:$PYTHONPATH
```

## Script → Results → Paper Mapping

| Paper Section | Script | Results | Key Numbers |
|---|---|---|---|
| §3 Fig 1: Pythia IOI accuracy | `ioi_accuracy_sweep.py` | `pythia_{160m,410m,1b}_ioi_sweep.json` | 160M dips to 35% at step 2000 |
| §3 Fig 1: Stanford GPT-2 | `stanford_gpt2_sweep.py` | `stanford_gpt2_ioi.json` | Dips to 10% at step 1500 |
| §3 App A: PolyPythias | `polypythias_sweep.py` | `polypythias_ioi.json` | All 9 variants dip below 50% |
| §4 Fig 2, Table 1: Probes | `duplication_probes.py` | `duplication_probes.json` | S2 = 99.3% (L5) at step 1000 |
| §5 Fig 3, Table 2: Sign flip | `causal_intervention.py` | `causal_intervention.json` | +0.94 (step 2K), −4.13 (step 143K) |
| §5 Controls | `negative_control.py` | `negative_controls.json` | Wrong pos: 0.000; Random: +0.83 |
| §6 Table 3: Head ablation | `head_ablation.py` | `head_ablation.json` | L8H9 δ = −2.36 at step 5K |
| §6: Loss validation | `head_ablation.py` | `head_ablation.json` | Losses match within 0.05 |
| App D: Projection controls | `projection_control.py` | `projection_controls.json` | Dup direction: −0.11 (wrong sign) |
| Retrained model training | `retrain_pythia_160m.py` | `retrain_ioi_analysis.json` | 103 checkpoints, seed = 42 |
| All figures | `generate_all_figures.py` | `figures/` | Reads results JSONs, outputs PNGs |

## Running Experiments

Each script is self-contained and runnable on a single GPU with 12GB+ VRAM (tested on RTX 3060 and A100).

```bash
# Set up path
export PYTHONPATH=src:$PYTHONPATH

# Run duplication probes (~30 min on RTX 3060)
python scripts/duplication_probes.py

# Run causal intervention (~45 min)
python scripts/causal_intervention.py

# Run head ablation and loss validation (~60 min)
python scripts/head_ablation.py

# Generate all figures from existing results (no GPU needed)
python scripts/generate_all_figures.py
```

## Retrained Checkpoints

103 dense checkpoints of Pythia-160M retrained from scratch (seed = 42), spanning steps 0–10000: every 10 steps for 0–100, every 50 steps for 100–3000, every 200 steps for 3000–10000. Available at [link upon acceptance].

## Models Used

- [EleutherAI/pythia-160m-deduped](https://huggingface.co/EleutherAI/pythia-160m-deduped) (and 410M, 1B)
- Stanford GPT-2 Small via [Mistral](https://crfm.stanford.edu/2021/08/26/mistral.html) (alias and battlestar seeds)
- [PolyPythias-160M](https://huggingface.co/collections/EleutherAI/polypythias-pythia-160m-668a215f47d6e7a0395e8b60) (9 variants)

## Verified Numbers

Every number reported in the paper has been verified against the corresponding JSON results file. The mapping is documented in the table above.
