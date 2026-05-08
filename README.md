# Detection Without Suppression: A Sign Flip in IOI Circuit Formation

Code, results, and figures for the ICML 2026 Mechanistic Interpretability Workshop submission.

## Repository Structure

```
scripts/       Experiment scripts (each self-contained, runnable on a single GPU)
results/       Raw JSON outputs from all experiments
figures/       All figures used in the paper (main text + appendix)
```

## Reproducing Paper Results

### Requirements
```
pip install torch transformers transformer-lens scikit-learn matplotlib numpy
```

### Script → Results → Paper Mapping

| Paper Evidence | Script | Output | Key Numbers |
|---|---|---|---|
| **§3 Fig 1: IOI dip across models** | `mega_experiments.py` | `pythia_{160m,410m,1b}_component_emergence.json` | Pythia-160M dips to 35% at step 2000 |
| **§3 Fig 1: Stanford GPT-2** | `stanford_gpt2_analysis.py` | `stanford_gpt2_ioi.json` | Stanford dips to 10% at step 1500 |
| **§3 Fig 1: PolyPythias** | `polypythias_fix.py` | `polypythias_ioi.json` | All 9 variants dip below 50% |
| **§4 Fig 2, Table 1: Duplication probes** | `duplication_probes.py` | `duplication_probes.json` | S2=99.3% (L5) at step 1000; 87.8% at step 0 |
| **§5 Fig 3, Table 2: S2 sign flip** | `causal_intervention.py` | `causal_intervention.json` | Remove: +0.94 (step 2K), -4.13 (step 143K) |
| **§5 Controls: Wrong position** | `negative_control.py` | `negative_controls.json` | ΔLD=0.000 at non-S2 position |
| **§5 Controls: Random perturbation** | `negative_control.py` | `negative_controls.json` | ΔLD=+0.83 (88% of real effect) |
| **§6 Table 3: Head ablation** | `cole_experiments_apr30.py` | `cole_experiments_apr30.json` | L0H0 δ=0.06 (step 1K), L8H9 δ=-2.36 (step 5K) |
| **§6: Loss validation** | `cole_experiments_apr30.py` | `cole_experiments_apr30.json` | Losses match within 0.05 from step 2000 |
| **App D: Projection controls** | `projection_control.py` | `projection_controls.json` | Dup direction: -0.11 (wrong sign); controls: ~0 |
| **Retrained model** | `retrain_pythia_160m.py` | `retrain_ioi_analysis.json` | 103 dense checkpoints, seed=42 |
| **All figures** | `generate_all_figures.py` | `figures/` | Reads results JSONs, outputs PNGs |

### Running an Experiment

Each script is self-contained. Example:

```bash
# Run duplication probes (requires GPU, ~30 min on RTX 3060)
python scripts/duplication_probes.py

# Run causal intervention (requires GPU, ~45 min on RTX 3060)
python scripts/causal_intervention.py

# Generate all figures from existing results (no GPU needed)
python scripts/generate_all_figures.py
```

Most scripts require the `circuitscaling` package from the [MLP-Paper-Cole](https://github.com/Tejas7007/MLP-Paper-Cole) repository for IOI dataset generation:
```bash
git clone https://github.com/Tejas7007/MLP-Paper-Cole.git
export PYTHONPATH=/path/to/MLP-Paper-Cole/src:$PYTHONPATH
```

### GPU Requirements

All experiments run on a single GPU with 12GB+ VRAM. Tested on RTX 3060 and A100. Pythia-160M fits comfortably; larger Pythia scales (410M, 1B) require 16GB+.

## Retrained Checkpoints

103 dense checkpoints of Pythia-160M retrained from scratch (seed=42) with coverage every 50 steps in the 0–3000 range. Available at [link upon acceptance].

## Models

- [EleutherAI/pythia-160m-deduped](https://huggingface.co/EleutherAI/pythia-160m-deduped) (and 410M, 1B)
- Stanford GPT-2 Small via [Mistral framework](https://crfm.stanford.edu/2021/08/26/mistral.html) (alias and battlestar seeds)
- [PolyPythias-160M](https://huggingface.co/collections/EleutherAI/polypythias-pythia-160m-668a215f47d6e7a0395e8b60) (9 variants)
