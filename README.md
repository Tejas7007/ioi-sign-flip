# Detection Without Suppression: A Sign Flip in IOI Circuit Formation

Code and results for the ICML 2026 Mechanistic Interpretability Workshop submission.

## Structure

```
scripts/       Experiment code (each script is self-contained)
results/       Raw JSON outputs from all experiments
figures/       All figures used in the paper (main + appendix)
```

## Mapping: Paper Section → Script → Results

| Paper Section | Script | Results File |
|---|---|---|
| §3 The Dip Is Robust (Fig 1) | `mega_experiments.py`, `stanford_gpt2_analysis.py`, `polypythias_fix.py` | `pythia_*_component_emergence.json`, `stanford_gpt2_ioi.json`, `polypythias_ioi.json` |
| §4 Duplication Probes (Fig 2, Table 1) | `duplication_probes.py` | `duplication_probes.json` |
| §5 S2 Sign Flip (Fig 3, Table 2) | `causal_intervention.py` | `causal_intervention.json` |
| §5 Controls | `negative_control.py`, `projection_control.py` | `negative_controls.json`, `projection_controls.json` |
| §6 Head Ablation (Table 3) | `cole_experiments_apr30.py` | `cole_experiments_apr30.json` |
| Retrained Pythia-160M | `retrain_pythia_160m.py` | `retrain_ioi_analysis.json` |
| All figures | `generate_all_figures.py` | `figures/` |

## Requirements

```
torch
transformers
transformer-lens
scikit-learn
matplotlib
numpy
```

## Retrained Checkpoints

103 dense checkpoints of Pythia-160M retrained from scratch (seed=42) are available at [link upon acceptance].

## Models Used

- EleutherAI/pythia-160m-deduped (and 410M, 1B variants)
- Stanford GPT-2 Small (alias and battlestar seeds)
- PolyPythias-160M (9 variants: seed1, seed3, seed5, data-seed1/2/3, weight-seed1/2/3)
