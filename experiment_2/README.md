# Experiment 2: Quick Wins - Enhanced Domain Adaptation

This folder contains the "Week 2 Quick Wins" implementation with enhanced domain adaptation strategies.

## Files

- **5.1_fine_tune_models_quick_wins.ipynb**: Enhanced fine-tuning notebook implementing:
  - Enhanced aggressive field augmentation
  - Domain-balanced sampling (60% PV / 40% Field per batch)
  - Longer fine-tuning (12 epochs)
  - Weighted validation F1 (emphasizes field performance)
  - Layer-wise learning rates
  - Evaluation without TTA (TTA degraded performance)

- **5.1_quick_wins_results_summary.txt**: Summary of experimental results, including training progress and performance metrics.

- **IMPROVEMENT_STRATEGY.md**: Multi-week improvement strategy document outlining the plan for addressing domain shift.

## Key Improvements

- FieldPlant F1 improved from 0.0294 to 0.6187 (training validation)
- Plant_doc F1 improved from 0.0689 to 0.3179 (training validation)
- Weighted validation F1 reached 0.6755
- Main dataset performance maintained at ~0.98

## Output Model

- `efficientnet_b0_quick_wins.pt` (saved in `../models/`)

