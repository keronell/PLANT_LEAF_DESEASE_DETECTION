# Experiment 1: Domain Shift Analysis and Initial Fine-Tuning

This folder contains experiments that identified the domain shift problem and implemented initial fine-tuning strategies.

## Files

- **4_test_common_classes.ipynb**: Notebook for testing models on common classes across all three datasets (main, Plant_doc, FieldPlant). This experiment revealed the severe domain shift problem when models trained on the main dataset were evaluated on external datasets.

- **5_fine_tune_models.ipynb**: Initial fine-tuning experiment that combined datasets, used lower learning rates, fewer epochs, and layer-wise learning rates. This was the first attempt to address domain shift through fine-tuning.

## Key Findings

- Models trained on the main dataset performed poorly on Plant_doc and FieldPlant datasets
- Domain shift was identified as the primary issue
- Initial fine-tuning showed some improvement but needed further optimization

