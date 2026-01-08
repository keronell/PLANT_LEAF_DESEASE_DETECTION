# Plant Leaf Disease Detection Project

This project implements deep learning models for plant leaf disease classification using EfficientNet-B0 and Vision Transformer (ViT) architectures.

## Project Structure

### 📁 `data_labeling/`
Data preparation and labeling notebooks.
- `1_data_labeling.ipynb`: Dataset labeling and organization

### 📁 `training_base_model/`
Initial model training on the main dataset.
- `2_dataset_and_augmentations.ipynb`: Dataset loading and augmentation strategies
- `3_train_model.ipynb`: Training EfficientNet-B0 and ViT-Base models

### 📁 `experiment_1/`
Domain shift analysis and initial fine-tuning experiments.
- `4_test_common_classes.ipynb`: Testing models on common classes across datasets (revealed domain shift)
- `5_fine_tune_models.ipynb`: Initial fine-tuning attempt

### 📁 `experiment_2/`
Enhanced domain adaptation with "Quick Wins" strategies.
- `5.1_fine_tune_models_quick_wins.ipynb`: Enhanced fine-tuning with domain-balanced sampling
- `5.1_quick_wins_results_summary.txt`: Results summary
- `IMPROVEMENT_STRATEGY.md`: Multi-week improvement strategy

### 📁 `data/`
Dataset directories:
- `Plant_leaf_diseases_dataset_with_augmentation/`: Main training dataset
- `Plant_doc/`: External dataset for domain adaptation
- `FieldPlant_reformatted/`: Field dataset for domain adaptation

### 📁 `models/`
Saved model checkpoints:
- `efficientnet_b0_best.pt`: Baseline EfficientNet-B0
- `vit_base_patch16_224_best.pt`: Baseline ViT
- `efficientnet_b0_fine_tuned.pt`: Fine-tuned EfficientNet (Experiment 1)
- `efficientnet_b0_quick_wins.pt`: Enhanced fine-tuned EfficientNet (Experiment 2)

### 📁 `metadata/`
- `label_mapping.json`: Class label mappings
- `dataset_index.json`: Dataset indexing information

### 📁 `visualizations/`
Generated plots and visualizations for presentations.

### 📁 `info/`
Documentation and model information.

### 📁 `old/`
Archive of original files before reorganization.

## Workflow

1. **Data Labeling**: Prepare and organize datasets
2. **Training Base Model**: Train initial models on main dataset
3. **Experiment 1**: Identify domain shift and test initial fine-tuning
4. **Experiment 2**: Implement enhanced domain adaptation strategies

## Key Results

- **Baseline**: Main dataset F1 ~0.99, but poor performance on external datasets
- **Experiment 1**: Initial fine-tuning showed moderate improvements
- **Experiment 2**: Significant improvements on FieldPlant (F1: 0.03 → 0.62) and Plant_doc (F1: 0.07 → 0.32)

## Next Steps

See `experiment_2/IMPROVEMENT_STRATEGY.md` for future improvement plans.

