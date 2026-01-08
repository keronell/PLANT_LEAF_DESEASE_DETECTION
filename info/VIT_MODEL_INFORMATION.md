# Vision Transformer (ViT) Model Information
## From 3_train_model.ipynb

---

## 📋 Model Architecture

### Model Name
- **Model Identifier**: `vit_base_patch16_224`
- **Library**: PyTorch Image Models (timm)
- **Architecture Type**: Vision Transformer (ViT) Base with 16x16 patches
- **Input Size**: 224×224 pixels

### Model Configuration
- **Pretrained**: `False` (trained from scratch)
- **Number of Classes**: 39 (plant disease classification)
- **Patch Size**: 16×16 pixels
- **Model Variant**: Base (standard ViT-Base configuration)

---

## 🎯 Training Configuration

### Hyperparameters
- **Learning Rate**: `3e-4` (0.0003)
- **Weight Decay**: `1e-4` (0.0001)
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: CosineAnnealingLR
  - **T_max**: 20 epochs
- **Loss Function**: CrossEntropyLoss
- **Max Epochs**: 20
- **Early Stopping Patience**: 5 epochs
- **Batch Size**: 32

### Training Strategy
- **Device**: CUDA (GPU)
- **Best Model Selection**: Based on validation macro F1 score
- **Model Checkpoint**: Saved to `models/vit_base_patch16_224_best.pt`

---

## 📊 Training Performance

### Training Progress (20 Epochs)

| Epoch | Train Loss | Train Acc | Train F1 | Val Loss | Val Acc | Val F1 | Time (s) |
|-------|------------|-----------|----------|----------|---------|--------|----------|
| 1     | 2.0253     | 0.413     | 0.405    | 1.3397   | 0.581   | 0.559  | 798.8    |
| 2     | 1.3983     | 0.571     | 0.567    | 0.8199   | 0.745   | 0.720  | 563.2    |
| 3     | 1.1323     | 0.646     | 0.644    | 0.6232   | 0.801   | 0.784  | 516.7    |
| 4     | 0.9091     | 0.712     | 0.711    | 0.5333   | 0.828   | 0.817  | 506.3    |
| 5     | 0.8673     | 0.725     | 0.724    | 0.5589   | 0.816   | 0.806  | 501.5    |
| 6     | 0.7157     | 0.770     | 0.769    | 0.4828   | 0.843   | 0.831  | 498.8    |
| 7     | 0.6033     | 0.804     | 0.803    | 0.3974   | 0.873   | 0.865  | 497.4    |
| 8     | 0.5191     | 0.830     | 0.829    | 0.3106   | 0.896   | 0.890  | 495.8    |
| 9     | 0.4351     | 0.857     | 0.857    | 0.2706   | 0.909   | 0.905  | 495.5    |
| 10    | 0.4069     | 0.866     | 0.866    | 0.3142   | 0.892   | 0.885  | 494.3    |
| 11    | 0.3413     | 0.887     | 0.887    | 0.2089   | 0.932   | 0.925  | 494.9    |
| 12    | 0.2790     | 0.906     | 0.906    | 0.1978   | 0.936   | 0.930  | 494.1    |
| 13    | 0.2462     | 0.918     | 0.918    | 0.1681   | 0.941   | 0.938  | 493.8    |
| 14    | 0.1994     | 0.934     | 0.934    | 0.1525   | 0.950   | 0.944  | 493.1    |
| 15    | 0.1748     | 0.943     | 0.943    | 0.1230   | 0.960   | 0.956  | 534.3    |
| 16    | 0.1512     | 0.951     | 0.951    | 0.1041   | 0.967   | 0.963  | 785.8    |
| 17    | 0.1301     | 0.958     | 0.957    | 0.0983   | 0.969   | 0.965  | 690.4    |
| 18    | 0.1152     | 0.962     | 0.962    | 0.0906   | 0.971   | 0.967  | 676.8    |
| 19    | 0.1042     | 0.966     | 0.966    | 0.0863   | 0.972   | 0.968  | 621.6    |
| 20    | 0.1028     | 0.966     | 0.966    | 0.0848   | 0.974   | **0.971** | 611.6    |

### Best Performance Metrics
- **Best Validation F1 Score**: **0.9711** (at epoch 20)
- **Best Validation Accuracy**: 0.974 (97.4%)
- **Best Validation Loss**: 0.0848
- **Final Training Accuracy**: 0.966 (96.6%)
- **Final Training F1**: 0.966

---

## 🧪 Test Set Evaluation

### Test Performance (on internal test split)
- **Test Loss**: 0.0740
- **Test Accuracy**: **0.976** (97.6%)
- **Test Macro F1**: **0.975** (97.5%)

### Test Set Details
- **Total Test Samples**: 6,159 images
- **Number of Classes**: 39
- **Evaluation Metric**: Macro F1 Score (primary), Accuracy

---

## 📈 Training Characteristics

### Training Time
- **Average Time per Epoch**: ~550-800 seconds (9-13 minutes)
- **Total Training Time**: ~3-4 hours for 20 epochs
- **Training Speed**: Slower than EfficientNet-B0 (which took ~300-500s per epoch)

### Convergence Pattern
- **Initial Performance**: Started with F1=0.405 (epoch 1)
- **Steady Improvement**: Consistent improvement throughout training
- **Best Performance**: Achieved at final epoch (20)
- **No Overfitting**: Validation metrics continued improving until end
- **Early Stopping**: Not triggered (patience=5, but model kept improving)

### Training Stability
- **Smooth Convergence**: Gradual decrease in loss
- **Consistent Improvement**: Both train and validation metrics improved steadily
- **No Significant Overfitting**: Gap between train and validation remained reasonable

---

## 🔧 Data Configuration

### Image Preprocessing
- **Input Size**: 224×224 pixels
- **Normalization**: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Data Augmentation (Training)
- **Domain-Specific Augmentation**: Applied based on image domain (PV vs Field)
- **PV Images (Basic)**: Random horizontal flip (p=0.5)
- **PV Images (Field-Style)**: 
  - Random horizontal flip
  - Random rotation (±20°)
  - Color jitter (brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
  - Random affine (translation=0.1, scale=0.8-1.2)
  - Random erasing (p=0.3)
- **Field Images**: 
  - Random horizontal flip
  - Color jitter (brightness=0.3, contrast=0.3)

### Dataset Statistics
- **Training Samples**: 49,179
- **Validation Samples**: 6,148
- **Test Samples**: 6,159
- **Total Samples**: 61,486
- **Number of Classes**: 39
- **Class Balancing**: WeightedRandomSampler used for training

---

## 💾 Model Storage

### Checkpoint Information
- **Checkpoint Path**: `models/vit_base_patch16_224_best.pt`
- **Saved Components**:
  - Model state dictionary
  - Model name
  - Number of classes
  - Best validation F1 score
  - Training history (loss, accuracy, F1 for all epochs)

### Model Loading
```python
checkpoint = torch.load("models/vit_base_patch16_224_best.pt", 
                       map_location=DEVICE, weights_only=False)
model_name = checkpoint["model_name"]  # "vit_base_patch16_224"
num_classes = checkpoint["num_classes"]  # 39
best_val_f1 = checkpoint["best_val_f1"]  # 0.9711
history = checkpoint["history"]  # Training history
```

---

## 📊 Comparison with EfficientNet-B0

| Metric | ViT-Base | EfficientNet-B0 |
|--------|----------|-----------------|
| **Best Val F1** | 0.9711 | 0.9894 |
| **Test Accuracy** | 0.976 | 0.993 |
| **Test F1** | 0.975 | 0.991 |
| **Training Time/Epoch** | ~550-800s | ~300-500s |
| **Model Size** | Larger | Smaller |
| **Convergence** | Slower start | Faster start |

**Note**: EfficientNet-B0 achieved slightly better performance but ViT still achieved excellent results (97.5% F1).

---

## 🎯 Key Features

1. **Transformer Architecture**: Uses self-attention mechanism instead of convolutional layers
2. **Patch-based Processing**: Images divided into 16×16 patches
3. **Position Embeddings**: Learns spatial relationships through positional encoding
4. **End-to-End Training**: Trained from scratch (no pretrained weights)
5. **Robust Performance**: Achieved 97.5% F1 score on test set

---

## 📝 Code Usage

### Model Creation
```python
import timm

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=39
)
```

### Training Call
```python
model_vit, history_vit, best_val_f1_vit = train_model(
    model_name="vit_base_patch16_224",
    num_classes=39,
    train_loader=train_loader,
    val_loader=val_loader,
    max_epochs=20,
    lr=3e-4,
    weight_decay=1e-4,
    device=DEVICE,
    early_stopping_patience=5
)
```

---

## 🔍 Additional Notes

- **Model Library**: Uses `timm` (PyTorch Image Models) for model creation
- **Evaluation Metrics**: Primary metric is macro F1 score for class imbalance handling
- **Device**: Trained on CUDA (GPU) for faster computation
- **Data Loader**: Uses `num_workers=0` for Windows compatibility
- **Memory**: Uses `pin_memory=True` when CUDA is available for faster data transfer

---

## 📅 Training Summary

- **Total Epochs Completed**: 20
- **Best Epoch**: 20 (final epoch)
- **Training Status**: Successfully completed
- **Model Status**: Saved and ready for inference
- **Performance**: Excellent (97.5% F1 on test set)

---

*Generated from analysis of 3_train_model.ipynb*

