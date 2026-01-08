# Strategy to Improve Plant_doc and FieldPlant Performance

## Current Situation Analysis

**Performance Summary:**
- **Main Dataset**: Excellent (99%+ accuracy) ✅
- **Plant_doc**: Catastrophic failure (9.92% accuracy) ❌
- **FieldPlant**: Catastrophic failure (13.79% accuracy) ❌

**Root Cause**: Severe domain shift - models trained on clean, studio-quality images (PlantVillage) fail on real-world field images with different:
- Lighting conditions
- Backgrounds
- Image quality
- Camera angles
- Image preprocessing

---

## Phase 1: Diagnostic Analysis (IMMEDIATE - Do First)

### 1.1 Confusion Matrix Analysis
**Action**: Create detailed confusion matrices for Plant_doc and FieldPlant to understand:
- Which classes are being confused
- If models are collapsing to a single class
- Class-wise performance breakdown

**Code to add to `5_fine_tune_models.ipynb`:**
```python
# After evaluation, add this analysis cell
def analyze_failures(results_dict, dataset_name, id_to_label):
    """Analyze failure patterns."""
    if dataset_name not in results_dict:
        return
    
    results = results_dict[dataset_name]
    cm = results['confusion_matrix']
    targets = results['all_targets']
    preds = results['all_preds']
    
    # Find most confused classes
    print(f"\n{'='*70}")
    print(f"FAILURE ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    for class_id in range(len(cm)):
        if cm[class_id].sum() > 0:
            acc = cm[class_id, class_id] / cm[class_id].sum()
            print(f"  Class {class_id} ({id_to_label[class_id]}): {acc:.3f}")
    
    # Most common misclassifications
    print("\nTop 10 Misclassifications:")
    misclass_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                misclass_pairs.append((i, j, cm[i, j]))
    misclass_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, count in misclass_pairs[:10]:
        print(f"  {id_to_label[i]} → {id_to_label[j]}: {count} times")

# Run analysis
analyze_failures(results_eff, "plant_doc", id_to_label)
analyze_failures(results_eff, "fieldplant", id_to_label)
```

### 1.2 Visual Inspection
**Action**: Sample and visualize misclassified images
- Save 20-30 misclassified examples per dataset
- Compare with correctly classified examples
- Identify visual patterns (lighting, background, image quality)

---

## Phase 2: Data-Level Improvements

### 2.1 Enhanced Data Augmentation for Field Images
**Current Issue**: Field images need more aggressive augmentation to match real-world variability.

**Action**: Create domain-specific augmentation pipeline:

```python
# Add to transforms in fine-tuning notebook
transform_field_aggressive = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),  # Add vertical flip
    T.RandomRotation(degrees=30),  # Increase rotation
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.7, 1.3)),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),  # Add perspective
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Add blur
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.ToTensor(),
    T.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

### 2.2 Domain Mixing Augmentation
**Action**: Mix images from different domains during training

```python
# Add Mixup/CutMix with domain awareness
def domain_mixup(images, labels, domains, alpha=0.4):
    """Mix images from different domains."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    # Only mix if domains are different
    mixed_images = lam * images + (1 - lam) * images[index]
    y_a, y_b = labels, labels[index]
    
    return mixed_images, y_a, y_b, lam
```

### 2.3 Increase Field Data Representation
**Current**: FieldPlant has only 4,640 training samples vs 49,179 main dataset samples.

**Actions**:
1. **Oversample field images**: Use 2-3x more field images per batch
2. **Collect more field data**: If possible, gather more real-world images
3. **Synthetic field data**: Use style transfer to convert PV images to field-like appearance

---

## Phase 3: Training Strategy Improvements

### 3.1 Progressive Domain Adaptation
**Strategy**: Gradually increase field data during training

**Implementation**:
```python
def progressive_domain_training(model, epochs=10):
    """Train with increasing field data proportion."""
    for epoch in range(epochs):
        # Start with 10% field data, increase to 50%
        field_ratio = 0.1 + (epoch / epochs) * 0.4
        
        # Create weighted sampler with higher weight for field images
        field_weight = 5.0  # 5x more likely to sample field images
        # ... implement weighted sampling
        
        # Train one epoch
        train_one_epoch(...)
```

### 3.2 Domain-Balanced Training
**Action**: Ensure each batch contains equal representation from all domains

```python
# Create domain-balanced sampler
class DomainBalancedSampler:
    def __init__(self, entries):
        # Group entries by domain
        self.pv_entries = [e for e in entries if e.get('domain') == 'pv']
        self.field_entries = [e for e in entries if e.get('domain') == 'field']
        
    def __iter__(self):
        # Yield equal number from each domain
        # ... implementation
```

### 3.3 Longer Fine-Tuning with Domain-Specific Validation
**Current**: Only 5 epochs of fine-tuning

**Action**: 
- Increase to 10-15 epochs
- Use **weighted validation F1** that emphasizes field performance:
  ```python
  weighted_val_f1 = (
      0.4 * main_f1 +      # 40% weight on main
      0.3 * plant_doc_f1 +  # 30% weight on plant_doc
      0.3 * fieldplant_f1   # 30% weight on fieldplant
  )
  ```

### 3.4 Learning Rate Schedule for Domain Adaptation
**Action**: Use different learning rates for different domains

```python
# Higher LR for field images to adapt faster
optimizer = AdamW([
    {'params': pv_params, 'lr': 1e-4},
    {'params': field_params, 'lr': 2e-4},  # 2x LR for field
], weight_decay=1e-4)
```

---

## Phase 4: Architecture & Model Improvements

### 4.1 Domain-Specific Batch Normalization
**Action**: Use separate BN statistics for different domains

```python
class DomainAdaptiveBN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_pv = nn.BatchNorm2d(num_features)
        self.bn_field = nn.BatchNorm2d(num_features)
    
    def forward(self, x, domain):
        if domain == 'pv':
            return self.bn_pv(x)
        else:
            return self.bn_field(x)
```

### 4.2 Test-Time Augmentation (TTA)
**Action**: Apply multiple augmentations at test time and average predictions

```python
def predict_with_tta(model, image, num_augments=10):
    """Predict with test-time augmentation."""
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = model(transform_eval(image))
        predictions.append(pred)
    
    # Augmented versions
    for _ in range(num_augments - 1):
        aug_img = transform_field_aggressive(image)
        with torch.no_grad():
            pred = model(aug_img)
            predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

### 4.3 Ensemble of Domain-Specialized Models
**Action**: Train separate models for each domain, then ensemble

```python
# Train 3 models:
# 1. Main dataset model (current best)
# 2. Plant_doc specialized model
# 3. FieldPlant specialized model

# At inference, detect domain and use appropriate model
def predict_with_domain_detection(image):
    domain = detect_domain(image)  # Simple heuristic or small classifier
    if domain == 'pv':
        return model_main(image)
    elif domain == 'plant_doc':
        return model_plant_doc(image)
    else:
        return model_fieldplant(image)
```

---

## Phase 5: Advanced Domain Adaptation Techniques

### 5.1 Domain Adversarial Training
**Action**: Add domain discriminator to force domain-invariant features

```python
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)  # 3 domains: pv, plant_doc, field
        )
    
    def forward(self, features):
        return self.classifier(features)

# Training loop with adversarial loss
def train_with_domain_adversarial(model, domain_disc, ...):
    # Classification loss
    cls_loss = criterion(outputs, targets)
    
    # Domain adversarial loss (gradient reversal)
    domain_pred = domain_disc(features)
    domain_loss = criterion_domain(domain_pred, domain_labels)
    
    # Total loss
    total_loss = cls_loss - 0.1 * domain_loss  # Negative for adversarial
```

### 5.2 Feature Alignment
**Action**: Align feature distributions across domains

```python
def mmd_loss(features_pv, features_field):
    """Maximum Mean Discrepancy loss for domain alignment."""
    # Compute MMD between PV and field features
    # ... implementation
    return mmd_value
```

---

## Phase 6: Data Collection & Augmentation

### 6.1 Style Transfer
**Action**: Convert PV images to field-like style

```python
# Use style transfer (e.g., CycleGAN) to convert PV → Field
# This creates synthetic field data from PV images
```

### 6.2 Pseudo-Labeling
**Action**: Use model predictions on unlabeled field data

```python
# 1. Predict on unlabeled field images
# 2. Use high-confidence predictions as labels
# 3. Add to training set
```

### 6.3 Active Learning
**Action**: Identify most informative field images to label

```python
# Select field images where model is most uncertain
# Prioritize labeling these
```

---

## Recommended Implementation Order

### Week 1: Diagnostics
1. ✅ Add confusion matrix analysis
2. ✅ Visualize misclassifications
3. ✅ Per-class performance breakdown

### Week 2: Quick Wins
1. ✅ Enhanced field augmentation
2. ✅ Domain-balanced sampling
3. ✅ Longer fine-tuning (10 epochs)
4. ✅ Weighted validation F1

### Week 3: Advanced Techniques
1. ✅ Test-Time Augmentation
2. ✅ Domain-specific batch normalization
3. ✅ Progressive domain adaptation

### Week 4: Advanced Domain Adaptation
1. ✅ Domain adversarial training
2. ✅ Feature alignment
3. ✅ Ensemble models

---

## Expected Improvements

| Strategy | Expected Plant_doc F1 | Expected FieldPlant F1 |
|----------|----------------------|----------------------|
| Baseline (Current) | 0.0689 | 0.0294 |
| Enhanced Augmentation | 0.15-0.25 | 0.20-0.35 |
| Domain-Balanced Training | 0.20-0.35 | 0.30-0.45 |
| TTA + Longer Fine-tuning | 0.25-0.40 | 0.35-0.50 |
| Domain Adversarial | 0.35-0.55 | 0.45-0.65 |
| Full Pipeline | **0.50-0.70** | **0.60-0.75** |

---

## Key Metrics to Monitor

1. **Per-domain F1 scores** (not just average)
2. **Confusion matrices** for each domain
3. **Class-wise performance** breakdown
4. **Domain classification accuracy** (if using domain detection)
5. **Feature distribution** visualization (t-SNE/UMAP)

---

## Code Template for Next Experiment

Create a new notebook: `6_domain_adaptation.ipynb` implementing:
1. Enhanced field augmentation
2. Domain-balanced sampling
3. Weighted validation F1
4. Longer fine-tuning (10-15 epochs)
5. TTA at evaluation

This should be your next step before moving to advanced techniques.

