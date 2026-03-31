# Architecture Recommendation: Water vs. Land Classification from Full-Waveform Bathymetric LiDAR

---

## 1. Recommended Model Architecture — Layer by Layer

### Minimum Viable Model (MVP): Gradient Boosted Trees on Engineered Features

**Input**: 25–35 scalar features per point (waveform + geometric)
**Output**: Binary label (0=land, 1=water) + probability

This is the right first model because: (1) no GPU required, (2) interpretable feature importance, (3) robust to missing values, (4) works with 234k points.

```
Input: [max_amp, n_gaps, max_gap, time_span, n_peaks, max_peak_spacing,
        total_energy, energy_ratio, reflectance_dB, z, 
        planarity, roughness, height_range_local, n_clusters,
        first_cluster_max_amp, last_cluster_max_amp, ...] → (N, ~30)
       
GradientBoostingClassifier / XGBClassifier
  n_estimators = 300
  max_depth = 6
  learning_rate = 0.05
  subsample = 0.8
  class_weight = 'balanced'  # for class imbalance
  
Output: probability [0, 1]
```

### Ideal Model: Multi-Modal Deep Network (WaveformNet + SpatialNet)

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                           │
│  ─────                                                           │
│  Waveform (dense grid):  (N, 400)   ← 200 SI units × 0.5ns grid │
│  Point features:         (N, 8)     ← [x_rel, y_rel, z, refl,  │
│                                         planarity, roughness,   │
│                                         height_range, density]  │
└──────────────────────┬──────────────────────┬───────────────────┘
                       │                      │
           ┌───────────▼──────────┐  ┌────────▼────────────┐
           │  WAVEFORM BRANCH     │  │  SPATIAL BRANCH      │
           │  (1D CNN)            │  │  (MLP)               │
           │                      │  │                      │
           │  Conv1D(32, k=3)     │  │  Linear(8 → 64)      │
           │  + BatchNorm + ReLU  │  │  + BatchNorm + ReLU  │
           │                      │  │                      │
           │  Conv1D(64, k=5)     │  │  Linear(64 → 64)     │
           │  + BatchNorm + ReLU  │  │  + BatchNorm + ReLU  │
           │                      │  │                      │
           │  Conv1D(64, k=11)    │  │  Linear(64 → 32)     │
           │  + BatchNorm + ReLU  │  │  + ReLU              │
           │                      │  │                      │
           │  MaxPool1D(4)        │  └────────────┬─────────┘
           │                      │               │
           │  Conv1D(128, k=5)    │               │
           │  + BatchNorm + ReLU  │               │
           │                      │               │
           │  AdaptiveAvgPool1D   │               │
           │  → flatten: (N, 128) │               │
           └──────────┬───────────┘               │
                      │                           │
                      └──────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  FUSION HEAD                │
                    │                             │
                    │  Linear(128+32 → 128)        │
                    │  + BatchNorm + ReLU          │
                    │  + Dropout(0.3)              │
                    │                             │
                    │  Linear(128 → 64)            │
                    │  + ReLU                     │
                    │  + Dropout(0.2)              │
                    │                             │
                    │  Linear(64 → 2)              │
                    │  (water_logit, land_logit)   │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  OUTPUT                     │
                    │  Softmax → P(water), P(land) │
                    └─────────────────────────────┘
```

---

## 2. Why This Fits THIS Specific Data

**Non-contiguous time arrays**: The waveform data is stored as non-contiguous time blocks. By projecting onto a **fixed-length dense 1D grid** (0–200 SI units at integer positions), we create a consistent representation. Missing positions are set to 0 (noise floor). The CNN then sees peaks at their absolute temporal positions — which is physically meaningful because peak timing encodes range (distance to the target). A water surface at ~40-50 SI and a bottom return at ~175 SI will always appear at consistent grid positions for a given flying altitude.

**Variable waveform length**: The grid embedding handles this naturally. Waveforms with 19 or 101 samples both map to the same 200-bin grid.

**Multi-modal data**: The green laser's most discriminative information is in the waveform shape (exponential decay, multi-cluster structure, peak count). But the spatial context (z-value, local roughness) provides complementary information, especially for edge cases (shallow water, wet soil, vegetation). The two-branch architecture exploits both.

**Physical distinctiveness**: The large terminal gap (~70-83 SI units) is a systematic feature in the dataset, present in 80.9% of waveforms. This is the dominant signal. Even a 1D CNN with kernel size 3 applied to the dense grid will easily learn to recognize the "long silent period followed by a peak" pattern that characterizes water column traversal.

---

## 3. How to Handle Variable-Length Waveforms — Justified Choice

**Chosen approach: Sparse-to-Dense Grid Projection**

```python
def waveform_to_dense_grid(times, amps, grid_size=200, noise_floor=0):
    """
    Project non-contiguous waveform to fixed-length dense 1D array.
    
    args:
        times: list of SI time values (integers)
        amps: list of ADC amplitude values (integers)
        grid_size: number of bins in output (200 covers SI range ~37 to ~200+)
        noise_floor: fill value for empty bins (0 = below detection threshold)
    
    returns:
        grid: np.array of shape (grid_size,), dtype=float32
    """
    # Find time offset (first sample is not at 0)
    t_min = min(times)
    grid = np.full(grid_size, noise_floor, dtype=np.float32)
    
    for t, a in zip(times, amps):
        idx = t - t_min  # relative time
        if 0 <= idx < grid_size:
            grid[idx] = float(a)
    
    return grid
```

**Why NOT RNN/LSTM**:
- RNN processes sequences step-by-step and implicitly assumes temporal continuity
- With non-contiguous time arrays, an LSTM would process the zero-padding between clusters as if it were real signal at "dt=1" intervals — fundamentally misleading
- Even with masking, the LSTM hidden state during the large gap accumulates nothing useful, then must "remember" the first cluster when the second cluster appears 70 steps later — not ideal

**Why NOT raw sequence with masking (Transformer)**:
- A Transformer with positional encoding based on actual time positions (not sequence index) COULD work — it would correctly attend to t=40 and t=175 regardless of the gap
- However, a 1D CNN on the dense grid achieves the same with less complexity and is more appropriate for a small dataset
- Transformer is recommended as a future improvement after the CNN baseline is established

**Why 1D CNN on dense grid**:
- Naturally handles multi-scale features: small kernels (k=3) detect sharp peaks; large kernels (k=11) detect broad shapes and exponential trends
- Efficient: the 200-bin grid is trivially small, forward pass for 234k points takes seconds
- Interpretable: activation maps show which temporal regions drive the classification

**Size of the grid**: Based on observed data, time values range from SI ~37 to ~188. A grid of 200 bins starting from SI=0 covers all observed values. Using SI=0 as origin aligns all waveforms to an absolute time reference. Alternatively, shift by t_min to be origin-relative (each waveform starts at 0) — the second approach discards absolute depth information but is more robust to altitude variation.

**Recommendation**: Use **origin-relative** representation (each waveform shifted to start at bin 0) for the minimum viable model (removes sensor-height dependency). Use **absolute time** representation for the ideal model (preserves range information, but requires consistent flying altitude).

---

## 4. Spatial Feature Incorporation — Justified Design

The spatial branch processes 8 features per point:

```python
spatial_features = [
    'x_relative',        # x - mean_x: removes large offset, preserves local position
    'y_relative',        # y - mean_y
    'z',                 # absolute elevation (critical: z encodes water depth context)
    '_riegl.reflectance',# reflectance in dB (most important single-point feature)
    'planarity',         # from PCA of k-NN (0=rough/vegetation, 1=flat surface)
    'roughness',         # sqrt(lambda_min) from PCA
    'height_range_local',# max-min z in neighborhood
    'height_std_local',  # std of z in neighborhood
]
```

**Why include z (absolute elevation)**: The Pielach River occupies a specific elevation range within the scene. Points at elevation ~262-264 m (the river surface level) have much higher prior probability of being water than points at 270+ m (dry land/vegetation). The model can learn this spatial prior.

**Why NOT use a graph neural network (GNN) for spatial context**:
- GNNs (PointNet++, DGCNN) are appropriate when the spatial structure IS the primary signal
- Here, the waveform IS the primary signal; spatial features are secondary
- Building the graph for 234k points is computationally expensive
- For the MVP, local geometric features as scalars are sufficient

**Future improvement**: Replace the spatial MLP branch with a PointNet++ mini-module that aggregates information from k-nearest neighbors in waveform-feature space.

---

## 5. Input/Output Tensor Shapes

### Minimum Viable Model (feature-based):
```
Input:  (N, 30)     # N points, 30 scalar features
Output: (N, 2)      # [P(land), P(water)] probabilities
```

### Ideal Deep Model:
```
Waveform input:     (N, 200)    # dense 1D grid per point
Spatial input:      (N, 8)      # scalar geometric features
Output logits:      (N, 2)      # [land_logit, water_logit]
Output probs:       (N, 2)      # after softmax
```

### During training (batched):
```
Waveform batch:     (B, 200)    # B = batch size (recommended: 512-1024)
Spatial batch:      (B, 8)
Label batch:        (B,)        # 0=land, 1=water
Weight batch:       (B,)        # per-sample weight for noisy label handling
```

---

## 6. Loss Function — Justified for Class Imbalance AND Noisy Auto-Labels

**Primary recommendation: Focal Loss with label smoothing**

```python
import torch
import torch.nn.functional as F

class FocalLossWithLabelSmoothing(torch.nn.Module):
    """
    Focal Loss + Label Smoothing for noisy auto-labels and class imbalance.
    
    - Focal loss (Lin et al. 2017): down-weights easy examples, focuses on hard ones
      → directly addresses class imbalance (rare water class gets more attention)
    - Label smoothing: converts hard 0/1 labels to 0.05/0.95
      → reduces overconfidence on noisy auto-labels
    """
    def __init__(self, gamma=2.0, alpha=0.75, smoothing=0.1):
        """
        gamma: focusing parameter (2.0 is standard)
        alpha: weight for positive (water) class (0.75 = upweight rare class)
        smoothing: label smoothing factor (0.1 = labels become 0.1/0.9)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        # Smooth targets
        n_classes = logits.shape[-1]
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Compute focal weight
        probs = F.softmax(logits, dim=-1)
        p_t = probs[range(len(targets)), targets]
        focal_weight = (1 - p_t) ** self.gamma
        
        # Class weight for imbalance
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Cross-entropy with smooth targets
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(smooth_targets * log_probs).sum(dim=-1)
        
        loss = focal_weight * alpha_t * ce
        return loss.mean()
```

**Why focal loss**: The dataset likely has significant class imbalance (water = narrow river channel, land = broad surrounding area). Focal loss naturally handles this by giving more gradient signal to hard/misclassified examples.

**Why label smoothing**: Auto-labels from rules are noisy (estimated 20-30% error rate). Label smoothing prevents the model from becoming overconfident on noisy labels, improving generalization.

**Alternative**: If confident/uncertain label split is used (Section 6.7 of knowledge base):
- Exclude uncertain labels from training (confidence score 0.3–0.7)
- Use binary cross-entropy with class weights on confident labels only
- This is simpler and often more effective when uncertain labels can be identified

**Confident learning (Cleanlab)**:
```python
# Use confident learning to identify and down-weight likely mislabeled samples
from cleanlab.classification import CleanLearning
from sklearn.ensemble import RandomForestClassifier

cl = CleanLearning(RandomForestClassifier(n_estimators=100, class_weight='balanced'))
cl.fit(X_train, y_train_noisy)
# CleanLearning internally identifies likely label errors and handles them
```

---

## 7. Data Augmentation Specific to LiDAR Waveforms

**Waveform augmentations** (apply randomly during training):

```python
def augment_waveform(grid):
    """Apply random augmentations to dense waveform grid."""
    
    # 1. Amplitude scaling (±20%): accounts for reflectance variation
    grid = grid * np.random.uniform(0.8, 1.2)
    
    # 2. Gaussian noise (SNR variation): simulate different turbidity
    grid = grid + np.random.normal(0, 0.02 * grid.max(), size=grid.shape)
    grid = np.clip(grid, 0, None)
    
    # 3. Time jitter (±2 SI units): small GPS/timing errors
    shift = np.random.randint(-2, 3)
    if shift != 0:
        grid = np.roll(grid, shift)
        if shift > 0:
            grid[:shift] = 0
        else:
            grid[shift:] = 0
    
    # 4. Random zero-masking (10% of samples): simulates noise filtering
    mask = np.random.random(grid.shape) < 0.1
    grid[mask] = 0
    
    return grid
```

**Point cloud augmentations**:
```python
def augment_point(features):
    """Apply augmentations to spatial features."""
    
    # 1. Small z jitter (±2 cm): within georeferencing accuracy
    features['z'] += np.random.normal(0, 0.02)
    
    # 2. Reflectance jitter (±1 dB): radiometric uncertainty
    features['_riegl.reflectance'] += np.random.normal(0, 1.0)
    
    # 3. Roughness scaling (±10%): local neighborhood variation
    features['roughness'] *= np.random.uniform(0.9, 1.1)
    
    return features
```

**What NOT to augment**:
- Do not flip the waveform time axis (time ordering is physically meaningful)
- Do not randomly zero-out entire clusters (removes the key multi-cluster signal)
- Do not flip labels (binary labels are ground truth)

---

## 8. Training Plan

### Phase 1: Auto-Labeling
```python
# 1. Run auto_labeler.py to generate pseudo-labels
# 2. Keep only high-confidence samples:
#    water_confidence >= 0.70 → label=1
#    water_confidence <= 0.30 → label=0
#    0.30 < confidence < 0.70 → skip (uncertain)
# Expected: ~60-70% of 234k points get confident labels
```

### Phase 2: Baseline Training (Random Forest)
```python
n_estimators = 300
max_depth = 8
min_samples_leaf = 50  # avoid overfitting to single noisy labels
class_weight = 'balanced'
random_state = 42

# 5-fold spatial cross-validation (split by y-coordinate strips)
# Monitor: F1, precision, recall for both classes
```

### Phase 3: Deep Model Training

```python
# Hyperparameters
batch_size = 512        # fits in memory, stable gradients
learning_rate = 1e-3    # start here with Adam
lr_schedule = 'cosine'  # cosine annealing over training
n_epochs = 100
early_stopping_patience = 15  # stop if validation loss doesn't improve
weight_decay = 1e-4     # L2 regularization

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# LR Schedule: cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

# Training loop
for epoch in range(n_epochs):
    model.train()
    for batch in train_loader:
        wf, spatial, labels = batch
        wf = augment_batch(wf)          # apply waveform augmentation
        logits = model(wf, spatial)
        loss = focal_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validate on held-out spatial strip
    model.eval()
    val_metrics = evaluate(model, val_loader)
    scheduler.step()
    
    # Early stopping check
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

**Monitoring**:
- Loss (focal): training and validation
- F1 score (macro-averaged): primary metric
- Confusion matrix
- Calibration plot (predicted probability vs. empirical frequency)

---

## 9. Ensemble Recommendations

After training individual models, ensemble for improved reliability:

```python
# Ensemble 1: Auto-labeler + RF + Deep model
# Weighted average of probabilities:
p_water_ensemble = (
    0.20 * p_auto_labeler +    # rule-based (physics-grounded but rigid)
    0.30 * p_random_forest +   # feature-based ML (robust baseline)
    0.50 * p_deep_model        # full model (most expressive)
)

# Ensemble 2: Multiple random forest instances (bagging)
from sklearn.ensemble import VotingClassifier
# Different feature subsets or different training splits

# Ensemble 3: Test-time augmentation for deep model
# Average predictions over 5 augmented versions of each waveform
```

**Calibration**: After ensemble, apply Platt scaling or isotonic regression to calibrate probabilities:
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
```

---

## 10. Minimum Viable Model vs. Ideal Model

### Option A: Minimum Viable Model (MVP) — "can be running today"

**Model**: XGBoost on 20 scalar waveform features
**Time to build**: 1–2 days
**Required files**: `auto_labeler.py` + `feature_extractor.py` + `baseline_model.py`

```python
# Core features for MVP (all computable without neural networks):
features = [
    'max_amplitude',           # strong discriminator
    'n_gaps',                  # KEY: 3+ gaps = likely water
    'max_gap_size',            # KEY: >50 SI = water column traversal
    'time_span',               # total waveform duration
    'n_peaks',                 # multi-return count
    'max_peak_spacing',        # peak separation = depth proxy
    'total_energy',            # AUC
    'energy_ratio_second_half',# exponential tail energy
    'reflectance_dB',          # from point cloud
    'z',                       # elevation
    # Simple geometric features:
    'height_range_local',      # local roughness proxy
    'z_relative_to_local_mean',
]

# Training: ~1 minute on 234k points
# Expected accuracy: ~80-85% on confident labels
```

### Option B: Ideal Model — "production-quality"

**Model**: Multi-modal 1D CNN + MLP (as specified in Section 1)
**Time to build**: 1–2 weeks
**Required files**: All 5 implementation files below

```python
# Additional features beyond MVP:
# - Dense waveform grid (200 bins) → CNN branch
# - Full geometric feature set (planarity, roughness, normal vectors) → MLP branch
# - Spatial context (k-NN features) → augmented spatial branch

# Training: ~30 minutes on GPU, ~2 hours on CPU for 234k points
# Expected accuracy: ~88-93% on confident labels (estimated)
# Key advantage: learns features from raw waveform directly
```

**Key difference**: The MVP is faster and more interpretable; the ideal model can learn complex waveform patterns (e.g., the asymmetric exponential decay) directly from data without hand-engineering them.

---

## 11. Iterative Improvement Loop

```
CYCLE 1 (Bootstrap):
  - Run auto_labeler.py → ~60% confident pseudo-labels
  - Train Random Forest (baseline_model.py) → ~80% accuracy
  - Manual inspection: validate ~200 predictions
  - Output: labeled dataset v1

CYCLE 2 (Refinement):
  - Use RF model's confident predictions to expand labeled set
  - Add uncertain samples from cycle 1 + model labels → dataset v2
  - Train deep_model.py → ~85-88% accuracy
  - Output: labeled dataset v2

CYCLE 3 (Fine-tuning):
  - Use deep model predictions to label remaining uncertain points
  - Apply confident learning to identify likely mislabeled samples in v2
  - Retrain with cleaned labels → ~90%+ accuracy
  - Output: final labeled dataset + production model

CYCLE 4 (Active learning):
  - Select most uncertain/valuable unlabeled points for manual annotation
  - Re-train with augmented manual labels
  - Evaluate generalization on a geographically separate river section
```

**Stopping criterion**: When the model's classification agrees with the physics-based rules on >95% of the "high-confidence" auto-labeled samples, the model can be considered converged.

---

## 12. Implementation Plan

### Step 1: `auto_labeler.py` — Rule-Based Labeling
```python
# Input: point_cloud_df.txt + waveform_df.txt
# Output: pseudo_labels.csv with columns [point_id, label, confidence, rule_triggered]
# Key functions:
#   parse_waveform(row) → (times, amps)
#   compute_gap_features(times, amps) → dict
#   compute_water_confidence(point_features, wf_features) → float
#   assign_label(confidence, threshold_water=0.70, threshold_land=0.30) → str
```

### Step 2: `feature_extractor.py` — Feature Engineering
```python
# Input: point_cloud_df.txt + waveform_df.txt
# Output: features.csv with all scalar features + waveform grid
# Key functions:
#   extract_waveform_features(times, amps) → scalar_dict
#   waveform_to_dense_grid(times, amps, grid_size=200) → np.array(200)
#   extract_geometric_features(pc_df, k=20) → feature_df
#   normalize_features(feature_df) → normalized_df
```

### Step 3: `baseline_model.py` — Random Forest/XGBoost Baseline
```python
# Input: features.csv + pseudo_labels.csv
# Output: baseline_predictions.csv + feature_importance.png
# Key functions:
#   load_confident_labels(labels_df, min_confidence=0.70) → filtered_df
#   train_random_forest(X_train, y_train) → model
#   spatial_cross_validate(X, y, coords, n_folds=5) → metrics
#   plot_feature_importance(model, feature_names) → figure
```

### Step 4: `deep_model.py` — Multi-Modal Neural Network
```python
# Input: dense waveform grids + scalar features + pseudo_labels
# Output: trained PyTorch model checkpoint + predictions
# Key classes:
#   WaveformCNN(nn.Module) → waveform branch
#   SpatialMLP(nn.Module) → spatial features branch
#   WaterLandClassifier(nn.Module) → full model with fusion head
#   FocalLossWithSmoothing → loss function
#   WaveformDataset(Dataset) → PyTorch dataset wrapper
```

### Step 5: `evaluate.py` + `train.py` — Training & Evaluation
```python
# train.py:
#   - Load data, split spatially
#   - Train with early stopping
#   - Log metrics to TensorBoard or W&B
#   - Save best checkpoint
#
# evaluate.py:
#   - Load model, run on full dataset
#   - Generate spatial maps of predictions
#   - Compare with auto-labeler confidence scores
#   - Produce confusion matrix, precision-recall curves
#   - Visualize waveform examples for true/false positives/negatives
```

---

## Summary Decision Matrix

| Consideration | MVP (XGBoost) | Ideal (CNN+MLP) |
|---------------|---------------|-----------------|
| Build time | 1–2 days | 1–2 weeks |
| Accuracy (est.) | ~80–85% | ~88–93% |
| GPU required | No | Optional (CPU fine) |
| Interpretability | High (feature importance) | Medium (activation maps) |
| Handles raw waveform | No (features only) | Yes (direct 1D grid) |
| Handles noisy labels | Moderate | Good (focal loss) |
| Class imbalance | class_weight='balanced' | Focal loss (alpha) |
| Variable waveform length | N/A (scalar features) | Dense grid embedding |
| Key differentiator | Fast iteration | End-to-end learning |

**Recommendation**: Start with MVP to validate the labeling pipeline and understand feature importance. Transition to the ideal model once the pipeline is verified and labels have been manually spot-checked on ~200 samples.
