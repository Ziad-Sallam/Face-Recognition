# Face-Recognition

We intend to perform face recognition. Face recognition means that for a given image you can
tell the subject id. Our database of subject is very simple. It has 40 subjects. Below we will
show the needed steps to achieve the goal of the assignment.

---

## Dataset

| Property | Value |
|---|---|
| Source | [AT&T Database of Faces (Kaggle)](https://www.kaggle.com/kasikrit/att-database-of-faces/) |
| Subjects | 40 |
| Images per subject | 10 |
| Total images | 400 |
| Image size | 92 × 112 px (grayscale) |
| Feature vector size | 10,304 pixels per image |

Expected folder structure under `./data/`:

```
data/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ... (10 images)
├── s2/
│   └── ...
└── s40/
    └── ...
```

---

## How to Run

```bash
# Step 1 — build data matrix and split into train/test
python step1_build_dataset.py

# Step 2 — compute PCA, save eigendecomposition, project data
python step2_pca.py
```

All outputs are written to `./output/`.

---

## Output Files

### Raw Data

| File | Shape | Size | Description |
|---|---|---|---|
| `data_full.csv` | 400 × 10305 | ~22 MB | Complete dataset. First column is `label`, then `p0`…`p10303` (pixel values). |
| `train.csv` | 200 × 10305 | ~11 MB | Training split. Odd-indexed images per subject (images 1, 3, 5, 7, 9). |
| `test.csv` | 200 × 10305 | ~11 MB | Test split. Even-indexed images per subject (images 2, 4, 6, 8, 10). |

**Column layout:**

```
label | p0    | p1    | p2    | … | p10303
------+-------+-------+-------+---+--------
1     | 143.0 | 87.0  | 201.0 | … | 55.0
1     | 139.0 | 91.0  | 198.0 | … | 60.0
…
40    | 112.0 | 76.0  | 188.0 | … | 49.0
```

- `label`: integer subject ID (1–40), **5 samples per subject** in each split.
- `p0`…`p10303`: flattened pixel values (row-major order from the 92×112 image).

---

### PCA — Eigendecomposition Cache

Computed once from the training set and saved so `step2_pca.py` skips recomputation on subsequent runs.

| File | Shape | Size | Description |
|---|---|---|---|
| `eigenvalues.csv` | 199 × 1 | ~4 KB | Sorted descending. Single column: `eigenvalue`. |
| `eigenvectors.csv` | 10304 × 199 | ~43 MB | Each column is one eigenvector (eigenface) in pixel space. Columns: `pc1`…`pc199`. |
| `mean_face.csv` | 10304 × 1 | ~74 KB | Mean of all training images. Column: `mean_pixel`. Used to centre new data before projection. |

> **Note:** The dual covariance trick is used — the 200×200 matrix `L = A·Aᵀ / n` is eigen-decomposed instead of the 10304×10304 covariance matrix. The eigenvectors are then mapped back to pixel space. This is mathematically equivalent but far more efficient.

**Loading the cache:**

```python
import pandas as pd
import numpy as np

eigenvalues  = pd.read_csv("output/eigenvalues.csv")["eigenvalue"].values   # (199,)
eigenvectors = pd.read_csv("output/eigenvectors.csv").values                 # (10304, 199)
mean_face    = pd.read_csv("output/mean_face.csv")["mean_pixel"].values      # (10304,)
```

---

### PCA — Projected Data (Model-Ready)

One pair of CSVs per variance threshold α. Each file contains the PCA scores with the label column prepended — load and train directly.

| File | Shape | Size | Components (k) | Variance Retained |
|---|---|---|---|---|
| `Z_train_alpha08.csv` | 200 × 33 | ~118 KB | 32 | 82.10% |
| `Z_test_alpha08.csv` | 200 × 33 | ~118 KB | 32 | 82.10% |
| `Z_train_alpha085.csv` | 200 × 35 | ~126 KB | 34 | 86.78% |
| `Z_test_alpha085.csv` | 200 × 35 | ~126 KB | 34 | 86.78% |
| `Z_train_alpha09.csv` | 200 × 37 | ~133 KB | 36 | 91.41% |
| `Z_test_alpha09.csv` | 200 × 37 | ~133 KB | 36 | 91.41% |
| `Z_train_alpha095.csv` | 200 × 39 | ~141 KB | 38 | 95.96% |
| `Z_test_alpha095.csv` | 200 × 39 | ~141 KB | 38 | 95.96% |

> Shape is `200 × (k + 1)` — the extra column is `label`.

**Column layout (example for α = 0.95, k = 38):**

```
label | pc1      | pc2      | … | pc38
------+----------+----------+---+---------
1     | -312.4   | 88.7     | … | 14.2
1     | -289.1   | 95.3     | … | -8.6
…
40    | 204.7    | -61.2    | … | 22.1
```

- `label`: subject ID (1–40), 5 training / 5 test samples per subject.
- `pc1`…`pck`: principal component scores (coordinates in PCA subspace).

---

## Loading Data for Model Training

```python
import pandas as pd

# ── choose your variance threshold ──────────────────────────────────
ALPHA = "095"   # options: "08", "085", "09", "095"

# ── training data ────────────────────────────────────────────────────
df_train = pd.read_csv(f"output/Z_train_alpha{ALPHA}.csv")
X_train  = df_train.drop(columns="label").values   # (200, k)
y_train  = df_train["label"].values                # (200,)

# ── test data ────────────────────────────────────────────────────────
df_test  = pd.read_csv(f"output/Z_test_alpha{ALPHA}.csv")
X_test   = df_test.drop(columns="label").values    # (200, k)
y_test   = df_test["label"].values                 # (200,)
```

**Example — train a classifier:**

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel="rbf", C=10, gamma="scale")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
```

**Example — project new/unseen images:**

```python
import numpy as np
from PIL import Image

# Load eigendecomposition
eigenvalues  = pd.read_csv("output/eigenvalues.csv")["eigenvalue"].values
eigenvectors = pd.read_csv("output/eigenvectors.csv").values
mean_face    = pd.read_csv("output/mean_face.csv")["mean_pixel"].values

k   = 38   # components for alpha=0.95
img = np.array(Image.open("new_face.pgm").convert("L")).flatten()
z   = (img - mean_face) @ eigenvectors[:, :k]     # (k,) — PCA score vector
```

---

### Visualisations

| File | Description |
|---|---|
| `eigenfaces.png` | Mean face + top 10 eigenfaces |
| `variance_curve.png` | Cumulative explained variance vs number of components, with α thresholds marked |
| `reconstructions.png` | 5 sample training faces reconstructed under each α threshold |

---

## File Tree

```
output/
├── data_full.csv               ← full 400-image dataset (label + pixels)
├── train.csv                   ← 200 training images   (label + pixels)
├── test.csv                    ← 200 test images        (label + pixels)
│
├── eigenvalues.csv             ← 199 eigenvalues (PCA cache)
├── eigenvectors.csv            ← 10304 × 199 eigenvectors (PCA cache)
├── mean_face.csv               ← 10304-pixel mean face (PCA cache)
│
├── Z_train_alpha08.csv         ← train PCA scores, α=0.80, k=32
├── Z_test_alpha08.csv          ← test  PCA scores, α=0.80, k=32
├── Z_train_alpha085.csv        ← train PCA scores, α=0.85, k=34
├── Z_test_alpha085.csv         ← test  PCA scores, α=0.85, k=34
├── Z_train_alpha09.csv         ← train PCA scores, α=0.90, k=36
├── Z_test_alpha09.csv          ← test  PCA scores, α=0.90, k=36
├── Z_train_alpha095.csv        ← train PCA scores, α=0.95, k=38
├── Z_test_alpha095.csv         ← test  PCA scores, α=0.95, k=38
│
├── eigenfaces.png
├── variance_curve.png
└── reconstructions.png
```

## Step 3 — K-Means Clustering (from scratch)
 
K-Means is implemented from scratch: random centroid initialisation from data
points, Euclidean distance assignment, mean update, convergence by centroid
movement threshold. Since K-Means is unsupervised, each cluster is assigned the
majority true label of its members (majority vote mapping) for evaluation.
 
**Grid search:** α ∈ {0.80, 0.85, 0.90, 0.95} × K ∈ {20, 40, 60}
 
### Results
 
| α | K=20 | K=40 | K=60 |
|---|---|---|---|
| 0.80 | 40.00% | 72.00% | **85.00%** |
| 0.85 | 38.00% | 62.50% | 77.00% |
| 0.90 | 37.50% | 55.00% | 69.50% |
| 0.95 | 38.00% | 65.00% | 82.00% |
 
### Best Model
 
| Metric | Value |
|---|---|
| Configuration | α = 0.80, K = 60 |
| Test Accuracy | **85.00%** |
| F1-Score (weighted) | 0.8238 |
 
### Observations
 
**K vs Accuracy:** Accuracy increases consistently with K across all alpha values.
K=20 forces many subjects to share a single cluster, making correct majority-vote
mapping nearly impossible. K=60 provides enough clusters to separate subjects.
 
**α vs Accuracy:** Lower α (fewer PCA components) tends to produce better K-Means
results here. α=0.80 with only 36 components gives the best result (85%).
Higher-dimensional spaces (α=0.95, 115 dims) spread the data in ways that
make centroid-based separation harder with only 200 training points.
 
---
 
## Step 4 — GMM Clustering (from scratch)
 
GMM is implemented from scratch using the Expectation-Maximisation (EM) algorithm
with full covariance matrices. Key implementation details:
 
- **Initialisation:** K random data points as initial means; identity covariance
  scaled by data variance.
- **E-step:** Log-space responsibilities using log-sum-exp for numerical stability.
- **M-step:** Full weighted covariance update with diagonal regularisation
  (`reg_covar = 1e-2`) to prevent singular matrices.
- **Convergence:** Absolute log-likelihood change threshold of 1.0.
- **Cluster labelling:** Same majority-vote mapping as K-Means.
**Grid search:** α ∈ {0.80, 0.85, 0.90, 0.95} × K ∈ {20, 40, 60}
 
### Results
 
| α | K=20 | K=40 | K=60 |
|---|---|---|---|
| 0.80 | 31.50% | 61.00% | 74.50% |
| 0.85 | 35.50% | 52.00% | 65.00% |
| 0.90 | 32.00% | 47.50% | 67.50% |
| 0.95 | 35.00% | 61.00% | **78.50%** |
 
### Best Model
 
| Metric | Value |
|---|---|
| Configuration | α = 0.95, K = 60 |
| Test Accuracy | **78.50%** |
| F1-Score (weighted) | 0.7725 |
 
### Observations
 
**K vs Accuracy:** Same trend as K-Means — accuracy improves as K increases.
Average accuracy across all α values goes from 33.5% (K=20) → 55.4% (K=40)
→ 71.4% (K=60), a very consistent pattern.
 
**α vs Accuracy:** Unlike K-Means, GMM benefits from higher α. At K=60, the
best result is at α=0.95 (78.5%) rather than α=0.80. More PCA components give
GMM richer covariance structure to model the data distribution, and with proper
regularisation GMM can exploit this even at 115 dimensions.
 
**Average accuracy per α (across all K):**
 
| α | Avg Accuracy |
|---|---|
| 0.80 | 55.67% |
| 0.85 | 50.83% |
| 0.90 | 49.00% |
| 0.95 | 58.17% |
 
---
 
## Step 3 vs Step 4 — K-Means vs GMM on PCA
 
| Method | Best Config | Accuracy | F1 |
|---|---|---|---|
| K-Means | α=0.80, K=60 | **85.00%** | 0.8238 |
| GMM | α=0.95, K=60 | 78.50% | 0.7725 |
 
K-Means outperforms GMM by ~6.5 percentage points on this dataset. This is
expected: with only 200 training samples and up to 115 PCA dimensions, GMM's
full covariance matrices have many parameters to estimate (d² per component),
making it less stable than K-Means which only estimates d values per centroid.
K-Means is a special case of GMM with equal, spherical covariances — its
simplicity is an advantage when training data is scarce.
 
---
 
## Bonus — Autoencoder + Clustering
 
### Autoencoder Architecture
 
A fully-connected autoencoder is trained to reconstruct face images from a
compressed 128-dimensional latent representation.
 
```
Input (10304)
    → Dense 1024, ReLU
    → Dense 512,  ReLU
    → Dense 128,  ReLU   ← latent space (bottleneck)
    → Dense 512,  ReLU
    → Dense 1024, ReLU
    → Dense 10304, Sigmoid
```
 
| Hyperparameter | Value |
|---|---|
| Latent dimension | 128 |
| Loss | MSE |
| Optimiser | Adam (lr=0.001) |
| Batch size | 16 |
| Epochs | 50 |
| Total parameters | 22,295,744 |
 
**Training loss (50 epochs):**
 
The model converged smoothly from a train loss of 0.0279 (epoch 1) down to
0.0038 (epoch 50), with validation loss tracking closely at 0.0097, showing no
significant overfitting. Reconstruction quality is visually acceptable — faces
are recognisable with some blurring, which is expected from MSE-trained
autoencoders.
 
The encoder output (`encoded_faces.csv`) contains 400 rows × 128 latent features
plus a `label` column, and is used directly by the bonus clustering step.
 
### Clustering on AE Latent Space
 
The same from-scratch KMeans and GMM implementations from steps 3 and 4 are
reused without modification. The only difference is the input data: instead of
PCA-projected scores, the 128-dimensional AE latent vectors are fed in.
 
**Grid search:** K ∈ {20, 40, 60} (no α dimension — latent dim is fixed at 128)
 
### Results
 
| Method | K=20 | K=40 | K=60 |
|---|---|---|---|
| K-Means (AE) | 40.00% | 59.50% | **68.00%** |
| GMM (AE) | 30.50% | 44.50% | **53.00%** |
 
### Best Models
 
| Method | Best K | Accuracy | F1 |
|---|---|---|---|
| K-Means (AE) | 60 | **68.00%** | 0.6600 |
| GMM (AE) | 60 | 53.00% | 0.4693 |
 
### Observations
 
**K vs Accuracy:** Both methods follow the same pattern seen in PCA steps —
accuracy rises with K. K=60 is best for both.
 
**K-Means vs GMM on AE:** K-Means outperforms GMM by ~15 points in the AE
space, a larger gap than in PCA space (~6.5 points). The AE latent space uses
ReLU activations, producing sparse, non-Gaussian feature distributions. GMM
assumes Gaussian components, so this mismatch hurts its performance more than
in the PCA space where features are more normally distributed.
 
---
 
## Full Comparison — All Methods
 
### Accuracy at each K
 
| Method | K=20 | K=40 | K=60 |
|---|---|---|---|
| K-Means (PCA, best α) | 40.00% | 72.00% | **85.00%** |
| GMM (PCA, best α) | 35.50% | 61.00% | **78.50%** |
| K-Means (AE) | 40.00% | 59.50% | 68.00% |
| GMM (AE) | 30.50% | 44.50% | 53.00% |
 
### Best result per method
 
| Method | Accuracy | F1 | Config |
|---|---|---|---|
| K-Means + PCA | **85.00%** | 0.8238 | α=0.80, K=60 |
| GMM + PCA | 78.50% | 0.7725 | α=0.95, K=60 |
| K-Means + AE | 68.00% | 0.6600 | K=60 |
| GMM + AE | 53.00% | 0.4693 | K=60 |
 
### Why PCA outperforms the Autoencoder here
 
The autoencoder underperforms PCA for this dataset for several reasons:
 
**Dataset size.** With only 200 training images and 22 million parameters, the
autoencoder is heavily over-parameterised. PCA has no free parameters to overfit —
it finds the exact optimal linear projection for the training data.
 
**Linear vs nonlinear.** PCA's linear projection is well-suited to face images,
which lie close to a linear subspace (eigenfaces). The autoencoder's nonlinear
encoding does not necessarily produce a better representation when the underlying
structure is already approximately linear.
 
**Feature distribution.** PCA features are decorrelated and approximately
Gaussian (well-suited for GMM and Euclidean K-Means). AE latent features with
ReLU are sparse and non-Gaussian, which hurts both clustering methods.
 
**Autoencoder advantages on larger datasets.** On larger, more varied datasets
the autoencoder's nonlinear capacity would be an advantage. It would also benefit
from data augmentation and more epochs with a validation-based early stopping
strategy.
 
---
 
## Output Files Summary
 
### PCA Pipeline
 
| File | Description |
|---|---|
| `output/train.csv` | 200 training images (label + pixels) |
| `output/test.csv` | 200 test images (label + pixels) |
| `output/eigenvalues.csv` | 199 PCA eigenvalues |
| `output/eigenvectors.csv` | 10304 × 199 eigenfaces |
| `output/mean_face.csv` | Mean training face |
| `output/Z_train_alpha{tag}.csv` | PCA train scores for α ∈ {08,085,09,095} |
| `output/Z_test_alpha{tag}.csv` | PCA test scores for α ∈ {08,085,09,095} |
| `output/eigenfaces.png` | Mean face + top 10 eigenfaces |
| `output/variance_curve.png` | Cumulative explained variance plot |
| `output/reconstructions.png` | Reconstruction quality at each α |
 
### Clustering — PCA
 
| File | Description |
|---|---|
| `output/kmeans_results.csv` | K-Means grid search results (α × K) |
| `output/kmeans_accuracy_plot.png` | Accuracy vs K grouped by α |
| `output/best_kmeans_confusion_matrix.png` | Confusion matrix for best K-Means |
| `output/gmm_results.csv` | GMM grid search results (α × K) |
| `output/gmm_accuracy_vs_k.png` | GMM accuracy vs K grouped by α |
| `output/gmm_accuracy_vs_alpha.png` | GMM accuracy vs α grouped by K |
| `output/gmm_heatmap.png` | GMM accuracy heat-map (α × K) |
| `output/best_gmm_confusion_matrix.png` | Confusion matrix for best GMM |
| `output/gmm_vs_kmeans.png` | Side-by-side GMM vs K-Means comparison |
 
### Autoencoder Bonus
 
| File | Description |
|---|---|
| `encoded_faces.csv` | 400 × 129 AE latent features + label |
| `face_autoencoder.h5` | Saved full autoencoder model |
| `face_encoder.h5` | Saved encoder-only model |
| `results/reconstructed/` | Original and reconstructed face images |
| `results/graphs/training_loss.png` | AE training/validation loss curve |
| `output/ae_reconstructions.png` | Side-by-side reconstruction samples |
| `output/ae_clustering_results.csv` | K-Means & GMM results on AE space |
| `output/k-means_ae_confusion_matrix.png` | Confusion matrix — K-Means on AE |
| `output/gmm_ae_confusion_matrix.png` | Confusion matrix — GMM on AE |
| `output/ae_vs_pca_comparison.png` | Bar chart: AE vs PCA all methods |
| `output/ae_vs_pca_line.png` | Line plot: accuracy vs K all methods |
 
---
 
## Loading Data for Model Training
 
```python
import pandas as pd
 
# ── PCA data ─────────────────────────────────────────────────────────
ALPHA = "095"   # options: "08", "085", "09", "095"
 
df_train = pd.read_csv(f"output/Z_train_alpha{ALPHA}.csv")
X_train  = df_train.drop(columns="label").values   # (200, k)
y_train  = df_train["label"].values                # (200,)
 
df_test  = pd.read_csv(f"output/Z_test_alpha{ALPHA}.csv")
X_test   = df_test.drop(columns="label").values    # (200, k)
y_test   = df_test["label"].values                 # (200,)
 
# ── AE latent data ────────────────────────────────────────────────────
df_ae    = pd.read_csv("encoded_faces.csv")
X_ae     = df_ae.drop(columns="label").values      # (400, 128)
y_ae     = df_ae["label"].values                   # (400,)
```
