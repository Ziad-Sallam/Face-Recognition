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