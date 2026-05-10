"""
Step 1: Build the Data Matrix and Split into Train/Test Sets
=============================================================
ORL (AT&T) Face Database
- 40 subjects, 10 images each  → 400 images total
- Each image: 92×112 grayscale → 10,304 pixels per vector
- Data folder structure expected:
      ./data/s1/1.pgm  ...  ./data/s1/10.pgm
      ./data/s2/1.pgm  ...  ./data/s2/10.pgm
      ...
      ./data/s40/1.pgm ... ./data/s40/10.pgm

Train/Test split:
  - Odd  rows (1,3,5,7,9  → indices 0,2,4,6,8) → training   (5 per subject = 200)
  - Even rows (2,4,6,8,10 → indices 1,3,5,7,9) → testing    (5 per subject = 200)

Output CSV layout (ready for model training):
  - First column  : 'label'   (integer subject id, 1–40)
  - Columns 1…10304: 'p0' … 'p10303'  (pixel values)

  Usage in downstream training code:
      import pandas as pd
      df    = pd.read_csv("output/train.csv")
      X     = df.drop(columns="label").values   # (200, 10304)
      y     = df["label"].values                # (200,)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = "./data"          # root folder containing s1/ … s40/
OUTPUT_DIR = "./output"
IMG_HEIGHT = 112
IMG_WIDTH  = 92
N_SUBJECTS = 40
N_IMGS     = 10                # images per subject
IMG_SIZE   = IMG_HEIGHT * IMG_WIDTH   # 10,304

# Column names for pixel features: p0, p1, …, p10303
PIXEL_COLS = [f"p{i}" for i in range(IMG_SIZE)]


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images, flatten each to a row vector.

    Returns
    -------
    D : ndarray, shape (400, 10304)  – Data matrix
    y : ndarray, shape (400,)        – Label vector (1 … 40)
    """
    rows, labels = [], []

    for subject_id in range(1, N_SUBJECTS + 1):
        subject_folder = os.path.join(data_dir, f"s{subject_id}")
        if not os.path.isdir(subject_folder):
            raise FileNotFoundError(
                f"Subject folder not found: {subject_folder}\n"
                "Make sure the ORL dataset is placed under ./data/s1/ … ./data/s40/"
            )

        for img_idx in range(1, N_IMGS + 1):
            img_path = os.path.join(subject_folder, f"{img_idx}.pgm")
            img = Image.open(img_path).convert("L")   # grayscale

            # Sanity-check dimensions
            if img.size != (IMG_WIDTH, IMG_HEIGHT):
                print(f"  [Warning] Unexpected size {img.size} for {img_path}")

            rows.append(np.array(img, dtype=np.float64).flatten())
            labels.append(subject_id)

    D = np.array(rows)          # (400, 10304)
    y = np.array(labels)        # (400,)
    return D, y


def train_test_split_orl(D: np.ndarray, y: np.ndarray):
    """
    Split by odd/even position within each subject's block of 10.

    Images for subject k are at rows  (k-1)*10  …  k*10-1.
    Within that block:
      positions 0,2,4,6,8  (1st,3rd,5th,7th,9th  image) → train
      positions 1,3,5,7,9  (2nd,4th,6th,8th,10th image) → test
    """
    train_idx, test_idx = [], []

    for subject_id in range(1, N_SUBJECTS + 1):
        base = (subject_id - 1) * N_IMGS
        for i in range(N_IMGS):
            if i % 2 == 0:      # odd image (1-indexed: 1,3,5,7,9)
                train_idx.append(base + i)
            else:               # even image (1-indexed: 2,4,6,8,10)
                test_idx.append(base + i)

    D_train = D[train_idx]    # (200, 10304)
    y_train = y[train_idx]    # (200,)
    D_test  = D[test_idx]     # (200, 10304)
    y_test  = y[test_idx]     # (200,)

    return D_train, y_train, D_test, y_test


def to_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Combine feature matrix X and label vector y into a single DataFrame.
    The 'label' column comes first so it's easy to spot and split.
    """
    df = pd.DataFrame(X, columns=PIXEL_COLS)
    df.insert(0, "label", y.astype(int))
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("Loading dataset …")
    D, y = load_dataset(DATA_DIR)
    print(f"  Data matrix D : {D.shape}   (samples × pixels)")
    print(f"  Label vector y: {y.shape}   labels range [{y.min()}, {y.max()}]")

    # ── 2. Save full matrix ───────────────────────────────────────────────────
    full_path = os.path.join(OUTPUT_DIR, "data_full.csv")
    to_dataframe(D, y).to_csv(full_path, index=False)
    print(f"\n  Saved full dataset → {full_path}")
    print(f"  Columns: label + {len(PIXEL_COLS)} pixel features")

    # ── 3. Train / Test split ────────────────────────────────────────────────
    print("\nSplitting into train / test …")
    D_train, y_train, D_test, y_test = train_test_split_orl(D, y)

    print(f"  D_train : {D_train.shape}   y_train : {y_train.shape}")
    print(f"  D_test  : {D_test.shape}    y_test  : {y_test.shape}")

    # Verify each label has exactly 5 train and 5 test samples
    for sid in range(1, N_SUBJECTS + 1):
        n_tr = np.sum(y_train == sid)
        n_te = np.sum(y_test  == sid)
        assert n_tr == 5 and n_te == 5, \
            f"Subject {sid}: {n_tr} train, {n_te} test — expected 5/5"
    print("  ✓ Each subject has exactly 5 training and 5 test images.")

    # ── 4. Save splits as CSV (label + pixel columns) ────────────────────────
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test.csv")

    to_dataframe(D_train, y_train).to_csv(train_path, index=False)
    to_dataframe(D_test,  y_test ).to_csv(test_path,  index=False)

    print(f"\n  Saved → {train_path}  (shape: {D_train.shape[0]} rows × {1 + D_train.shape[1]} cols)")
    print(f"  Saved → {test_path}   (shape: {D_test.shape[0]} rows × {1 + D_test.shape[1]} cols)")

    print("""
  ── How to load in your model training script ───────────────────────
  import pandas as pd

  df_train = pd.read_csv("output/train.csv")
  X_train  = df_train.drop(columns="label").values   # (200, 10304)
  y_train  = df_train["label"].values                # (200,)

  df_test  = pd.read_csv("output/test.csv")
  X_test   = df_test.drop(columns="label").values    # (200, 10304)
  y_test   = df_test["label"].values                 # (200,)
  ────────────────────────────────────────────────────────────────────""")

    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()