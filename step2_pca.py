"""
Step 2: PCA on the ORL Face Dataset (from-scratch implementation)
=================================================================
- Loads train.csv / test.csv produced by step1_build_dataset.py
- Computes PCA from scratch (mean-centering → covariance → eigen-decomposition)
- Caches eigenvalues & eigenvectors as CSV for reuse / inspection
- Retains components for α ∈ {0.80, 0.85, 0.90, 0.95} variance thresholds
- Projects train & test data and saves model-ready CSVs (label + PC columns)
- Visualises mean face, top eigenfaces, and reconstructed faces

Output CSV layout for projected data (ready for model training):
  - First column     : 'label'  (integer subject id, 1–40)
  - Columns 1 … k   : 'pc1' … 'pck'  (principal component scores)

  Usage in downstream training code:
      import pandas as pd
      df    = pd.read_csv("output/Z_train_alpha095.csv")
      X     = df.drop(columns="label").values   # (200, k)
      y     = df["label"].values                # (200,)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./output"
IMG_HEIGHT = 112
IMG_WIDTH  = 92
ALPHAS     = [0.80, 0.85, 0.90, 0.95]   # variance-retention thresholds


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  PCA helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pca(D_train: np.ndarray):
    """
    Compute PCA from scratch.

    Steps
    -----
    1. Mean-centre the training data.
    2. Build the covariance matrix using the *dual* trick when n < d
       (avoids forming a 10304×10304 matrix).
    3. Eigen-decompose, sort by descending eigenvalue.
    4. Normalise eigenvectors.

    Returns
    -------
    mean_face   : (d,)        mean image vector
    eigenvalues : (n-1,)      sorted descending
    eigenvectors: (d, n-1)    corresponding eigenvectors (columns)
    """
    n, d = D_train.shape          # n=200, d=10304

    # 1. Mean face
    mean_face = D_train.mean(axis=0)           # (d,)

    # 2. Mean-centred matrix  A  of shape (n, d)
    A = D_train - mean_face                    # (200, 10304)

    # 3. Dual covariance trick: L = A @ A^T  has shape (n, n)
    #    If v is an eigenvector of L, then A^T v is an eigenvector of S = A^T A / n
    L = (A @ A.T) / n                          # (200, 200)

    eigenvalues_dual, eigenvectors_dual = np.linalg.eigh(L)

    # Sort descending
    idx              = np.argsort(eigenvalues_dual)[::-1]
    eigenvalues_dual = eigenvalues_dual[idx]
    eigenvectors_dual = eigenvectors_dual[:, idx]

    # Keep only positive eigenvalues (drop near-zero numerical noise)
    pos_mask         = eigenvalues_dual > 1e-8
    eigenvalues_dual = eigenvalues_dual[pos_mask]
    eigenvectors_dual = eigenvectors_dual[:, pos_mask]

    # 4. Map back to d-dimensional space: u_i = A^T v_i / ||A^T v_i||
    eigenvectors = A.T @ eigenvectors_dual          # (d, k)
    norms        = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    eigenvectors = eigenvectors / norms             # normalise columns

    return mean_face, eigenvalues_dual, eigenvectors


def n_components_for_variance(eigenvalues: np.ndarray, alpha: float) -> int:
    """Return the minimum number of PCs to retain ≥ alpha of total variance."""
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k      = int(np.searchsorted(cumvar, alpha)) + 1
    return k


def project(X: np.ndarray, mean_face: np.ndarray,
            eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """Project mean-centred X onto the top-k eigenvectors."""
    return (X - mean_face) @ eigenvectors[:, :k]   # (n_samples, k)


def reconstruct(Z: np.ndarray, mean_face: np.ndarray,
                eigenvectors: np.ndarray) -> np.ndarray:
    """Reconstruct images from PCA coefficients Z."""
    return Z @ eigenvectors[:, :Z.shape[1]].T + mean_face


# ── CSV helpers ───────────────────────────────────────────────────────────────

def save_projected_csv(Z: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save PCA-projected data with label column as a model-ready CSV."""
    k       = Z.shape[1]
    pc_cols = [f"pc{i+1}" for i in range(k)]
    df      = pd.DataFrame(Z, columns=pc_cols)
    df.insert(0, "label", y.astype(int))
    df.to_csv(path, index=False)


def save_eigen_csv(eigenvalues: np.ndarray,
                   eigenvectors: np.ndarray,
                   mean_face: np.ndarray,
                   out_dir: str) -> None:
    """Persist eigenvalues, eigenvectors, and mean face as CSV files."""
    # eigenvalues: one value per row, single column
    pd.DataFrame({"eigenvalue": eigenvalues}).to_csv(
        os.path.join(out_dir, "eigenvalues.csv"), index=False)

    # eigenvectors: rows = pixels (10304), cols = components
    k    = eigenvectors.shape[1]
    cols = [f"pc{i+1}" for i in range(k)]
    pd.DataFrame(eigenvectors, columns=cols).to_csv(
        os.path.join(out_dir, "eigenvectors.csv"), index=False)

    # mean face: one value per row, single column
    pd.DataFrame({"mean_pixel": mean_face}).to_csv(
        os.path.join(out_dir, "mean_face.csv"), index=False)


def load_eigen_csv(out_dir: str):
    """Load cached eigenvalues, eigenvectors, and mean face from CSV."""
    eigenvalues  = pd.read_csv(
        os.path.join(out_dir, "eigenvalues.csv"))["eigenvalue"].values
    eigenvectors = pd.read_csv(
        os.path.join(out_dir, "eigenvectors.csv")).values
    mean_face    = pd.read_csv(
        os.path.join(out_dir, "mean_face.csv"))["mean_pixel"].values
    return mean_face, eigenvalues, eigenvectors


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def show_image(vec, ax, title=""):
    ax.imshow(vec.reshape(IMG_HEIGHT, IMG_WIDTH), cmap="gray")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_mean_and_eigenfaces(mean_face, eigenvectors, n_show=10, save_path=None):
    fig, axes = plt.subplots(1, n_show + 1, figsize=(2 * (n_show + 1), 2.5))
    show_image(mean_face, axes[0], "Mean Face")
    for i in range(n_show):
        show_image(eigenvectors[:, i], axes[i + 1], f"EF {i+1}")
    fig.suptitle("Mean Face & Top Eigenfaces", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_reconstructions(D_train, mean_face, eigenvectors,
                         alphas, eigenvalues, n_subjects=5, save_path=None):
    """
    For n_subjects sample training faces, show original + reconstruction
    under each alpha threshold.
    """
    ks     = [n_components_for_variance(eigenvalues, a) for a in alphas]
    n_cols = 1 + len(alphas)
    fig    = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_subjects))
    gs     = gridspec.GridSpec(n_subjects, n_cols, figure=fig,
                               hspace=0.4, wspace=0.1)

    for row, sample_idx in enumerate(range(0, n_subjects * 2, 2)):
        ax = fig.add_subplot(gs[row, 0])
        show_image(D_train[sample_idx], ax,
                   "Original" if row == 0 else "")

        for col, (alpha, k) in enumerate(zip(alphas, ks), start=1):
            Z   = project(D_train[[sample_idx]], mean_face, eigenvectors, k)
            rec = reconstruct(Z, mean_face, eigenvectors)[0]
            ax  = fig.add_subplot(gs[row, col])
            show_image(rec, ax, f"α={alpha}\nk={k}" if row == 0 else "")

    fig.suptitle("Original vs PCA Reconstructions (Training Faces)", fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_variance_curve(eigenvalues, alphas, save_path=None):
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, linewidth=1.5)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Retained (%)")
    ax.set_title("PCA – Cumulative Explained Variance")
    ax.grid(True, alpha=0.3)

    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    for alpha, color in zip(alphas, colors):
        k = n_components_for_variance(eigenvalues, alpha)
        ax.axhline(alpha * 100, color=color, linestyle="--",
                   label=f"α={alpha} → k={k}")
        ax.axvline(k, color=color, linestyle=":")

    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load train/test splits from CSV ───────────────────────────────────────
    print("Loading train/test CSVs …")
    df_train = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
    df_test  = pd.read_csv(os.path.join(OUTPUT_DIR, "test.csv"))

    X_train = df_train.drop(columns="label").values   # (200, 10304)
    y_train = df_train["label"].values                # (200,)
    X_test  = df_test.drop(columns="label").values    # (200, 10304)
    y_test  = df_test["label"].values                 # (200,)

    print(f"  X_train: {X_train.shape}    X_test: {X_test.shape}")

    # ── Compute PCA (or load cached eigendecomposition from CSV) ─────────────
    eigen_csv   = os.path.join(OUTPUT_DIR, "eigenvalues.csv")
    evec_csv    = os.path.join(OUTPUT_DIR, "eigenvectors.csv")
    mean_csv    = os.path.join(OUTPUT_DIR, "mean_face.csv")

    if os.path.exists(eigen_csv) and os.path.exists(evec_csv) and os.path.exists(mean_csv):
        print("\nLoading cached eigendecomposition from CSV …")
        mean_face, eigenvalues, eigenvectors = load_eigen_csv(OUTPUT_DIR)
    else:
        print("\nComputing PCA from scratch (this may take a moment) …")
        mean_face, eigenvalues, eigenvectors = compute_pca(X_train)

        save_eigen_csv(eigenvalues, eigenvectors, mean_face, OUTPUT_DIR)
        print(f"  Eigendecomposition saved to {OUTPUT_DIR}/")
        print(f"    eigenvalues.csv  — shape: ({len(eigenvalues)}, 1)")
        print(f"    eigenvectors.csv — shape: {eigenvectors.shape}")
        print(f"    mean_face.csv    — shape: ({len(mean_face)}, 1)")

    print(f"\n  Mean face   : {mean_face.shape}")
    print(f"  Eigenvalues : {eigenvalues.shape}")
    print(f"  Eigenvectors: {eigenvectors.shape}  (pixels × components)")

    # ── Report dimensionality for each threshold ──────────────────────────────
    print("\n" + "─" * 52)
    print(f"{'α':>6}  {'k (components)':>16}  {'Actual variance':>16}")
    print("─" * 52)
    for alpha in ALPHAS:
        k      = n_components_for_variance(eigenvalues, alpha)
        actual = np.sum(eigenvalues[:k]) / np.sum(eigenvalues)
        print(f"  {alpha:.2f}  {k:>16d}  {actual*100:>15.2f}%")
    print("─" * 52)

    # ── Project & save model-ready CSVs for every threshold ──────────────────
    print("\nProjecting and saving reduced CSVs …")
    for alpha in ALPHAS:
        k   = n_components_for_variance(eigenvalues, alpha)
        tag = str(alpha).replace(".", "")    # e.g. 0.95 → "095"

        Z_train = project(X_train, mean_face, eigenvectors, k)
        Z_test  = project(X_test,  mean_face, eigenvectors, k)

        train_path = os.path.join(OUTPUT_DIR, f"Z_train_alpha{tag}.csv")
        test_path  = os.path.join(OUTPUT_DIR, f"Z_test_alpha{tag}.csv")

        save_projected_csv(Z_train, y_train, train_path)
        save_projected_csv(Z_test,  y_test,  test_path)

        print(f"  α={alpha}: Z_train {Z_train.shape}  Z_test {Z_test.shape}"
              f"  → Z_*_alpha{tag}.csv  (cols: label + pc1…pc{k})")

    print("""
  ── How to load in your model training script ───────────────────────
  import pandas as pd

  df    = pd.read_csv("output/Z_train_alpha095.csv")
  X     = df.drop(columns="label").values   # (200, k)
  y     = df["label"].values                # (200,)
  ────────────────────────────────────────────────────────────────────""")

    # ── Visualisations ────────────────────────────────────────────────────────
    print("Generating visualisations …")

    plot_mean_and_eigenfaces(
        mean_face, eigenvectors, n_show=10,
        save_path=os.path.join(OUTPUT_DIR, "eigenfaces.png"))

    plot_variance_curve(
        eigenvalues, ALPHAS,
        save_path=os.path.join(OUTPUT_DIR, "variance_curve.png"))

    plot_reconstructions(
        X_train, mean_face, eigenvectors, ALPHAS, eigenvalues, n_subjects=5,
        save_path=os.path.join(OUTPUT_DIR, "reconstructions.png"))

    print("\nStep 2 complete.  All outputs are in ./output/")


if __name__ == "__main__":
    main()