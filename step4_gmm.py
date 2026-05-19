import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

OUTPUT_DIR = "./output"
ALPHAS     = [0.80, 0.85, 0.90, 0.95]
K_VALUES   = [20, 40, 60]


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  GMM Implementation (EM Algorithm — from scratch)
# ═══════════════════════════════════════════════════════════════════════════════

class GMM:
    """
    Gaussian Mixture Model fitted with the Expectation-Maximisation algorithm.

    Parameters
    ----------
    k          : number of Gaussian components
    max_iters  : maximum EM iterations
    tol        : absolute log-likelihood change threshold for convergence
    reg_covar  : ridge added to every covariance diagonal for numerical stability
                 (should be on the order of the data variance, not 1e-6)
    """

    def __init__(self, k: int = 20, max_iters: int = 100,
                 tol: float = 1.0, reg_covar: float = 1e-2):
        self.k          = k
        self.max_iters  = max_iters
        self.tol        = tol
        self.reg_covar  = reg_covar

        self.weights_     = None   # (k,)
        self.means_       = None   # (k, d)
        self.covariances_ = None   # (k, d, d)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_params(self, X: np.ndarray):
        """K-Means-style initialisation: pick k random data points as means."""
        n, d = X.shape

        idx           = np.random.choice(n, self.k, replace=False)
        self.means_   = X[idx].copy()
        self.weights_ = np.ones(self.k) / self.k

        # Initialise covariances as scaled identity (scale = mean data variance)
        var_scale = np.var(X)
        self.covariances_ = np.array(
            [np.eye(d) * var_scale for _ in range(self.k)]
        )

    # ── Gaussian PDF (log-space) ──────────────────────────────────────────────

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray,
                      cov: np.ndarray) -> np.ndarray:
        """
        log N(x | mean, cov) for every row in X.
        Uses Cholesky decomposition for stability.
        """
        d    = X.shape[1]
        diff = X - mean                          # (n, d)

        try:
            L       = np.linalg.cholesky(cov)   # (d, d)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            z       = np.linalg.solve(L, diff.T) # (d, n)
            maha    = np.sum(z ** 2, axis=0)     # (n,)
        except np.linalg.LinAlgError:
            # Heavier regularisation if Cholesky still fails
            cov_reg = cov + np.eye(d) * 1.0
            L       = np.linalg.cholesky(cov_reg)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            z       = np.linalg.solve(L, diff.T)
            maha    = np.sum(z ** 2, axis=0)

        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)   # (n,)

    # ── E-step ────────────────────────────────────────────────────────────────

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Compute log-responsibilities  log r_{ik}.  Returns (n, k)."""
        n = X.shape[0]
        log_resp = np.zeros((n, self.k))

        for j in range(self.k):
            log_resp[:, j] = (np.log(self.weights_[j] + 1e-300) +
                              self._log_gaussian(X, self.means_[j],
                                                 self.covariances_[j]))

        # Log-sum-exp normalisation
        log_max  = log_resp.max(axis=1, keepdims=True)
        log_norm = np.log(np.sum(np.exp(log_resp - log_max), axis=1,
                                 keepdims=True)) + log_max
        log_resp -= log_norm
        return log_resp   # (n, k)

    # ── M-step ────────────────────────────────────────────────────────────────

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        """Update weights, means, and full covariances."""
        n, d = X.shape
        resp = np.exp(log_resp)          # (n, k)
        Nk   = resp.sum(axis=0)          # (k,)

        self.weights_ = Nk / n

        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]   # (k, d)

        for j in range(self.k):
            diff = X - self.means_[j]                      # (n, d)
            cov  = (resp[:, j:j+1] * diff).T @ diff / Nk[j]
            cov += np.eye(d) * self.reg_covar              # regularise
            self.covariances_[j] = cov


    def _log_likelihood(self, X: np.ndarray) -> float:
        n = X.shape[0]
        ll = np.zeros((n, self.k))
        for j in range(self.k):
            ll[:, j] = (np.log(self.weights_[j] + 1e-300) +
                        self._log_gaussian(X, self.means_[j],
                                           self.covariances_[j]))
        log_max  = ll.max(axis=1, keepdims=True)
        log_sum  = np.log(np.sum(np.exp(ll - log_max), axis=1)) + log_max.squeeze()
        return float(log_sum.sum())


    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit GMM via EM.  Returns hard cluster assignments (n,).
        """
        self._init_params(X)
        prev_ll   = -np.inf
        log_resp  = None

        for iteration in range(self.max_iters):
            log_resp = self._e_step(X)
            self._m_step(X, log_resp)

            ll = self._log_likelihood(X)
            delta = abs(ll - prev_ll)

            if delta < self.tol:
                print(f"      EM converged at iteration {iteration + 1}  "
                      f"(Δll = {delta:.4f})")
                break
            prev_ll = ll
        else:
            print(f"      EM reached max_iters={self.max_iters}  "
                  f"(Δll = {delta:.4f})")

        return np.argmax(log_resp, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the most probable component."""
        return np.argmax(self._e_step(X), axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Cluster → Class Mapping Helpers  (identical to K-Means step)
# ═══════════════════════════════════════════════════════════════════════════════

def get_cluster_label_mapping(clusters: np.ndarray,
                               y_true: np.ndarray, k: int) -> dict:
    """Majority-vote mapping: each cluster gets the most common true label."""
    mapping = {}
    for i in range(k):
        indices = np.where(clusters == i)[0]
        if len(indices) == 0:
            mapping[i] = -1
            continue
        actual_labels  = y_true[indices]
        majority_label = np.bincount(actual_labels).argmax()
        mapping[i]     = majority_label
    return mapping


def map_clusters_to_classes(clusters: np.ndarray, mapping: dict) -> np.ndarray:
    """Convert raw cluster IDs to predicted class labels."""
    return np.array([mapping[c] for c in clusters])


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Main — Grid Search
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results         = []
    best_model_info = {"accuracy": 0}

    print("=" * 60)
    print("  GMM Grid Search  (α × K)")
    print("=" * 60)

    for alpha in ALPHAS:
        tag        = str(alpha).replace(".", "")
        train_path = os.path.join(OUTPUT_DIR, f"Z_train_alpha{tag}.csv")
        test_path  = os.path.join(OUTPUT_DIR, f"Z_test_alpha{tag}.csv")

        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)

        X_train = df_train.drop(columns="label").values
        y_train = df_train["label"].values.astype(int)
        X_test  = df_test.drop(columns="label").values
        y_test  = df_test["label"].values.astype(int)

        print(f"\n  α = {alpha}  |  PCA dims = {X_train.shape[1]}")
        # used the PCA-projected data directly — no secondary reduction.
        # the PCA step already provides a well-conditioned, low-dimensional space.

        for k in K_VALUES:
            print(f"    K = {k}  … ", end="", flush=True)

            gmm            = GMM(k=k, max_iters=100, tol=1.0, reg_covar=1e-2)
            train_clusters = gmm.fit(X_train)

            label_mapping  = get_cluster_label_mapping(train_clusters, y_train, k)
            y_train_pred   = map_clusters_to_classes(train_clusters, label_mapping)

            test_clusters  = gmm.predict(X_test)
            y_test_pred    = map_clusters_to_classes(test_clusters, label_mapping)

            acc = accuracy_score(y_test, y_test_pred)
            results.append({"Alpha": alpha, "K": k, "Test Accuracy": acc})
            print(f"Test Acc = {acc:.4f}")

            if acc > best_model_info["accuracy"]:
                best_model_info = {
                    "accuracy":    acc,
                    "alpha":       alpha,
                    "k":           k,
                    "y_test_true": y_test,
                    "y_test_pred": y_test_pred,
                    "model":       gmm,
                }

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("  Full Results Table")
    print("=" * 60)
    print(df_results.to_string(index=False))

    df_results.to_csv(os.path.join(OUTPUT_DIR, "gmm_results.csv"), index=False)
    print(f"\n  Results saved → {OUTPUT_DIR}/gmm_results.csv")

    plt.figure(figsize=(9, 5))
    for alpha in ALPHAS:
        subset = df_results[df_results["Alpha"] == alpha]
        plt.plot(subset["K"], subset["Test Accuracy"],
                 marker="o", label=f"α = {alpha}")
    plt.title("GMM — Classification Accuracy vs Number of Components (K)")
    plt.xlabel("Number of Gaussian Components (K)")
    plt.ylabel("Test Accuracy")
    plt.xticks(K_VALUES)
    plt.legend(title="Alpha (variance retained)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gmm_accuracy_vs_k.png"), dpi=150)
    plt.show()
    print(f"  Plot saved → {OUTPUT_DIR}/gmm_accuracy_vs_k.png")

    plt.figure(figsize=(9, 5))
    for k in K_VALUES:
        subset = df_results[df_results["K"] == k]
        plt.plot(subset["Alpha"], subset["Test Accuracy"],
                 marker="s", label=f"K = {k}")
    plt.title("GMM — Classification Accuracy vs Alpha (Variance Threshold)")
    plt.xlabel("Alpha (Variance Retained)")
    plt.ylabel("Test Accuracy")
    plt.xticks(ALPHAS)
    plt.legend(title="K (components)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gmm_accuracy_vs_alpha.png"), dpi=150)
    plt.show()
    print(f"  Plot saved → {OUTPUT_DIR}/gmm_accuracy_vs_alpha.png")

    pivot = df_results.pivot(index="Alpha", columns="K", values="Test Accuracy")
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={"label": "Test Accuracy"})
    plt.title("GMM — Test Accuracy Heat-map (Alpha × K)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gmm_heatmap.png"), dpi=150)
    plt.show()
    print(f"  Heat-map saved → {OUTPUT_DIR}/gmm_heatmap.png")

    # ═══════════════════════════════════════════════════════════════════════════
    #  4.  Best Model Evaluation
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Best Model Evaluation")
    print("=" * 60)
    print(f"  Best configuration → α = {best_model_info['alpha']},  "
          f"K = {best_model_info['k']}")

    y_true   = best_model_info["y_test_true"]
    y_pred   = best_model_info["y_test_pred"]
    best_acc = best_model_info["accuracy"]
    best_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy           : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"  F1-Score (weighted): {best_f1:.4f}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title(
        f"GMM Confusion Matrix\n"
        f"α = {best_model_info['alpha']},  K = {best_model_info['k']}  |  "
        f"Acc = {best_acc:.4f}  F1 = {best_f1:.4f}"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_gmm_confusion_matrix.png"), dpi=150)
    plt.show()
    print(f"  Confusion matrix saved → {OUTPUT_DIR}/best_gmm_confusion_matrix.png")

    # ═══════════════════════════════════════════════════════════════════════════
    #  5.  GMM vs K-Means Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    kmeans_csv = os.path.join(OUTPUT_DIR, "kmeans_results.csv")
    if os.path.exists(kmeans_csv):
        df_km         = pd.read_csv(kmeans_csv)
        df_km["Model"] = "K-Means"
        df_gm          = df_results.copy()
        df_gm["Model"] = "GMM"
        df_both        = pd.concat([df_km, df_gm], ignore_index=True)

        fig, axes = plt.subplots(1, len(ALPHAS), figsize=(14, 4), sharey=True)
        for ax, alpha in zip(axes, ALPHAS):
            sub = df_both[df_both["Alpha"] == alpha]
            for model, grp in sub.groupby("Model"):
                ax.plot(grp["K"], grp["Test Accuracy"],
                        marker="o", label=model)
            ax.set_title(f"α = {alpha}")
            ax.set_xlabel("K")
            ax.set_xticks(K_VALUES)
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[0].set_ylabel("Test Accuracy")
        fig.suptitle("GMM vs K-Means — Test Accuracy Comparison", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "gmm_vs_kmeans.png"), dpi=150)
        plt.show()
        print(f"\n  Comparison plot saved → {OUTPUT_DIR}/gmm_vs_kmeans.png")
    else:
        print("\n  (K-Means results CSV not found — skipping comparison plot)")

    print("\n" + "=" * 60)
    print("  Analysis Summary")
    print("=" * 60)

    alpha_mean = df_results.groupby("Alpha")["Test Accuracy"].mean()
    print("\n  Average accuracy per Alpha (averaged over all K):")
    for alpha, acc in alpha_mean.items():
        print(f"    α = {alpha:.2f}  →  {acc:.4f}")

    k_mean = df_results.groupby("K")["Test Accuracy"].mean()
    print("\n  Average accuracy per K (averaged over all Alpha):")
    for k, acc in k_mean.items():
        print(f"    K = {k:2d}  →  {acc:.4f}")

    print("\nStep 4 (GMM) complete.  All outputs are in ./output/")


if __name__ == "__main__":
    main()