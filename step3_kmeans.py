import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Ensure reproducibility as requested by the assignment
np.random.seed(42)

# =========================================================
# 1. K-Means Implementation
# =========================================================
class KMeans:
    def __init__(self, k=20, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        for i in range(self.max_iters):
            # 2. Assign clusters: Find distance from each point to each centroid
            distances = self._compute_distances(X)
            clusters = np.argmin(distances, axis=1)
            
            # 3. Update centroids: Calculate mean of points in each cluster
            new_centroids = np.zeros((self.k, n_features))
            for cluster_idx in range(self.k):
                points_in_cluster = X[clusters == cluster_idx]
                if len(points_in_cluster) > 0:
                    new_centroids[cluster_idx] = np.mean(points_in_cluster, axis=0)
                else:
                    # If empty, re-initialize randomly
                    new_centroids[cluster_idx] = X[np.random.choice(n_samples)]
                    
            # Check for convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
                
            self.centroids = new_centroids
            
        return clusters
    
    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def _compute_distances(self, X):
        # Efficient Euclidean distance calculation
        # (a-b)^2 = a^2 + b^2 - 2ab
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

# =========================================================
# 2. Accuracy Evaluation Helpers
# =========================================================
def get_cluster_label_mapping(clusters, y_true, k):
    """
    Since K-Means is unsupervised, we assign the most frequent 
    actual class (majority vote) in each cluster as its predicted label.
    """
    mapping = {}
    for i in range(k):
        indices = np.where(clusters == i)[0]
        if len(indices) == 0:
            mapping[i] = -1
            continue
        
        # Get the actual labels for the points in this cluster
        actual_labels = y_true[indices]
        # Find the most common label (majority vote)
        majority_label = np.bincount(actual_labels).argmax()
        mapping[i] = majority_label
        
    return mapping

def map_clusters_to_classes(clusters, mapping):
    """Transforms raw cluster IDs into predicted class labels."""
    return np.array([mapping[c] for c in clusters])

# =========================================================
# 3. Main Execution & Grid Search
# =========================================================
def main():
    OUTPUT_DIR = "./output"
    ALPHAS = [0.80, 0.85, 0.90, 0.95]
    K_VALUES = [20, 40, 60]
    
    results = []
    best_model_info = {"accuracy": 0}

    print("--- Running K-Means Grid Search ---")
    
    # Grid search over Alpha and K
    for alpha in ALPHAS:
        tag = str(alpha).replace(".", "")
        if tag == "08": tag = "080" # adjusting based on potential filename formatting
        elif tag == "09": tag = "090"
            
        # Try finding the file. We strip trailing zeros to match your pca step.
        tag = str(alpha).replace(".", "") 
        train_path = os.path.join(OUTPUT_DIR, f"Z_train_alpha{tag}.csv")
        test_path = os.path.join(OUTPUT_DIR, f"Z_test_alpha{tag}.csv")
        
        # Load PCA data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        X_train = df_train.drop(columns="label").values
        y_train = df_train["label"].values.astype(int)
        
        X_test = df_test.drop(columns="label").values
        y_test = df_test["label"].values.astype(int)
        
        for k in K_VALUES:
            kmeans = KMeans(k=k, max_iters=200)
            train_clusters = kmeans.fit(X_train)
            
            # Create mapping based on training data
            label_mapping = get_cluster_label_mapping(train_clusters, y_train, k)
            y_train_pred = map_clusters_to_classes(train_clusters, label_mapping)
            
            # Predict on test data
            test_clusters = kmeans.predict(X_test)
            y_test_pred = map_clusters_to_classes(test_clusters, label_mapping)
            
            # Calculate accuracy
            acc = accuracy_score(y_test, y_test_pred)
            results.append({"Alpha": alpha, "K": k, "Test Accuracy": acc})
            
            print(f"Alpha: {alpha:.2f} | K: {k} | Test Acc: {acc:.4f}")
            
            # Save the best model state
            if acc > best_model_info["accuracy"]:
                best_model_info = {
                    "accuracy": acc,
                    "alpha": alpha,
                    "k": k,
                    "y_test_true": y_test,
                    "y_test_pred": y_test_pred,
                    "model": kmeans
                }

    # =========================================================
    # 4. Plotting (Relating K and Alpha to Accuracy)
    # =========================================================
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_results, x="K", y="Test Accuracy", hue="Alpha", marker="o")
    plt.title("Classification Accuracy vs K (Grouped by Alpha)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Test Accuracy")
    plt.xticks(K_VALUES)
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_accuracy_plot.png"))
    plt.show()

    # =========================================================
    # 5. Best Model Evaluation
    # =========================================================
    print("\n--- Best Model Evaluation ---")
    print(f"Best Configuration -> Alpha: {best_model_info['alpha']}, K: {best_model_info['k']}")
    
    y_true = best_model_info["y_test_true"]
    y_pred = best_model_info["y_test_pred"]
    
    best_acc = best_model_info["accuracy"]
    best_f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {best_acc:.4f}")
    print(f"F1-Score (Weighted): {best_f1:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix (Alpha={best_model_info['alpha']}, K={best_model_info['k']})")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(os.path.join(OUTPUT_DIR, "best_kmeans_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    main()