# L4: Unsupervised Machine Learning - Clustering
# This script introduces unsupervised learning with a focus on the k-means
# clustering algorithm. We will generate synthetic data, apply k-means to find
# clusters, visualize the results, and evaluate the quality of the clustering.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, calinski_harabasz_score

# --- 1. Generating Synthetic Data for Clustering ---
# `make_blobs` is perfect for this, as it generates isotropic Gaussian blobs,
# which are well-suited for k-means.
# `centers` defines the number of clusters to generate.
# `cluster_std` controls the spread of the clusters.
# We get back the features `X` and the true cluster labels `y_true`.
print("--- 1. Generating Synthetic Data ---")
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.80,
    random_state=42
)

print(f"Shape of generated features (X): {X.shape}")
print(f"True cluster labels (y_true): {np.unique(y_true)}")

# Visualize the raw data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', edgecolors='k')
plt.title("Generated Raw Data for Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("l4_raw_data.png")
plt.show()


# --- 2. Applying the k-Means Algorithm ---
# k-means aims to partition the data into 'k' clusters in which each observation
# belongs to the cluster with the nearest mean (cluster center or centroid).
print("\n--- 2. Applying k-Means ---")
# We need to specify `n_clusters`, the number of clusters to find.
# `n_init='auto'` is the recommended setting to handle how the initial
# centroids are chosen, making the algorithm more robust.
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')

# `fit_predict` computes cluster centers and predicts cluster index for each sample.
y_pred = kmeans.fit_predict(X)

# The algorithm's results are stored in its attributes.
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

print(f"Predicted cluster labels for first 10 samples: {labels[:10]}")
print(f"Cluster centers found by k-means:\n{cluster_centers}")


# --- 3. Visualizing the Clustering Results ---
# We can now plot the data again, but this time colored by the predicted
# cluster labels. We'll also plot the centroids found by k-means.
print("\n--- 3. Visualizing k-Means Results ---")
plt.figure(figsize=(8, 6))
# Plot the data points, colored by their assigned cluster label
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis', edgecolors='k')
# Plot the cluster centroids as red 'X's
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            c='red', s=200, marker='X', label='Centroids')
plt.title("k-Means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("l4_kmeans_clusters.png")
plt.show()


# --- 4. Measuring the Quality of Clustering ---
# Unlike classification, we often don't have true labels for clustering tasks.
# However, since we generated this data, we can use metrics that require them
# to understand how well the algorithm performed.

print("\n--- 4. Evaluating Clustering Quality ---")

# a) Rand Score: Measures the similarity between two data clusterings.
# It requires the true labels. A score of 1.0 is a perfect match.
rand = rand_score(y_true, y_pred)
print(f"Rand Score (requires true labels): {rand:.4f}")

# b) Calinski-Harabasz Score (Variance Ratio Criterion):
# This metric does *not* require true labels. It measures the ratio of the
# sum of between-cluster dispersion to within-cluster dispersion.
# Higher scores generally indicate better-defined clusters.
calinski = calinski_harabasz_score(X, y_pred)
print(f"Calinski-Harabasz Score (no true labels needed): {calinski:.4f}")
print("This score can be used to compare different clustering results on the same data.")


# --- 5. Generating a Random Integer Number ---
# A common task is to select a random sample.
print("\n--- 5. Selecting a Random Sample ---")
n_samples = X.shape[0]
random_index = np.random.randint(0, n_samples)
random_sample = X[random_index]
random_sample_cluster = y_pred[random_index]

print(f"Randomly selected sample index: {random_index}")
print(f"Random sample's features: {random_sample}")
print(f"Random sample's assigned cluster: {random_sample_cluster}")


print("\n--- Lab L4 Complete ---")
print("Run this script to see the output and generated plots.")