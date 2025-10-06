# L1: Samples, Features, Classes - Visualizing and Understanding Data
# This script covers loading data from a CSV file, indexing NumPy arrays,
# handling subplots in Matplotlib, and calculating Euclidean distance.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- Loading Data from CSV using NumPy ---
# We'll load the data from the 'l1_data.csv' file.
# np.loadtxt is a simple way to load data from a text file.
# We skip the first row (header) and use ',' as the delimiter.
print("--- Loading Data ---")
data = np.loadtxt('L1/l1_data.csv', delimiter=',', skiprows=1)

# The data is now in a NumPy array.
# The first two columns are features, and the third is the class label.
X = data[:, :2]  # All rows, columns 0 and 1 (Features)
y = data[:, 2]   # All rows, column 2 (Class labels)

print(f"Shape of features (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")
print(f"First 5 samples:\n{X[:5]}")
print(f"First 5 labels: {y[:5]}")

# --- Indexing NumPy Arrays ---
# Indexing is crucial for accessing and manipulating data.
print("\n--- Array Indexing ---")

# Get the first sample (row)
first_sample = X[0, :]
print(f"First sample (features only): {first_sample}")

# Get the feature values for all samples belonging to class 0
class_0_features = X[y == 0]
print(f"Number of samples in class 0: {len(class_0_features)}")
print(f"Features of the first 3 samples in class 0:\n{class_0_features[:3]}")

# --- Handling Subplots in Matplotlib ---
# Subplots allow you to place multiple plots in a single figure.
# This is useful for comparing different views of the data.
print("\n--- Matplotlib Subplots ---")

# Create a figure with 2 rows and 2 columns of subplots
# `figsize` controls the overall size of the figure.
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Data Visualization with Subplots', fontsize=16)

# Top-left subplot: Scatter plot of all data
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')
axs[0, 0].set_title('All Classes')
axs[0, 0].set_xlabel('Feature 1')
axs[0, 0].set_ylabel('Feature 2')

# Top-right subplot: Histogram of Feature 1 for each class
for i in np.unique(y):
    axs[0, 1].hist(X[y == i, 0], alpha=0.5, label=f'Class {int(i)}')
axs[0, 1].set_title('Histogram of Feature 1')
axs[0, 1].set_xlabel('Feature 1 Value')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].legend()

# Bottom-left subplot: Histogram of Feature 2 for each class
for i in np.unique(y):
    axs[1, 0].hist(X[y == i, 1], alpha=0.5, label=f'Class {int(i)}')
axs[1, 0].set_title('Histogram of Feature 2')
axs[1, 0].set_xlabel('Feature 2 Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()

# Bottom-right subplot: Empty (can be used for text or other plots)
axs[1, 1].axis('off') # Turn off the axis lines and labels
axs[1, 1].text(0.5, 0.5, 'Subplots provide a\nway to organize\nmultiple views of data.',
               ha='center', va='center', fontsize=12)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("L1/l1_subplots.png")
print("Subplots figure saved to L1/l1_subplots.png")
plt.show()


# --- Calculating Euclidean Distance ---
# Euclidean distance is a common way to measure the similarity between two points.
# The formula for distance between p and q is sqrt(sum((pi - qi)^2)).
print("\n--- Euclidean Distance ---")

# Take two points from our dataset
p1 = X[0]  # First sample
p2 = X[1]  # Second sample

# Manual calculation
manual_dist = np.sqrt(np.sum((p1 - p2)**2))
print(f"Manually calculated distance between p1 and p2: {manual_dist:.4f}")

# Using np.linalg.norm (vector norm)
norm_dist = np.linalg.norm(p1 - p2)
print(f"Distance using np.linalg.norm: {norm_dist:.4f}")

# Using scipy's cdist function
# cdist is highly efficient for computing distances between all pairs of points
# in two collections. We need to reshape our points to be 2D arrays.
p1_reshaped = p1.reshape(1, -1)
p2_reshaped = p2.reshape(1, -1)
cdist_dist = cdist(p1_reshaped, p2_reshaped, 'euclidean')
print(f"Distance using scipy.spatial.distance.cdist: {cdist_dist[0, 0]:.4f}")

# Calculate the distance matrix between the first 5 points
dist_matrix = cdist(X[:5], X[:5], 'euclidean')
print("\nDistance matrix between the first 5 points:")
print(np.round(dist_matrix, 2))


print("\n--- Lab L1 Complete ---")
print("Run this script to see the output and the generated plot.")