# L2: Classification of Synthetic Data, Basic Experimental Protocols
# This script covers the basics of a machine learning classification experiment.
# We will generate synthetic data, split it into training and testing sets,
# train a classifier, evaluate its performance, and use cross-validation.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --- Generating Synthetic Classification Dataset ---
# `make_classification` creates a dataset with a specified number of samples,
# features, and classes. `n_informative` specifies how many features actually
# contain useful information. `random_state` ensures reproducibility.
print("--- 1. Generating Synthetic Data ---")
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=42
)

print(f"Shape of generated features (X): {X.shape}")
print(f"Shape of generated labels (y): {y.shape}")

# Visualize the generated dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("Generated Synthetic Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("L2/l2_synthetic_data.png")
plt.show()


# --- Separation of Data with train_test_split ---
# It's crucial to evaluate a classifier on data it has not seen during training.
# `train_test_split` shuffles and splits the data into a training set and a test set.
# `test_size=0.3` means 30% of the data will be used for testing.
print("\n--- 2. Splitting Data into Training and Test Sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")


# --- Initializing and Training a Classifier ---
# We will use the Gaussian Naive Bayes classifier, a simple but effective algorithm.
# The process involves two steps:
# 1. Initialization: Create an instance of the classifier.
# 2. Training: Call the `fit` method with the training data.
print("\n--- 3. Training a Gaussian Naive Bayes Classifier ---")
clf = GaussianNB()
clf.fit(X_train, y_train)
print("Classifier trained successfully.")


# --- Obtaining Predictions and Measuring Accuracy ---
# Now we use the trained classifier to make predictions on the unseen test data.
# We then compare these predictions to the true labels (`y_test`) to measure performance.
# `accuracy_score` calculates the proportion of correctly classified samples.
print("\n--- 4. Evaluating the Classifier on the Test Set ---")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Predictions on test set: {y_pred}")
print(f"True labels of test set: {y_test}")
print(f"Accuracy on the test set: {acc:.4f}")

# --- Visualizing the Decision Boundary ---
# To better understand what the classifier has learned, we can visualize its
# decision boundary. This boundary separates the regions where the classifier
# predicts different classes.
print("\n--- Visualizing the Decision Boundary ---")
fig, ax = plt.subplots(figsize=(8, 6))

# Create a meshgrid of points to classify
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Get predictions for each point in the meshgrid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot the test points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',
            marker='x', s=100, linewidth=2, label="Test Data")
ax.set_title("Classifier Decision Boundary and Test Data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
plt.savefig("L2/l2_decision_boundary.png")
plt.show()


# --- Cross-Validation ---
# A single train-test split can be sensitive to how the data is divided.
# Cross-validation provides a more robust estimate of a model's performance.
# `RepeatedKFold` splits the data into `k` folds (subsets) and repeats the process
# `n_repeats` times. In each repetition, `k-1` folds are used for training and
# 1 fold for testing.
print("\n--- 5. Cross-Validation ---")
# We use the *entire* dataset (X, y) for cross-validation.
# We'll use 5 folds and repeat the process 3 times.
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# `cross_val_score` automatically performs the cross-validation loop.
# It trains the classifier and computes the score for each fold.
# We use a new classifier instance to ensure fair evaluation.
cv_scores = cross_val_score(GaussianNB(), X, y, cv=rkf, scoring='accuracy')

print(f"Cross-validation scores (3 repeats of 5-fold CV):\n{np.round(cv_scores, 4)}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation of CV accuracy: {cv_scores.std():.4f}")

print("\n--- Lab L2 Complete ---")
print("Run this script to see the output and generated plots.")