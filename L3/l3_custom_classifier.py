# L3: Implementing Custom Classifiers
# This script demonstrates how to create your own classifier using scikit-learn's
# base classes. It also covers imbalanced data, the difference between accuracy
# and balanced accuracy, and multi-class classification.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# --- Creating a Custom Classifier in Python ---
# To create a scikit-learn compatible classifier, we need to:
# 1. Create a class for our classifier.
# 2. Inherit from `BaseEstimator` and `ClassifierMixin`.
#    - `BaseEstimator` provides `get_params` and `set_params` methods.
#    - `ClassifierMixin` provides a `score` method (which uses accuracy).
# 3. Implement the following methods:
#    - `__init__`: The constructor.
#    - `fit(X, y)`: The training method.
#    - `predict(X)`: The prediction method.

class NearestCentroidClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple classifier that assigns a sample to the class of the nearest centroid.
    The centroid is the mean of the feature vectors of all samples in a class.
    """
    def __init__(self):
        # No hyperparameters to set in this simple example.
        pass

    def fit(self, X, y):
        """
        Trains the classifier by calculating the centroid for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        # 1. Validate the input data
        X, y = check_X_y(X, y)

        # 2. Store the unique class labels found in the data
        self.classes_ = unique_labels(y)

        # 3. Calculate and store the centroids
        self.centroids_ = []
        for cls in self.classes_:
            # Find all samples belonging to the current class
            X_cls = X[y == cls]
            # Calculate the mean of their features (the centroid)
            centroid = X_cls.mean(axis=0)
            self.centroids_.append(centroid)
        self.centroids_ = np.array(self.centroids_)

        # Return the classifier
        return self

    def predict(self, X):
        """
        Makes predictions for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        # 1. Check if `fit` has been called
        check_is_fitted(self)

        # 2. Validate the input
        X = check_array(X)

        # 3. Calculate distances from each sample to each centroid
        # `cdist` calculates the distance between each pair of the two collections of inputs.
        distances = cdist(X, self.centroids_, metric='euclidean')

        # 4. Find the index of the closest centroid for each sample
        closest_centroid_indices = np.argmin(distances, axis=1)

        # 5. Return the corresponding class labels
        return self.classes_[closest_centroid_indices]


# --- 1. Testing the Custom Classifier on a Balanced Dataset ---
print("--- 1. Testing on a Balanced Dataset ---")
X_bal, y_bal = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)

# Instantiate, train, and predict
custom_clf = NearestCentroidClassifier()
custom_clf.fit(X_train, y_train)
y_pred = custom_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on balanced data: {acc:.4f}")


# --- 2. Imbalanced Data and Balanced Accuracy ---
# Accuracy can be misleading on imbalanced datasets. If 95% of samples are
# class A, a model that always predicts A will have 95% accuracy but is useless.
# Balanced accuracy is the average of recall obtained on each class.
print("\n--- 2. Handling Imbalanced Data ---")
X_imb, y_imb = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42 # 90% class 0, 10% class 1
)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42)

# Train and predict on the imbalanced data
custom_clf.fit(X_train_i, y_train_i)
y_pred_i = custom_clf.predict(X_test_i)

# Calculate both accuracy and balanced accuracy
acc_imb = accuracy_score(y_test_i, y_pred_i)
bal_acc_imb = balanced_accuracy_score(y_test_i, y_pred_i)

print(f"Class distribution in test set: {np.bincount(y_test_i)}")
print(f"Accuracy on imbalanced data: {acc_imb:.4f}")
print(f"Balanced Accuracy on imbalanced data: {bal_acc_imb:.4f}")
print("Notice how balanced accuracy gives a more realistic performance measure.")


# --- 3. Multi-Class Classification ---
# Our classifier should work for more than two classes without any changes,
# because it iterates through all unique classes it finds in `y`.
print("\n--- 3. Handling Multi-Class Data ---")
X_multi, y_multi = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=4, random_state=42
)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)

# Train and predict on the multi-class data
custom_clf.fit(X_train_m, y_train_m)
y_pred_m = custom_clf.predict(X_test_m)
acc_m = accuracy_score(y_test_m, y_pred_m)
print(f"Accuracy on 4-class data: {acc_m:.4f}")

# Visualize the decision boundary for the multi-class case
print("Visualizing multi-class decision boundary...")
fig, ax = plt.subplots(figsize=(8, 6))
x_min, x_max = X_multi[:, 0].min() - 1, X_multi[:, 0].max() + 1
y_min, y_max = X_multi[:, 1].min() - 1, X_multi[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = custom_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
ax.scatter(X_test_m[:, 0], X_test_m[:, 1], c=y_test_m, s=20, edgecolor='k', cmap='viridis')
ax.set_title("Decision Boundary for Custom Classifier (Multi-Class)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.savefig("L3/l3_multiclass_boundary.png")
plt.show()

print("\n--- Lab L3 Complete ---")
print("Run this script to see the output and generated plot.")