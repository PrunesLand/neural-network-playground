# L5: Data Preprocessing - Normalization and Dimensionality Reduction
# This script demonstrates essential data preprocessing techniques that are
# often required before applying machine learning algorithms. We will cover:
# - Loading a built-in dataset from scikit-learn.
# - Normalizing data with StandardScaler.
# - Reducing dimensionality with Principal Component Analysis (PCA).
# - Selecting features with SelectKBest.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Loading a Dataset from Scikit-learn ---
# The `load_digits` dataset contains images of handwritten digits (0-9).
# Each image is 8x8 pixels, flattened into a 64-dimensional feature vector.
print("--- 1. Loading the Digits Dataset ---")
digits = load_digits()
X, y = digits.data, digits.target

print(f"Shape of the data (X): {X.shape}")
print(f"Number of classes: {len(digits.target_names)}")
print(f"Example of a feature vector (first sample):\n{X[0]}")

# Let's visualize a few of the digits to understand the data
fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap='binary', interpolation='nearest')
    ax.set_title(f"Label: {y[i]}")
plt.savefig("l5_digits_examples.png")
plt.show()


# --- 2. Standard Normalization using StandardScaler ---
# Many algorithms perform better when features are on a similar scale.
# StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1.
print("\n--- 2. Normalizing Data with StandardScaler ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Mean of original data (first feature): {X[:, 0].mean():.4f}")
print(f"Std of original data (first feature): {X[:, 0].std():.4f}")
print(f"Mean of scaled data (first feature): {X_scaled[:, 0].mean():.4f}")
print(f"Std of scaled data (first feature): {X_scaled[:, 0].std():.4f}")
# Note: The mean is close to 0, and the std is 1.


# --- 3. Dimensionality Reduction with PCA ---
# Principal Component Analysis (PCA) is used to reduce the number of features
# while preserving as much of the data's variance as possible.
# This is useful for visualization and can sometimes improve model performance.
print("\n--- 3. Dimensionality Reduction with PCA ---")
# We'll reduce the 64 dimensions to just 2 so we can visualize the data.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled) # Use scaled data for PCA

print(f"Original number of features: {X_scaled.shape[1]}")
print(f"Reduced number of features: {X_pca.shape[1]}")
print(f"Explained variance ratio by 2 components: {pca.explained_variance_ratio_.sum():.4f}")
print("This means our 2 components capture about 29% of the original variance.")

# Visualize the data in the 2D PCA space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='jet', alpha=0.7)
plt.title("Digits Dataset Visualized with PCA (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in digits.target_names])
plt.savefig("l5_pca_visualization.png")
plt.show()


# --- 4. Feature Selection with SelectKBest ---
# Instead of transforming features like PCA, feature selection *selects* a
# subset of the original features. `SelectKBest` selects features based on
# a specified statistical test.
# We'll use `f_classif` (ANOVA F-value) which is suitable for classification.
print("\n--- 4. Feature Selection with SelectKBest ---")
# Let's select the 20 best features out of the original 64.
k_best = SelectKBest(score_func=f_classif, k=20)
X_kbest = k_best.fit_transform(X, y) # Use original data here

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features with SelectKBest: {X_kbest.shape[1]}")

# We can see the scores for each feature
feature_scores = k_best.scores_
print(f"Scores for the first 10 features: {np.round(feature_scores[:10], 2)}")


# --- 5. Comparing Performance ---
# Let's see how these preprocessing steps affect the performance of a classifier.
# We will use a Support Vector Classifier (SVC).
print("\n--- 5. Comparing Classifier Performance ---")
# Split data for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Helper function to train and evaluate a model
def train_and_evaluate(X_train_processed, X_test_processed, y_train, y_test, description):
    model = SVC(random_state=42)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {description}: {acc:.4f}")
    return acc

# Scenario 1: Raw Data
train_and_evaluate(X_train, X_test, y_train, y_test, "Raw Data")

# Scenario 2: Scaled Data
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
train_and_evaluate(X_train_s, X_test_s, y_train, y_test, "Scaled Data")

# Scenario 3: PCA-reduced Data
pca = PCA(n_components=0.95, random_state=42).fit(X_train_s) # Keep 95% of variance
X_train_p = pca.transform(X_train_s)
X_test_p = pca.transform(X_test_s)
print(f"PCA reduced features to {X_train_p.shape[1]} components.")
train_and_evaluate(X_train_p, X_test_p, y_train, y_test, "PCA-transformed Data")

# Scenario 4: KBest-selected Features
selector = SelectKBest(f_classif, k=40).fit(X_train, y_train)
X_train_k = selector.transform(X_train)
X_test_k = selector.transform(X_test)
train_and_evaluate(X_train_k, X_test_k, y_train, y_test, "KBest-selected Features")


print("\n--- Lab L5 Complete ---")
print("Run this script to see the output and generated plots.")