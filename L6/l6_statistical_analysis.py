# L6: Statistical Analysis of Experiment Results
# This script covers the basics of using statistical tests to determine if the
# difference in performance between two machine learning models is significant.
# We will cover normality testing, interpreting p-values, and applying paired
# statistical tests.

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# --- 1. Generating Simulated Experiment Results ---
# To compare models, we need multiple performance scores. A common way to get
# these is by running repeated cross-validation. We will simulate this by
# evaluating two classifiers on the same dataset across multiple CV folds.

print("--- 1. Generating Model Performance Scores ---")
X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)

# Define the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# Initialize the classifiers we want to compare
clf1 = GaussianNB()
clf2 = DecisionTreeClassifier(random_state=42)

# Get the accuracy scores for each classifier over the CV folds
scores1 = cross_val_score(clf1, X, y, cv=cv, scoring='accuracy')
scores2 = cross_val_score(clf2, X, y, cv=cv, scoring='accuracy')

print(f"Classifier 1 (GaussianNB) Mean Accuracy: {scores1.mean():.4f} +/- {scores1.std():.4f}")
print(f"Classifier 2 (DecisionTree) Mean Accuracy: {scores2.mean():.4f} +/- {scores2.std():.4f}")


# --- 2. The Shapiro-Wilk Normality Test ---
# Parametric tests like the T-test assume that the data (or the differences,
# in our case) are drawn from a normal distribution. The Shapiro-Wilk test
# helps us check this assumption.
# The Null Hypothesis (H0) of the Shapiro-Wilk test is that the data is normally distributed.
print("\n--- 2. Checking for Normality ---")

# We are interested in the *differences* between the paired scores.
differences = scores1 - scores2

# Perform the Shapiro-Wilk test
stat, p_value_shapiro = shapiro(differences)

print(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
print(f"p-value: {p_value_shapiro:.4f}")

# --- How to Interpret p-value and alpha ---
# The significance level, alpha (α), is a threshold we choose (commonly 0.05).
# - If p-value <= alpha: We REJECT the null hypothesis. The result is "statistically significant."
# - If p-value > alpha: We FAIL TO REJECT the null hypothesis. The result is "not statistically significant."
#
# For the Shapiro-Wilk test:
# - If p <= 0.05, we reject H0 and conclude the data is NOT normal.
# - If p > 0.05, we fail to reject H0 and can assume the data IS normal.

alpha = 0.05
if p_value_shapiro > alpha:
    print("The differences appear to be normally distributed (fail to reject H0).")
    is_normal = True
else:
    print("The differences do not appear to be normally distributed (reject H0).")
    is_normal = False


# --- 3. Paired Statistical Tests ---
# We use paired tests because the scores are not independent; each pair of scores
# (score1_fold_i, score2_fold_i) was obtained on the same training/testing data split.
# The Null Hypothesis (H0) for these tests is that there is no difference between the models' performances.
print("\n--- 3. Performing a Paired Statistical Test ---")

if is_normal:
    # --- Paired Student's T-Test (Parametric) ---
    # Use this test when the differences are normally distributed.
    print("Using the Paired Student's T-Test.")
    stat, p_value_test = ttest_rel(scores1, scores2)
else:
    # --- Wilcoxon Signed-Rank Test (Non-parametric) ---
    # Use this test when the normality assumption is violated.
    print("Using the Wilcoxon Signed-Rank Test.")
    stat, p_value_test = wilcoxon(scores1, scores2)

print(f"Test Statistic: {stat:.4f}")
print(f"p-value: {p_value_test:.4f}")


# --- 4. Drawing a Conclusion ---
# Now we interpret the p-value from our comparison test (T-test or Wilcoxon).
# H0: The two models perform the same.
# H1: The two models perform differently.
print("\n--- 4. Conclusion ---")

if p_value_test <= alpha:
    print(f"p-value ({p_value_test:.4f}) is less than or equal to alpha ({alpha}).")
    print("We reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in performance between the two classifiers.")
    # We can check which one performed better on average.
    if scores1.mean() > scores2.mean():
        print("Classifier 1 (GaussianNB) is significantly better.")
    else:
        print("Classifier 2 (DecisionTree) is significantly better.")
else:
    print(f"p-value ({p_value_test:.4f}) is greater than alpha ({alpha}).")
    print("We fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference in performance between the classifiers.")

print("\n--- Lab L6 Complete ---")
print("Run this script to see how to compare models statistically.")