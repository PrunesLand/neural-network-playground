# L7: Regression Task, Ensemble Models
# This final lab script introduces regression problems, where the goal is to
# predict a continuous value instead of a class label. It also demonstrates
# the power of ensemble models, which combine multiple models to achieve
# better predictive performance.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_friedman3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- 1. Generating Synthetic Regression Problems ---
# We will use two types of datasets to highlight the strengths and weaknesses
# of different models.

# Dataset 1: Simple Linear Problem
# `make_regression` creates a simple problem where the output is a linear
# combination of the input features, plus some noise.
print("--- 1a. Generating a Simple Linear Regression Dataset ---")
X_lin, y_lin = make_regression(
    n_samples=200,
    n_features=1,
    noise=20.0,
    random_state=42
)

# Dataset 2: Complex Non-Linear Problem
# `make_friedman3` creates a more complex dataset based on a non-linear formula.
# Simple linear models will struggle with this.
print("--- 1b. Generating a Complex Non-Linear Regression Dataset ---")
X_nl, y_nl = make_friedman3(n_samples=200, random_state=42)
# For simplicity and visualization, we'll just use the first feature.
X_nl = X_nl[:, 0].reshape(-1, 1)


# --- 2. Using a Single Regressor (Linear Regression) ---
# We'll train a `LinearRegression` model and evaluate its performance.
print("\n--- 2. Training a Single Linear Regressor ---")

# We'll use the non-linear data to show the limitations of a linear model.
X_train, X_test, y_train, y_test = train_test_split(X_nl, y_nl, test_size=0.3, random_state=42)

# Initialize, train, and predict
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_lin = linear_reg.predict(X_test)

# --- 3. Evaluating Regression Quality ---
# For regression, we can't use accuracy. Instead, we use metrics that measure
# the error between predicted and true values. Mean Absolute Error (MAE) is
# the average of the absolute differences between predictions and actual values.
mae_lin = mean_absolute_error(y_test, y_pred_lin)
print(f"Mean Absolute Error (Linear Regression) on non-linear data: {mae_lin:.4f}")


# --- 4. Ensemble Models: Principle of Operation ---
# Ensemble models work by training multiple individual models (called base
# estimators) and combining their predictions. This can lead to models that
# are more accurate and robust than any single base estimator.
#
# A `RandomForestRegressor` is an ensemble model. It works by:
# 1. Building many Decision Tree models.
# 2. Training each tree on a random subset of the data samples (bagging).
# 3. Considering only a random subset of features for splitting at each node.
# 4. Averaging the predictions from all the individual trees to get the final output.

print("\n--- 4. Training an Ensemble Regressor (Random Forest) ---")
# Initialize the ensemble model with 100 decision trees.
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Mean Absolute Error (Random Forest) on non-linear data: {mae_rf:.4f}")
print("Notice the significant improvement in MAE compared to the linear model.")


# --- 5. Visualizing and Comparing Model Predictions ---
# A plot can clearly show how the ensemble model captures the non-linear
# patterns in the data that the linear model misses.
print("\n--- 5. Visualizing Model Performance ---")
plt.figure(figsize=(12, 8))

# Sort the test data for a clean line plot
sort_axis = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sort_axis]
y_test_sorted = y_test[sort_axis]
y_pred_lin_sorted = y_pred_lin[sort_axis]
y_pred_rf_sorted = y_pred_rf[sort_axis]

# Plot the original data points
plt.scatter(X_test, y_test, edgecolor='k', c='silver', s=50, label='True Values')
# Plot the Linear Regression model's predictions
plt.plot(X_test_sorted, y_pred_lin_sorted, color='red', lw=2, label=f'Linear Regression (MAE={mae_lin:.2f})')
# Plot the Random Forest model's predictions
plt.plot(X_test_sorted, y_pred_rf_sorted, color='blue', lw=2, label=f'Random Forest (MAE={mae_rf:.2f})')

plt.title("Linear Model vs. Ensemble Model on Non-Linear Data")
plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.legend()
plt.savefig("l7_regression_comparison.png")
plt.show()

print("\n--- Lab L7 Complete ---")
print("Run this script to see the regression results and the generated plot.")