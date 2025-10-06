# L0: Introduction to NumPy and Matplotlib
# This script provides a basic introduction to the NumPy and Matplotlib libraries.
# It covers creating arrays, performing operations on them, sampling from random
# distributions, and creating simple visualizations.

# --- Importing Libraries ---
# We begin by importing the necessary libraries.
# numpy is conventionally imported as np, and matplotlib.pyplot as plt.
import numpy as np
import matplotlib.pyplot as plt

# --- Creating NumPy Arrays ---
# NumPy's primary object is the homogeneous multidimensional array.
print("--- NumPy Array Creation ---")

# Create a 1D array from a Python list
a = np.array([1, 2, 3, 4, 5])
print(f"1D Array: {a}")
print(f"Array type: {a.dtype}")

# Create a 2D array and specify the data type
b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
print(f"2D Array:\n{b}")
print(f"Array type: {b.dtype}")

# Create an array of zeros
c = np.zeros((2, 3))
print(f"Array of zeros:\n{c}")

# Create an array of ones
d = np.ones((3, 2))
print(f"Array of ones:\n{d}")

# Create an array with a range of elements
e = np.arange(10, 20, 2) # (start, stop, step)
print(f"Ranged array: {e}")

# --- Operations on Arrays ---
# NumPy allows for powerful element-wise operations.
print("\n--- Array Operations ---")

# Basic arithmetic
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(f"Addition:\n{arr1 + arr2}")
print(f"Multiplication:\n{arr1 * arr2}")

# Operations along an axis
# axis=0 operates on columns, axis=1 operates on rows
print(f"Sum of all elements: {arr1.sum()}")
print(f"Sum of columns: {arr1.sum(axis=0)}")
print(f"Mean of rows: {arr1.mean(axis=1)}")

# --- Sampling from Random Distributions ---
# NumPy's random module provides functions for sampling from various distributions.
print("\n--- Random Sampling ---")

# Sample from a uniform distribution over [0, 1)
uniform_samples = np.random.rand(5)
print(f"Uniform samples: {uniform_samples}")

# Sample from a standard normal distribution (mean=0, variance=1)
normal_samples = np.random.randn(5)
print(f"Normal samples: {normal_samples}")

# Sample integers from a given range
random_integers = np.random.randint(0, 10, 5) # (low, high, size)
print(f"Random integers: {random_integers}")

# --- Matplotlib Visualization ---
# Matplotlib is a comprehensive library for creating static, animated, and
# interactive visualizations in Python.
print("\n--- Matplotlib Visualization ---")

# Generate some data to visualize
# Two clusters of points
cluster1_x = np.random.normal(2, 1, 50) # mean=2, std=1, 50 points
cluster1_y = np.random.normal(2, 1, 50)
cluster2_x = np.random.normal(6, 1, 50)
cluster2_y = np.random.normal(6, 1, 50)

# 1. Create a figure and an axes object
# The figure is the overall window or page that everything is drawn on.
# The axes is the area on which the data is plotted.
fig, ax = plt.subplots()

# 2. Visualize the data with a scatter plot
# We plot each cluster with a different color and label.
ax.scatter(cluster1_x, cluster1_y, color='blue', label='Cluster 1')
ax.scatter(cluster2_x, cluster2_y, color='red', label='Cluster 2')

# 3. Customize the plot
ax.set_title("Scatter Plot of Two Clusters")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend() # Display the labels for each dataset
ax.grid(True) # Add a grid for better readability

# 4. Save the figure to a file
# This is useful for including plots in reports or presentations.
plt.savefig("L0/l0_scatter_plot.png")
print("Plot saved to L0/l0_scatter_plot.png")

# 5. Show the plot
# This will open a window displaying the plot.
plt.show()

print("\n--- Lab L0 Complete ---")
print("Run this script to see the output and the generated plot.")