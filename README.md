# Neural Network Laboratories Simulation

This project is a simulation of a series of neural network laboratories, designed to provide a hands-on-keyboard experience with fundamental concepts and technologies in machine learning. Each lab is self-contained in its own directory and includes a Python script with detailed explanations and comments.

## Project Structure

The project is organized into the following directories, each corresponding to a specific lab:

- **L0/**: Introduction to NumPy and Matplotlib
- **L1/**: Data Visualization and Handling
- **L2/**: Fundamentals of Classification
- **L3/**: Building a Custom Classifier
- **L4/**: Unsupervised Learning - Clustering
- **L5/**: Data Preprocessing Techniques
- **L6/**: Statistical Analysis of Results
- **L7/**: Regression and Ensemble Models

## Getting Started

### Recommended Method (Automated)

The easiest way to run the entire lab simulation is to use the provided shell script. This will automatically install all dependencies and run each lab in sequence.

1.  **Make the script executable (if needed):**
    ```bash
    chmod +x run_all_labs.sh
    ```
2.  **Run the script:**
    ```bash
    ./run_all_labs.sh
    ```

### Manual Method

If you prefer to run each lab individually, you can follow these steps:

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Navigate to a lab directory:**
    ```bash
    cd L0
    ```

3.  **Run the Python script:**
    ```bash
    # The script name may vary per lab directory
    python l0_introduction.py
    ```

Each lab's script is designed to be run independently and will often generate plots or other output to illustrate the concepts being taught. The code is heavily commented to guide you through the process.

## Technologies Used

- **NumPy**: For numerical operations and data handling.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For machine learning algorithms, data preprocessing, and evaluation tools.
- **SciPy**: For scientific and technical computing.