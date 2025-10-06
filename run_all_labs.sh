#!/bin/bash

# This script automates the setup and execution of the neural network lab simulations.
# It installs the required Python libraries and then runs each lab script sequentially.

echo "--- Starting Neural Network Lab Simulation Setup ---"

# Step 1: Install dependencies
echo "Installing required Python libraries from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check your Python environment and pip."
    exit 1
fi
echo "Dependencies installed successfully."

# Step 2: Run each lab script
echo -e "\n--- Running Lab Scripts ---"

for i in {0..7}
do
    LAB_DIR="L$i"
    # Find the python script in the directory, assuming there is only one
    SCRIPT_FILE=$(find "$LAB_DIR" -maxdepth 1 -name "*.py" | head -n 1)

    if [ -f "$SCRIPT_FILE" ]; then
        echo -e "\n--- Running Lab $i: $SCRIPT_FILE ---"
        python "$SCRIPT_FILE"
        if [ $? -ne 0 ]; then
            echo "Error: Script $SCRIPT_FILE failed to execute. Aborting."
            exit 1
        fi
    else
        echo "Warning: No Python script found for Lab $i in directory $LAB_DIR."
    fi
done

echo -e "\n--- All Labs Completed Successfully! ---"