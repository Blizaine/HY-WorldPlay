#!/bin/bash

echo "========================================"
echo "HY-WorldPlay 1.5 GUI Launcher"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH"
    exit 1
fi

# Check if worldplay environment exists
if ! conda env list | grep -q "^worldplay "; then
    echo "Creating conda environment 'worldplay'..."
    conda create --name worldplay python=3.10 -y
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment"
        exit 1
    fi
fi

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate worldplay

echo
echo "Checking dependencies..."
if ! python -c "import gradio" 2>/dev/null; then
    echo "Installing required packages..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements"
        exit 1
    fi
fi

echo
echo "========================================"
echo "Starting HY-WorldPlay GUI..."
echo "========================================"
echo
echo "The interface will open in your browser at:"
echo "http://localhost:7860"
echo
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo

python app_gradio.py