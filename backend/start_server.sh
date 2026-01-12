#!/bin/bash

# AngelBot.AI Server Startup Script
# Fixes cuDNN library path conflicts and starts the server

cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Get the absolute path to cuDNN library
VENV_PATH="$(pwd)/.venv"
CUDNN_PATH="${VENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
CUDA_PATH="${VENV_PATH}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"

# Set library paths to fix conflicts between PyTorch and CTranslate2
export LD_LIBRARY_PATH="${CUDNN_PATH}:${CUDA_PATH}:${LD_LIBRARY_PATH}"

echo "==========================================="
echo "AngelBot.AI Backend Server"
echo "==========================================="
echo "Virtual Env: $VENV_PATH"
echo "cuDNN Path: $CUDNN_PATH"
echo "Starting on http://127.0.0.1:8000"
echo "==========================================="

# Start the server
python -m uvicorn main:app --reload --port 8000
