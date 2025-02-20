#!/bin/bash

# Enhanced debug information
echo "=== CUDA Environment ==="
echo "CUDA Version: $(nvcc --version | grep release)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"

echo "=== GPU Information ==="
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free,memory.used --format=csv,noheader
echo "=== GPU Processes ==="
nvidia-smi pmon -c 1

echo "=== PTX File ==="
echo "PTX File Info:"
ls -la /app/src/utils/compute_forces.ptx
file /app/src/utils/compute_forces.ptx
echo "PTX File Contents (first few lines):"
head -n 5 /app/src/utils/compute_forces.ptx

# Verify GPU access with more details
if ! nvidia-smi; then
    echo "Error: Cannot access NVIDIA GPU"
    echo "Container GPU capabilities:"
    echo "$(cat /proc/driver/nvidia/capabilities 2>/dev/null || echo 'No capabilities file found')"
    exit 1
fi

# Set additional CUDA environment variables
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/.cuda_cache
export CUDA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-0}
mkdir -p $CUDA_CACHE_PATH

# Start the application with enhanced GPU debugging
RUST_LOG=debug,cuda=trace,gpu=trace exec /app/webxr
