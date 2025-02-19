#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Check GPU availability
log "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    log "Error: nvidia-smi not found. GPU support may not be available."
    exit 1
fi

# Check CUDA capabilities
log "Checking CUDA capabilities..."
nvidia-smi --query-gpu=compute_mode,memory.total,memory.free --format=csv,noheader
if [ $? -ne 0 ]; then
    log "Error: Failed to query GPU information"
    exit 1
fi

# Verify PTX file exists
log "Verifying PTX file..."
if [ ! -f "/app/compute_forces/compute_forces.ptx" ]; then
    log "Error: compute_forces.ptx not found"
    exit 1
fi

# Verify settings file permissions
log "Verifying settings.yaml permissions..."
if [ ! -f "/app/settings.yaml" ]; then
    log "Error: settings.yaml not found"
    exit 1
fi
chmod 666 /app/settings.yaml
log "settings.yaml permissions verified"

# Set GPU debugging environment variables
export RUST_LOG=${RUST_LOG:-debug}
export RUST_BACKTRACE=${RUST_BACKTRACE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}

# Log GPU configuration
log "GPU Configuration:"
log "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
log "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
log "RUST_LOG: $RUST_LOG"
log "RUST_BACKTRACE: $RUST_BACKTRACE"

# Start nginx
log "Starting nginx..."
nginx -t && nginx
log "nginx started successfully"

# Execute the webxr binary with GPU feature enabled
log "Executing webxr..."
exec /app/webxr
