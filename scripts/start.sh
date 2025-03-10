#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Check for GPU environment variables
log "Checking GPU environment variables..."

if [ -z "${NVIDIA_GPU_UUID:-}" ]; then
    # Use the specific GPU UUID that we know works
    NVIDIA_GPU_UUID="GPU-553dc306-dab3-32e2-c69b-28175a6f4da6"
    log "Setting NVIDIA_GPU_UUID to known value: $NVIDIA_GPU_UUID"
    export NVIDIA_GPU_UUID
    
    # Also set NVIDIA_VISIBLE_DEVICES to ensure Docker uses this GPU
    if [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
        export NVIDIA_VISIBLE_DEVICES="$NVIDIA_GPU_UUID"
        log "Setting NVIDIA_VISIBLE_DEVICES to: $NVIDIA_VISIBLE_DEVICES"
    fi
    
    # For older CUDA versions, also set CUDA_VISIBLE_DEVICES
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Use device index 0 since NVIDIA_VISIBLE_DEVICES will map to this
        export CUDA_VISIBLE_DEVICES="0"
        log "Setting CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"
    fi
else
    log "Using GPU UUID: $NVIDIA_GPU_UUID"
fi

# Parse command line arguments
START_WEBXR=true
if [ $# -gt 0 ] && [ "$1" = "--no-webxr" ]; then
    START_WEBXR=false
fi

# Verify settings file permissions
log "Verifying settings.yaml permissions..."
if [ ! -f "/app/settings.yaml" ]; then
    log "Error: settings.yaml not found"
    exit 1
fi
chmod 666 /app/settings.yaml
log "settings.yaml permissions verified"

# Set up runtime environment
# Start nginx
log "Starting nginx..."
nginx -t && nginx
log "nginx started successfully"

# Execute the webxr binary only if not in debug mode
if [ "$START_WEBXR" = true ]; then
    log "Executing webxr..."
    log "GPU information:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi
    else
        log "nvidia-smi not available"
    fi
    exec /app/webxr
else
    log "Skipping webxr execution (debug mode)"
    # Keep the container running
    tail -f /dev/null
fi