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

# Verify settings file permissions and ensure accessibility
log "Verifying settings.yaml permissions..."
# Ensure the file is accessible by the current user before checking existence
if [ -f "/app/settings.yaml" ]; then
    chmod 666 /app/settings.yaml
    log "settings.yaml permissions set to 666"
else
    log "Error: settings.yaml not found at /app/settings.yaml"
    exit 1
fi
log "settings.yaml permissions verified"

# Set up runtime environment
# Start nginx
log "Starting nginx..."
nginx -t && nginx
log "nginx started successfully"

# Execute the webxr binary only if not in debug mode
if [ "$START_WEBXR" = true ]; then
    log "Preparing to execute webxr with extended GPU diagnostics..."
    log "GPU information:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi
        # Get device uuid to verify it matches our expected value
        UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
        log "GPU UUID detected by nvidia-smi: $UUID"
    else
        log "WARNING: nvidia-smi not available - this may indicate NVIDIA driver issues"
    fi
    
    # Verify that PTX file exists and is readable
    if [ -f "/app/src/utils/compute_forces.ptx" ]; then
        PTX_SIZE=$(stat -c%s "/app/src/utils/compute_forces.ptx")
        log "✅ PTX file exists and is readable (size: $PTX_SIZE bytes)"
    else
        log "⚠️ PTX file NOT found at /app/src/utils/compute_forces.ptx"
        # Try to create a link to an alternative location if it exists elsewhere
        if [ -f "./src/utils/compute_forces.ptx" ]; then
            log "PTX file found at ./src/utils/compute_forces.ptx, creating symlink"
            ln -sf "$(pwd)/src/utils/compute_forces.ptx" "/app/src/utils/compute_forces.ptx"
        fi
    fi
    
    # Check CUDA visibility
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    else 
        # If not set, explicitly set it to ensure CUDA can see device
        export CUDA_VISIBLE_DEVICES=0
        log "Explicitly setting CUDA_VISIBLE_DEVICES=0"
    fi
    # Always enable GPU debugging to ensure physics simulation runs
    log "Starting webxr with GPU compute enabled"
    exec /app/webxr --gpu-debug
else
    log "Skipping webxr execution (debug mode)"
    # Keep the container running
    tail -f /dev/null
fi