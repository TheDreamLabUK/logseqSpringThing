#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

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
    exec /app/webxr
else
    log "Skipping webxr execution (debug mode)"
    # Keep the container running
    tail -f /dev/null
fi