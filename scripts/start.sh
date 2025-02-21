#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

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

# Execute the webxr binary
log "Executing webxr..."
exec /app/webxr