#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Verify settings file permissions
log "Verifying settings.toml permissions..."
if [ ! -f "/app/settings.toml" ]; then
    log "Error: settings.toml not found"
    exit 1
fi
chmod 666 /app/settings.toml
log "settings.toml permissions verified"

# Set up runtime environment
log "Setting up runtime environment..."
mkdir -p /app/data/metadata /app/data/markdown
chmod -R 777 /app/data
log "Runtime environment configured successfully"

# Start nginx
log "Starting nginx..."
nginx -t && nginx
log "nginx started successfully"

# Execute the webxr binary
log "Executing webxr..."
exec /app/webxr
