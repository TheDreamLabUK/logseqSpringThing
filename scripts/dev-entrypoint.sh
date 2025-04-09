#!/bin/bash
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Start the Rust server in the background
log "Checking PTX file..."
ls -l /app/src/utils/compute_forces.ptx
log "Starting Rust server..."
/app/webxr --gpu-debug &

# Wait for the Rust server to be ready
while ! nc -z localhost 4000; do
    log "Waiting for Rust server..."
    sleep 1
done
log "Rust server is ready!"

# Start Vite dev server
cd /app/client
log "Starting Vite dev server..."
npm run dev

# Keep container running if Vite exits
tail -f /dev/null

