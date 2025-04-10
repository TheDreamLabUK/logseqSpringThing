#!/bin/bash
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Add cleanup function
cleanup() {
    log "Shutting down services..."
    if [ -n "${RUST_PID:-}" ]; then
        kill -TERM $RUST_PID || true
        wait $RUST_PID 2>/dev/null || true
    fi
    if [ -n "${VITE_PID:-}" ]; then
        kill -TERM $VITE_PID || true
        wait $VITE_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGTERM SIGINT SIGQUIT

# Verify ports are not in use
check_ports() {
    if lsof -i:4000 || lsof -i:3001; then
        log "Error: Required ports are in use"
        exit 1
    fi
}

# Start the Rust server in the background
start_rust_server() {
    log "Checking PTX file..."
    ls -l /app/src/utils/compute_forces.ptx
    log "Starting Rust server..."
    /app/webxr --gpu-debug &
    RUST_PID=$!
    
    # Wait for the Rust server to be ready
    for i in {1..30}; do
        if kill -0 $RUST_PID 2>/dev/null; then
            log "Rust server process is running (PID: $RUST_PID)"
            return 0
        fi
        log "Waiting for Rust server process... (attempt $i/30)"
        sleep 1
    done
    
    log "Error: Rust server process failed to start"
    exit 1
}

# Start Vite dev server
start_vite() {
    cd /app/client
    log "Starting Vite dev server..."
    npm run dev &
    VITE_PID=$!
}

# Main execution
check_ports
start_rust_server
start_vite

# Monitor child processes
while true; do
    if ! kill -0 $RUST_PID 2>/dev/null; then
        log "Rust server died unexpectedly"
        cleanup
    fi
    if ! kill -0 $VITE_PID 2>/dev/null; then
        log "Vite server died unexpectedly"
        cleanup
    fi
    sleep 5
done

