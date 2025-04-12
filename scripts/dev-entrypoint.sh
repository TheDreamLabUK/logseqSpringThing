#!/bin/bash
set -e

# Configure log paths
LOGS_DIR="/app/logs"
RUST_LOG_FILE="${LOGS_DIR}/rust.log"
MAX_LOG_SIZE_MB=1
MAX_LOG_FILES=3

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to rotate logs
rotate_logs() {
    if [ -f "${RUST_LOG_FILE}" ]; then
        # Get file size in bytes and convert to MB
        local size_bytes=$(stat -c %s "${RUST_LOG_FILE}" 2>/dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))

        if [ "${size_mb}" -ge "${MAX_LOG_SIZE_MB}" ]; then
            log "Rotating logs (current size: ${size_mb}MB)"

            # Remove oldest log if it exists
            if [ -f "${RUST_LOG_FILE}.${MAX_LOG_FILES}" ]; then
                rm "${RUST_LOG_FILE}.${MAX_LOG_FILES}"
            fi

            # Shift all logs
            for (( i=${MAX_LOG_FILES}-1; i>=1; i-- )); do
                j=$((i+1))
                if [ -f "${RUST_LOG_FILE}.${i}" ]; then
                    mv "${RUST_LOG_FILE}.${i}" "${RUST_LOG_FILE}.${j}"
                fi
            done

            # Move current log to .1
            cp "${RUST_LOG_FILE}" "${RUST_LOG_FILE}.1"

            # Clear current log
            > "${RUST_LOG_FILE}"

            log "Log rotation complete"
        fi
    fi
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

# Start the Rust server in the background with output redirected to log file
start_rust_server() {
    log "Checking PTX file..."
    ls -l /app/src/utils/compute_forces.ptx
    log "Starting Rust server (output redirected to ${RUST_LOG_FILE})..."

    # Start Rust server with output redirected to log file
    /app/webxr --gpu-debug > "${RUST_LOG_FILE}" 2>&1 &
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

# Start Vite dev server with output to console
start_vite() {
    cd /app/client
    log "Starting Vite dev server..."

    # Run npm directly with environment variables to enhance output

    # Run with all output to console and enhanced debugging
    FORCE_COLOR=1 DEBUG=vite:* npm run dev 2>&1 &
    VITE_PID=$!

    # Give Vite a moment to start up
    sleep 2
    log "Vite dev server started (PID: $VITE_PID)"
    log "You can now access the Vite dev server at http://localhost:3001"
}

# Main execution
check_ports
start_rust_server
start_vite

# Monitor child processes and rotate logs
while true; do
    if ! kill -0 $RUST_PID 2>/dev/null; then
        log "Rust server died unexpectedly"
        cleanup
    fi
    if ! kill -0 $VITE_PID 2>/dev/null; then
        log "Vite server died unexpectedly"
        cleanup
    fi

    # Check if logs need rotation
    rotate_logs

    sleep 5
done

