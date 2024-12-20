#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check if a port is available
wait_for_port() {
    local port=$1
    local retries=60
    local wait=5
    while ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/0.0.0.0/$port" 2>/dev/null && [ $retries -gt 0 ]; do
        log "Waiting for port $port to become available... ($retries retries left)"
        sleep $wait
        retries=$((retries-1))
    done
    if [ $retries -eq 0 ]; then
        log "Timeout waiting for port $port"
        return 1
    fi
    log "Port $port is available"
    return 0
}

# Function to check RAGFlow connectivity
check_ragflow() {
    log "Checking RAGFlow connectivity..."
    if curl -s -f --max-time 5 "http://ragflow-server/v1/" > /dev/null; then
        log "RAGFlow server is reachable"
        return 0
    else
        log "Warning: Cannot reach RAGFlow server"
        return 1
    fi
}

# Function to verify production build
verify_build() {
    log "Verifying production build..."
    if [ ! -d "/app/data/public/dist" ]; then
        log "Error: Production build directory not found"
        return 1
    fi
    
    if [ ! -f "/app/data/public/dist/index.html" ]; then
        log "Error: Production build index.html not found"
        return 1
    fi
    
    log "Production build verified"
    return 0
}

# Function to verify settings file permissions
verify_settings_permissions() {
    log "Verifying settings.toml permissions..."
    
    # Check if settings.toml exists
    if [ ! -f "/app/settings.toml" ]; then
        log "Error: settings.toml not found"
        return 1
    fi
    
    # Check if file is readable
    if [ ! -r "/app/settings.toml" ]; then
        log "Error: settings.toml is not readable"
        return 1
    fi
    
    # Check if file is writable
    if [ ! -w "/app/settings.toml" ]; then
        log "Error: settings.toml is not writable"
        return 1
    fi
    
    log "settings.toml permissions verified"
    return 0
}

# Set up runtime environment
setup_runtime() {
    log "Setting up runtime environment..."

    # Set up XDG_RUNTIME_DIR
    export XDG_RUNTIME_DIR="/tmp/runtime"
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"

    # Verify GPU is available
    if ! command -v nvidia-smi &> /dev/null; then
        log "Error: nvidia-smi not found. GPU support is required."
        return 1
    fi

    # Check GPU is accessible
    if ! nvidia-smi &> /dev/null; then
        log "Error: Cannot access NVIDIA GPU. Check device is properly passed to container."
        return 1
    fi

    log "Runtime environment configured successfully"
    return 0
}

# Main script execution starts here
main() {
    # Verify settings file permissions
    if ! verify_settings_permissions; then
        log "Failed to verify settings.toml permissions"
        exit 1
    fi

    # Set up runtime environment
    if ! setup_runtime; then
        log "Failed to set up runtime environment"
        exit 1
    fi

    # Check RAGFlow availability (optional)
    if ! check_ragflow; then
        log "Warning: RAGFlow server not available - some features may be limited"
    fi

    # Verify production build
    if ! verify_build; then
        log "Failed to verify production build"
        exit 1
    fi

    # Start the Rust backend first (it needs to bind to port 3000)
    log "Starting webxr..."
    /app/webxr &
    RUST_PID=$!

    # Wait for Rust server to be ready
    if ! wait_for_port 3000; then
        log "Failed to start Rust server"
        kill $RUST_PID
        exit 1
    fi
    log "Rust server started successfully"

    # Start nginx (it needs to bind to port 4000)
    log "Starting nginx..."
    nginx -t && nginx
    if [ $? -ne 0 ]; then
        log "Failed to start nginx"
        kill $RUST_PID
        exit 1
    fi
    log "nginx started successfully"

    # Monitor both processes
    wait $RUST_PID
}

# Execute main function
main
