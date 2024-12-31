#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check if a port is available
check_port_available() {
    local port=$1
    local max_retries=10
    local wait=1

    log "Checking if port $port is available..."
    
    for ((i=1; i<=max_retries; i++)); do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/0.0.0.0/$port" 2>/dev/null; then
            log "Port $port is available"
            return 0
        fi
        if [ $i -lt $max_retries ]; then
            log "Port $port not ready, attempt $i of $max_retries..."
            sleep $wait
        fi
    done

    log "Error: Port $port is not available after $max_retries attempts"
    return 1
}

# Function to check service health
check_service_health() {
    local port=$1
    local endpoint=${2:-"/"}
    local retries=30
    local wait=2

    log "Checking health for service on port $port..."
    
    while [ $retries -gt 0 ]; do
        if curl -s -f --max-time 5 "http://localhost:$port$endpoint" > /dev/null; then
            log "Service on port $port is healthy"
            return 0
        fi
        
        retries=$((retries-1))
        if [ $retries -eq 0 ]; then
            log "Error: Service health check failed on port $port"
            return 1
        fi
        log "Service not ready, retrying in $wait seconds... ($retries attempts left)"
        sleep $wait
    done

    return 1
}

# Function to check RAGFlow connectivity
check_ragflow() {
    log "Checking RAGFlow connectivity..."
    if curl -s -f --max-time 5 "http://ragflow-server/v1/" > /dev/null; then
        log "RAGFlow server is reachable"
        return 0
    else
        log "Warning: RAGFlow server not available - some features may be limited"
        return 1
    fi
}

# Function to verify production build
verify_build() {
    log "Verifying production build..."
    
    # Check build directory exists and is accessible
    if [ ! -d "/app/static" ]; then
        log "Error: Production build directory not found"
        return 1
    fi
    
    # Check index.html exists and is readable
    if [ ! -r "/app/static/index.html" ]; then
        log "Error: index.html not found or not readable"
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
        log "Warning: nvidia-smi not found. GPU support may be limited."
    else
        # Check GPU is accessible
        if ! nvidia-smi &> /dev/null; then
            log "Warning: Cannot access NVIDIA GPU. Some features may be limited."
        fi
    fi

    log "Runtime environment configured successfully"
    return 0
}

# Function to cleanup processes
cleanup() {
    log "Cleaning up processes..."
    
    # Kill nginx gracefully if running
    if pgrep nginx > /dev/null; then
        log "Stopping nginx..."
        nginx -s quit
        sleep 2
        # Force kill if still running
        pkill -9 nginx || true
    fi
    
    # Kill Rust backend if running
    if [ -n "${RUST_PID:-}" ]; then
        log "Stopping Rust backend..."
        kill -TERM $RUST_PID 2>/dev/null || true
        sleep 2
        # Force kill if still running
        kill -9 $RUST_PID 2>/dev/null || true
    fi
    
    log "Cleanup complete"
}

# Main script execution starts here
main() {
    # Set up trap for cleanup
    trap cleanup EXIT INT TERM

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

    # Check RAGFlow connectivity
    check_ragflow

    # Verify production build
    if ! verify_build; then
        log "Failed to verify production build"
        exit 1
    fi

    # Check if backend port is available
    # Start webxr binary with output logging
    log "Starting webxr..."
    /app/webxr > /tmp/webxr.log 2>&1 &
    RUST_PID=$!

    # Give webxr time to initialize
    sleep 5

    # Check if process is still running
    if ! kill -0 $RUST_PID 2>/dev/null; then
        log "Error: webxr process failed to start"
        cat /tmp/webxr.log
        exit 1
    fi

    # Give the backend time to start
    log "Waiting for backend to initialize..."
    sleep 10
    
    # Check if process is still running
    if ! kill -0 $RUST_PID 2>/dev/null; then
        log "Error: Backend process died during startup"
        cat /tmp/webxr.log
        exit 1
    fi
    log "Backend process is running"

    # Start nginx
    log "Starting nginx..."
    nginx -t || { log "nginx config test failed"; kill $RUST_PID; exit 1; }
    nginx || { log "Failed to start nginx"; kill $RUST_PID; exit 1; }
    log "nginx started successfully"

    # Check frontend health
    if ! check_service_health 4000 "/"; then
        log "Error: Frontend health check failed"
        cat /tmp/webxr.log
        kill $RUST_PID
        nginx -s quit
        exit 1
    fi
    log "Frontend is healthy"

    # Wait for webxr process
    wait $RUST_PID
}

# Execute main function
main
