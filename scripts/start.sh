#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check if a service is healthy
check_service_health() {
    local port=$1
    local endpoint=${2:-"/"}
    local websocket=${3:-false}
    local retries=30
    local wait=2

    log "Checking health for service on port $port..."
    
    while [ $retries -gt 0 ]; do
        # Check if port is open
        if ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/0.0.0.0/$port" 2>/dev/null; then
            retries=$((retries-1))
            if [ $retries -eq 0 ]; then
                log "Error: Port $port is not available"
                return 1
            fi
            log "Port $port not ready, retrying in $wait seconds... ($retries attempts left)"
            sleep $wait
            continue
        fi

        # Check HTTP endpoint
        if ! curl -s -f --max-time 5 "http://localhost:$port$endpoint" > /dev/null; then
            retries=$((retries-1))
            if [ $retries -eq 0 ]; then
                log "Error: Service health check failed on port $port"
                return 1
            fi
            log "Service not ready, retrying in $wait seconds... ($retries attempts left)"
            sleep $wait
            continue
        fi

        # Check WebSocket endpoint if required
        if [ "$websocket" = true ] && ! curl -s -f --max-time 5 -N -H "Connection: Upgrade" -H "Upgrade: websocket" "http://localhost:$port/wss" > /dev/null; then
            retries=$((retries-1))
            if [ $retries -eq 0 ]; then
                log "Error: WebSocket health check failed on port $port"
                return 1
            fi
            log "WebSocket not ready, retrying in $wait seconds... ($retries attempts left)"
            sleep $wait
            continue
        fi

        log "Service on port $port is healthy"
        return 0
    done

    return 1
}

# Function to check RAGFlow connectivity with retries
check_ragflow() {
    log "Checking RAGFlow connectivity..."
    local retries=5
    local wait=10
    while [ $retries -gt 0 ]; do
        if curl -s -f --max-time 5 "http://ragflow-server/v1/" > /dev/null; then
            log "RAGFlow server is reachable"
            return 0
        else
            retries=$((retries-1))
            if [ $retries -eq 0 ]; then
                log "Warning: Cannot reach RAGFlow server after multiple attempts"
                return 1
            fi
            log "RAGFlow not ready, retrying in $wait seconds... ($retries attempts left)"
            sleep $wait
        fi
    done
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

    # Check RAGFlow availability (optional)
    if ! check_ragflow; then
        log "Warning: RAGFlow server not available - some features may be limited"
    fi

    # Verify production build
    if ! verify_build; then
        log "Failed to verify production build"
        exit 1
    fi

    # Start nginx (it needs to bind to port 4000)
    log "Starting nginx..."
    nginx -t && nginx
    if [ $? -ne 0 ]; then
        log "Failed to start nginx"
        exit 1
    fi

    # Wait for nginx to be healthy
    if ! check_service_health 4000 "/" true; then
        log "Failed to verify nginx is running"
        exit 1
    fi
    log "nginx started successfully"

    # Execute the webxr binary as the main process
    log "Executing webxr..."
    exec /app/webxr
}

# Execute main function
main
