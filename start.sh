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

# Wait for RAGFlow to be available
log "Waiting for RAGFlow server..."
retries=24
while ! check_ragflow && [ $retries -gt 0 ]; do
    log "Retrying RAGFlow connection... ($retries attempts left)"
    retries=$((retries-1))
    sleep 5
done

if [ $retries -eq 0 ]; then
    log "Failed to connect to RAGFlow server after multiple attempts"
    exit 1
fi

# Update nginx configuration with environment variables
envsubst '${DOMAIN}' < /etc/nginx/nginx.conf > /etc/nginx/nginx.conf.tmp && mv /etc/nginx/nginx.conf.tmp /etc/nginx/nginx.conf

# Start nginx
log "Starting nginx..."
nginx -t && nginx
if [ $? -ne 0 ]; then
    log "Failed to start nginx"
    exit 1
fi
log "nginx started successfully"

# Start the Rust backend
log "Starting webxr..."
exec /app/webxr
