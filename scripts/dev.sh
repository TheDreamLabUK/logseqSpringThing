#!/bin/bash

# Exit on error, but allow specific commands to fail
set -e

# Determine script location and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m'

# Docker compose command with version check
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Container names
WEBXR_CONTAINER="logseq-xr-webxr"
WEBXR_SERVICE="webxr"

# Function to log messages with timestamps and optional color
log() {
    local color="${2:-$NC}"
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] ${color}$1${NC}"
}

# Function to log debug messages
debug() {
    if [ "${DEBUG:-0}" = "1" ]; then
        log "$1" "$GRAY"
    fi
}

# Function to log error messages
error() {
    log "ERROR: $1" "$RED"
}

# Function to log warning messages
warn() {
    log "WARNING: $1" "$YELLOW"
}

# Function to log success messages
success() {
    log "SUCCESS: $1" "$GREEN"
}

# Function to log info messages
info() {
    log "INFO: $1" "$BLUE"
}

# Function to check if container exists
container_exists() {
    local container_name="$1"
    docker ps -a -q -f name="^/${container_name}$" > /dev/null 2>&1
}

# Function to check if container is running
container_is_running() {
    local container_name="$1"
    docker ps -q -f name="^/${container_name}$" > /dev/null 2>&1
}

# Function to setup environment
setup_env() {
    # Change to project root
    cd "$PROJECT_ROOT" || {
        error "Failed to change to project root directory"
        exit 1
    }

    # Check environment
    if [ ! -f .env ]; then
        warn ".env file not found in $PROJECT_ROOT"
    else
        debug "Loading .env file..."
        set -a
        source .env || warn "Error sourcing .env file"
        set +a
    fi
}

# Function to wait for container to be ready
wait_for_container() {
    local container_name="$1"
    local max_attempts=30
    local attempt=1
    local delay=2
    
    info "Waiting for $container_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        # Check if container exists
        if ! container_exists "$container_name"; then
            error "Container $container_name does not exist"
            return 1
        fi
        
        # Check if container is running
        if ! container_is_running "$container_name"; then
            warn "Container $container_name is not running (attempt $attempt/$max_attempts)"
            sleep $delay
            ((attempt++))
            continue
        fi
        
        # Check health status
        local health
        health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container_name" 2>/dev/null)
        
        case "$health" in
            "healthy")
                if docker logs "$container_name" 2>&1 | grep -q "Frontend is healthy"; then
                    success "Container $container_name is ready"
                    return 0
                fi
                ;;
            "none")
                if docker logs "$container_name" 2>&1 | grep -q "Frontend is healthy"; then
                    success "Container $container_name is ready"
                    return 0
                fi
                ;;
        esac
        
        info "Attempt $attempt/$max_attempts: Waiting for container to be healthy..."
        sleep $delay
        ((attempt++))
    done
    
    error "Timed out waiting for $container_name to be ready"
    docker logs "$container_name" 2>&1 | tail -n 50
    return 1
}

# Function to rebuild and restart container
rebuild_container() {
    local container_name="$1"
    local service_name="$2"
    info "Rebuilding $container_name..."
    
    # Stop and remove all containers and volumes
    info "Stopping and removing containers..."
    $DOCKER_COMPOSE down -v
    docker rm -f "$container_name" 2>/dev/null || true
    docker volume prune -f
    
    # Build the image
    if ! $DOCKER_COMPOSE build "$service_name"; then
        error "Failed to build $container_name"
        return 1
    fi
    
    # Start the container
    info "Starting $container_name..."
    if ! $DOCKER_COMPOSE up -d "$service_name"; then
        error "Failed to start $container_name"
        return 1
    fi
    
    # Wait for container to be ready
    if ! wait_for_container "$container_name"; then
        return 1
    fi
    
    success "Successfully rebuilt and restarted $container_name"
    return 0
}

# Function to test backend endpoints
test_backend() {
    local failed=0
    info "Testing backend endpoints..."
    
    # Test graph data endpoint
    local status
    status=$(docker exec $WEBXR_CONTAINER curl -s -o /dev/null -w "%{http_code}" \
        --max-time 5 \
        -H "Accept: application/json" "http://localhost:3001/api/graph/data")
    if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
        success "Backend graph data endpoint successful (HTTP $status)"
    else
        error "Backend graph data endpoint failed (HTTP $status)"
        ((failed++))
    fi
    
    # Test settings endpoints
    local settings_endpoints=(
        "/api/settings"
        "/api/settings/"
        "/api/settings/visualization"
        "/api/settings/websocket"
        "/api/settings/system"
        "/api/settings/all"
    )
    
    for endpoint in "${settings_endpoints[@]}"; do
        info "Testing $endpoint..."
        status=$(docker exec $WEBXR_CONTAINER curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            -H "Accept: application/json" "http://localhost:3001$endpoint")
        if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
            success "Backend $endpoint successful (HTTP $status)"
        else
            error "Backend $endpoint failed (HTTP $status)"
            ((failed++))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        success "All backend tests passed"
        return 0
    else
        error "$failed backend tests failed"
        return 1
    fi
}

# Function to show endpoints
show_endpoints() {
    echo
    info "Services are running!"
    echo "HTTP:      http://localhost:4000"
    echo "WebSocket: ws://localhost:4000/wss"
    echo
    info "Available commands:"
    echo "logs:    $DOCKER_COMPOSE logs -f"
    echo "stop:    $DOCKER_COMPOSE down"
    echo "restart: $DOCKER_COMPOSE restart"
    echo
}

# Function to handle cleanup on exit
cleanup() {
    info "Cleaning up..."
    if [ "${DEBUG:-0}" = "1" ]; then
        $DOCKER_COMPOSE logs
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Main function
main() {
    local command="${1:-start}"
    
    # Setup environment first
    setup_env
    
    case "$command" in
        "start")
            info "Starting containers..."
            $DOCKER_COMPOSE up -d
            if wait_for_container "$WEBXR_CONTAINER"; then
                show_endpoints
                info "Showing logs (Ctrl+C to exit)..."
                $DOCKER_COMPOSE logs -f
            fi
            ;;
        "stop")
            info "Stopping containers..."
            $DOCKER_COMPOSE down
            ;;
        "restart")
            info "Restarting containers..."
            $DOCKER_COMPOSE up -d
            if wait_for_container "$WEBXR_CONTAINER"; then
                show_endpoints
            fi
            ;;
        "rebuild")
            if rebuild_container "$WEBXR_CONTAINER" "$WEBXR_SERVICE"; then
                show_endpoints
            fi
            ;;
        "test")
            if ! wait_for_container "$WEBXR_CONTAINER"; then
                exit 1
            fi
            test_backend
            ;;
        "rebuild-test")
            if ! rebuild_container "$WEBXR_CONTAINER" "$WEBXR_SERVICE"; then
                exit 1
            fi
            test_backend
            ;;
        "logs")
            info "Showing logs (Ctrl+C to exit)..."
            $DOCKER_COMPOSE logs -f
            ;;
        *)
            error "Unknown command: $command"
            echo "Usage: $0 [start|stop|restart|rebuild|test|rebuild-test|logs]"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
