#!/bin/bash

# Allow commands to fail without exiting
set +e

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

# Function to check Docker container status
check_container_status() {
    local container_name="$1"
    local status
    local health
    
    # Get container status
    status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null)
    if [ -z "$status" ]; then
        error "Container $container_name not found"
        return 2
    fi
    
    # Check if container is running
    if [ "$status" != "running" ]; then
        error "Container $container_name is in state: $status"
        return 1
    fi
    
    # Get healthcheck status if available
    health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container_name" 2>/dev/null)
    
    case "$health" in
        "healthy")
            success "Container $container_name is running and healthy"
            return 0
            ;;
        "none")
            success "Container $container_name is running"
            return 0
            ;;
        *)
            if [ "$health" != "starting" ]; then
                error "Container $container_name health check failed: $health"
                return 1
            fi
            info "Container $container_name is starting up..."
            return 0
            ;;
    esac
}

# Function to check container logs for specific patterns
check_container_logs() {
    local container_name="$1"
    local success_pattern="$2"
    local error_pattern="$3"
    local timeout="$4"
    local start_time
    start_time=$(date +%s)

    while true; do
        # Check if container is still running
        if ! docker ps -q -f name="$container_name" > /dev/null; then
            error "Container $container_name is not running"
            return 1
        fi

        # Get recent logs
        local logs
        logs=$(docker logs "$container_name" --since=30s 2>&1)

        # Check for success pattern
        if echo "$logs" | grep -q "$success_pattern"; then
            success "Container $container_name is ready"
            return 0
        fi

        # Check for error pattern
        if [ -n "$error_pattern" ] && echo "$logs" | grep -q "$error_pattern"; then
            error "Container $container_name encountered an error"
            echo "$logs" | grep -C 5 "$error_pattern"
            return 1
        fi

        # Check timeout
        if [ $(($(date +%s) - start_time)) -gt "$timeout" ]; then
            error "Container $container_name failed to become ready within ${timeout}s"
            echo "Last logs:"
            echo "$logs"
            return 1
        fi

        sleep 1
    done
}

# Function to handle exit
handle_exit() {
    echo # New line for clean exit
    info "Received exit signal"
    info "You can manually stop the containers with: $DOCKER_COMPOSE down"
    info "Exiting without cleanup..."
    exit 0
}

# Set up trap for clean exit
trap handle_exit INT TERM

# Function to start and monitor containers
start_containers() {
    info "Building and starting containers..."
    
    # Build containers
    debug "Running docker compose build..."
    if ! $DOCKER_COMPOSE build --pull; then
        error "Failed to build containers"
        return 1
    fi
    success "Container build completed"

    # Start containers
    debug "Running docker compose up..."
    if ! $DOCKER_COMPOSE up -d; then
        error "Failed to start containers"
        return 1
    fi
    success "Containers started"

    # Give containers time to initialize
    info "Waiting for containers to initialize..."
    sleep 10

    # Wait for containers to be ready
    local containers=($($DOCKER_COMPOSE ps --services))
    for container in "${containers[@]}"; do
        local container_name
        case "$container" in
            "webxr")
                container_name="logseq-xr-webxr"
                ;;
            *)
                container_name="$container"
                ;;
        esac
        
        info "Checking status of $container_name..."
        if ! check_container_status "$container_name"; then
            error "Container $container_name failed to start properly"
            $DOCKER_COMPOSE logs "$container"
            return 1
        fi
    done

    return 0
}

# Main execution
main() {
    info "Starting LogseqXR services..."
    
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

    # Start containers
    if ! start_containers; then
        error "Failed to start services"
        info "Check the logs above for details"
        exit 1
    fi

    # Show endpoints
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

    # Show logs
    info "Showing logs (Ctrl+C to exit)..."
    $DOCKER_COMPOSE logs -f &
    wait
}

# Run main function
main
