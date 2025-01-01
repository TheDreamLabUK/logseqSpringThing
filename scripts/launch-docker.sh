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
CLOUDFLARED_CONTAINER="cloudflared-tunnel"
WEBXR_SERVICE="webxr"
CLOUDFLARED_SERVICE="cloudflared"

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

# Function to check and fix directory permissions
check_fix_permissions() {
    local data_dir="$PROJECT_ROOT/data"
    local current_user=$(id -u)
    local current_group=$(id -g)

    info "Checking directory permissions..."

    # Create directories if they don't exist
    for dir in "$data_dir/markdown" "$data_dir/piper" "$data_dir/metadata"; do
        if [ ! -d "$dir" ]; then
            info "Creating directory: $dir"
            mkdir -p "$dir" || {
                error "Failed to create directory: $dir"
                return 1
            }
        fi
    done

    # Check permissions and fix if needed
    for dir in "$data_dir/markdown" "$data_dir/piper" "$data_dir/metadata"; do
        local dir_perms=$(stat -c "%a" "$dir" 2>/dev/null)
        if [ "$dir_perms" != "777" ]; then
            warn "Fixing permissions for $dir"
            sudo chmod 777 "$dir" || {
                error "Failed to set permissions on $dir"
                error "Please run: sudo chmod -R 777 $dir"
                return 1
            }
        fi
    done

    # Initialize metadata.json if it doesn't exist
    local metadata_file="$data_dir/markdown/metadata.json"
    if [ ! -f "$metadata_file" ]; then
        info "Creating metadata.json"
        echo '{}' > "$metadata_file" || {
            error "Failed to create metadata.json"
            return 1
        }
    fi

    # Ensure metadata.json is writable
    chmod 666 "$metadata_file" || {
        error "Failed to set permissions on metadata.json"
        return 1
    }

    success "Directory permissions verified"
    return 0
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

    # Check and fix permissions
    if ! check_fix_permissions; then
        error "Failed to verify/fix permissions"
        exit 1
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

# Function to start containers
start_containers() {
    info "Starting containers..."
    
    # Ensure we're using the correct UID/GID
    export DOCKER_UID=$(id -u)
    export DOCKER_GID=$(id -g)
    
    $DOCKER_COMPOSE up -d || {
        error "Failed to start containers"
        return 1
    }
    success "Containers started successfully"
}

# Function to stop containers
stop_containers() {
    info "Stopping and removing containers..."
    $DOCKER_COMPOSE down || true
}

# Function to rebuild containers
rebuild_container() {
    info "Rebuilding $WEBXR_CONTAINER..."
    
    # Stop and remove containers
    stop_containers

    # Build and start containers
    info "Starting $WEBXR_CONTAINER..."
    if ! $DOCKER_COMPOSE up -d --build; then
        error "Failed to start containers"
        return 1
    fi

    # Wait for container to be ready
    if ! wait_for_container "$WEBXR_CONTAINER"; then
        error "Container failed to become ready"
        return 1
    fi

    success "Successfully rebuilt and restarted $WEBXR_CONTAINER"
}

# Function to test backend endpoints
test_backend() {
    info "Testing backend endpoints..."

    # Test graph data endpoint
    local response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:4000/api/graph/data")
    if [ "$response" = "200" ]; then
        success "Backend graph data endpoint successful (HTTP 200)"
    else
        error "Backend graph data endpoint failed (HTTP $response)"
        return 1
    fi

    # Test settings endpoint
    info "Testing /api/settings..."
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:4000/api/settings")
    if [ "$response" = "200" ]; then
        success "Backend /api/settings successful (HTTP 200)"
    else
        error "Backend /api/settings failed (HTTP $response)"
    fi

    # Test Cloudflare tunnel
    test_cloudflare_tunnel
}

# Function to test Cloudflare tunnel
test_cloudflare_tunnel() {
    info "Testing Cloudflare tunnel..."
    
    # Wait for cloudflared container to start
    local max_attempts=30
    local attempt=1
    local delay=2
    
    info "Waiting for Cloudflare tunnel to be ready..."
    while [ $attempt -le $max_attempts ]; do
        info "Attempt $attempt/$max_attempts: Checking tunnel status..."
        
        # Check if container is running
        if ! docker ps -q -f name="^/$CLOUDFLARED_CONTAINER$" > /dev/null 2>&1; then
            warn "Cloudflared tunnel container is not running"
            sleep $delay
            ((attempt++))
            continue
        fi

        # Check if tunnel is registered
        if docker logs $CLOUDFLARED_CONTAINER 2>&1 | grep -q "Registered tunnel connection"; then
            success "Cloudflare tunnel is registered and ready"
            
            # Get tunnel hostname from config
            local tunnel_hostname=$(grep -o 'hostname: .*' config.yml | cut -d' ' -f2)
            if [ -n "$tunnel_hostname" ]; then
                success "Using tunnel hostname: $tunnel_hostname"
                
                # Test tunnel endpoint
                local response=$(curl -s -o /dev/null -w "%{http_code}" "https://$tunnel_hostname" || echo "000")
                if [ "$response" = "200" ]; then
                    success "Tunnel endpoint is accessible"
                    return 0
                else
                    error "Tunnel endpoint returned HTTP $response"
                    return 1
                fi
            else
                error "Could not find tunnel hostname in config.yml"
                return 1
            fi
        fi

        sleep $delay
        ((attempt++))
    done

    error "Could not establish Cloudflare tunnel after $max_attempts attempts"
    docker logs $CLOUDFLARED_CONTAINER | tail -n 50
    return 1
}

# Function to show endpoints
show_endpoints() {
    echo
    info "Services are running!"
    echo "HTTP:      http://localhost:4000"
    echo "WebSocket: wss://localhost:4000/wss"
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

# Function to check if the container is healthy
check_container_health() {
    local container_name="$1"
    local max_attempts="$2"
    local attempt=1

    info "Waiting for $container_name to be ready..."
    info "Attempt $attempt/$max_attempts: Waiting for container to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if [ "$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null)" = "healthy" ]; then
            success "Container $container_name is ready"
            return 0
        fi
        
        ((attempt++))
        info "Attempt $attempt/$max_attempts: Waiting for container to be healthy..."
        sleep 2
    done

    error "Container $container_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Note: The /api/settings endpoint is expected to return 500 errors
# This is intentional as server-side settings are temporarily disabled
# and all settings are managed client-side for development purposes.
# Do not attempt to fix these errors until server-side settings are re-enabled.

# Main function to handle script execution
main() {
    local command="${1:-start}"
    
    # Setup environment first
    setup_env
    
    case "$command" in
        "start")
            info "Starting containers..."
            start_containers
            if wait_for_container "$WEBXR_CONTAINER"; then
                show_endpoints
                info "Showing logs (Ctrl+C to exit)..."
                $DOCKER_COMPOSE logs -f
            fi
            ;;
        "stop")
            info "Stopping containers..."
            stop_containers
            ;;
        "restart")
            info "Restarting containers..."
            start_containers
            if wait_for_container "$WEBXR_CONTAINER"; then
                show_endpoints
            fi
            ;;
        "rebuild")
            if rebuild_container "$WEBXR_CONTAINER"; then
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
            if ! rebuild_container "$WEBXR_CONTAINER"; then
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
