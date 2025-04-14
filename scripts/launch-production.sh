#!/usr/bin/env bash

###############################################################################
# PRODUCTION DEPLOYMENT SCRIPT
###############################################################################
# This script builds and deploys the application in production mode.
# Key differences from development mode:
# - No volume mounts for client code (static build included in container)
# - Production optimized builds
# - Cloudflared for secure WebSocket connections
# - Proper routing through Nginx
# - GPU acceleration configured for production use
###############################################################################

###############################################################################
# SAFETY SETTINGS
###############################################################################
# -e  Exit on any command returning a non-zero status
# -u  Treat unset variables as errors
# -o pipefail  Return error if any part of a pipeline fails
set -euo pipefail

###############################################################################
# DETECT SCRIPT & PROJECT ROOT
###############################################################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

###############################################################################
# COLOR CONSTANTS
###############################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No color

###############################################################################
# CONFIGURATION
###############################################################################
# Container names for easier management
WEBXR_CONTAINER="logseq-spring-thing-webxr"
CLOUDFLARED_CONTAINER="cloudflared-tunnel"

# Docker compose file for production
DOCKER_COMPOSE_FILE="docker-compose.production.yml"

# Default CUDA architecture if not specified in .env
DEFAULT_CUDA_ARCH="89"  # Ada Lovelace architecture

###############################################################################
# LOGGING & EXIT HANDLING
###############################################################################
log() {
    # Logs a message with a timestamp
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

section() {
    # Prints a section header
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

handle_exit() {
    # Called when the script receives a signal (Ctrl+C, kill, etc.)
    log "\n${YELLOW}Exiting to shell. Containers will continue running.${NC}"
    log "${YELLOW}Use 'docker compose down' to stop containers if needed.${NC}"
    exit 0
}

# Trap Ctrl+C, kill, etc. so we can exit gracefully
trap handle_exit INT TERM

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

# Determine Docker Compose command (v1 or v2)
get_docker_compose_cmd() {
    if docker compose version &>/dev/null; then
        echo "docker compose"
    elif docker-compose version &>/dev/null; then
        echo "docker-compose"
    else
        log "${RED}Error: Docker Compose not found${NC}"
        exit 1
    fi
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

###############################################################################
# VALIDATION FUNCTIONS
###############################################################################

check_dependencies() {
    section "Checking Dependencies"

    # Check Docker
    if ! command_exists docker; then
        log "${RED}Error: Docker is not installed${NC}"
        return 1
    fi
    log "${GREEN}✓ Docker is installed${NC}"

    # Check Docker Compose
    if ! command_exists docker-compose && ! docker compose version &>/dev/null; then
        log "${RED}Error: Docker Compose not found${NC}"
        return 1
    fi
    log "${GREEN}✓ Docker Compose is installed${NC}"

    # Check NVIDIA tools for GPU support
    if ! command_exists nvidia-smi; then
        log "${YELLOW}Warning: nvidia-smi not found. GPU acceleration may not work.${NC}"
    else
        log "${GREEN}✓ NVIDIA drivers are installed${NC}"
    fi

    # Check CUDA compiler for PTX generation
    if ! command_exists nvcc; then
        log "${YELLOW}Warning: NVIDIA CUDA Compiler (nvcc) not found${NC}"
        log "${YELLOW}Will attempt to use pre-compiled PTX file if available${NC}"
    else
        log "${GREEN}✓ CUDA toolkit is installed${NC}"
    fi

    return 0
}

check_environment_file() {
    section "Checking Environment Configuration"

    # Check if .env file exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log "${RED}Error: .env file not found in $PROJECT_ROOT${NC}"
        log "${YELLOW}Please create a .env file based on .env.template${NC}"
        return 1
    fi
    log "${GREEN}✓ .env file exists${NC}"

    # Check if settings.yaml exists
    if [ ! -f "$PROJECT_ROOT/data/settings.yaml" ]; then
        log "${RED}Error: settings.yaml not found in $PROJECT_ROOT/data${NC}"
        log "${YELLOW}Please create a settings.yaml file${NC}"
        return 1
    fi
    log "${GREEN}✓ settings.yaml file exists${NC}"

    return 0
}

check_gpu_availability() {
    section "Checking GPU Availability"

    if ! command_exists nvidia-smi; then
        log "${YELLOW}Warning: Cannot check GPU availability (nvidia-smi not found)${NC}"
        return 0
    fi

    # Check if GPU is available
    # Check if nvidia-smi command runs successfully.
    # It might return non-zero even if GPUs exist but are busy/inaccessible temporarily.
    if ! nvidia-smi > /dev/null 2>&1; then
        log "${YELLOW}Warning: nvidia-smi command failed or no NVIDIA GPU detected by it.${NC}"
        log "${YELLOW}GPU acceleration features might be unavailable. Script will continue.${NC}"
        # Allow script to continue, but log the warning. Return 0 as it's non-fatal for the script logic.
        return 0
    fi

    # Get GPU info
    log "${YELLOW}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader

    # Check if GPU has enough memory (at least 2GB free)
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
    local free_memory
    free_memory=$(echo "$gpu_info" | head -n1 | grep -o '[0-9]\+')

    if [ "$free_memory" -lt 2048 ]; then
        log "${YELLOW}Warning: Less than 2GB of GPU memory available (${free_memory} MiB)${NC}"
        log "${YELLOW}Performance may be degraded${NC}"
    else
        log "${GREEN}✓ Sufficient GPU memory available (${free_memory} MiB)${NC}"
    fi

    return 0
}

check_ragflow_network() {
    section "Checking RAGFlow Network"

    if ! docker network ls | grep -q "docker_ragflow"; then
        log "${YELLOW}RAGFlow network not found, creating it...${NC}"
        docker network create docker_ragflow
        log "${GREEN}✓ Created docker_ragflow network${NC}"
    else
        log "${GREEN}✓ RAGFlow network exists${NC}"
    fi

    return 0
}

###############################################################################
# BUILD FUNCTIONS
###############################################################################

check_ptx_status() {
    section "Checking PTX Status"

    # Check if PTX file exists
    if [ -f "$PROJECT_ROOT/src/utils/compute_forces.ptx" ]; then
        log "${GREEN}✓ PTX file exists${NC}"

        # Check if source CUDA file exists
        if [ -f "$PROJECT_ROOT/src/utils/compute_forces.cu" ]; then
            # Check if PTX is older than CUDA source
            if [ "$PROJECT_ROOT/src/utils/compute_forces.ptx" -ot "$PROJECT_ROOT/src/utils/compute_forces.cu" ]; then
                log "${YELLOW}PTX file is older than CUDA source${NC}"
                log "${YELLOW}PTX will be compiled during Docker build${NC}"
                # Set flag to force PTX compilation in Docker
                export REBUILD_PTX=true
            else
                log "${GREEN}✓ PTX file is up-to-date${NC}"
            fi
        fi
    else
        log "${YELLOW}PTX file not found${NC}"
        log "${YELLOW}PTX will be compiled during Docker build${NC}"
        # Set flag to force PTX compilation in Docker
        export REBUILD_PTX=true
    fi

    return 0
}

# build_client function removed as client is built inside Dockerfile.production
build_docker_images() {
    section "Building Docker Images"

    # Get Docker Compose command
    DOCKER_COMPOSE=$(get_docker_compose_cmd)

    # Set build arguments
    export NVIDIA_GPU_UUID=${NVIDIA_GPU_UUID:-"GPU-553dc306-dab3-32e2-c69b-28175a6f4da6"}
    export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-$NVIDIA_GPU_UUID}
    export GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "production")
    export NODE_ENV=production
    export REBUILD_PTX=${REBUILD_PTX:-false}

    log "${YELLOW}Building Docker images with:${NC}"
    log "  - NVIDIA_GPU_UUID: $NVIDIA_GPU_UUID"
    log "  - GIT_HASH: $GIT_HASH"
    log "  - REBUILD_PTX: $REBUILD_PTX"

    # Build Docker images
    # Relying on exported variables from earlier 'source .env'
    $DOCKER_COMPOSE -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" build --no-cache

    log "${GREEN}✓ Docker images built successfully${NC}"

    return 0
}

###############################################################################
# DEPLOYMENT FUNCTIONS
###############################################################################

clean_existing_containers() {
    section "Cleaning Existing Containers"

    # Get Docker Compose command
    DOCKER_COMPOSE=$(get_docker_compose_cmd)

    # Stop and remove existing containers
    log "${YELLOW}Stopping and removing existing containers...${NC}"

    # Check if containers exist
    if docker ps -a --format '{{.Names}}' | grep -q "$WEBXR_CONTAINER"; then
        log "${YELLOW}Stopping and removing $WEBXR_CONTAINER...${NC}"
        docker stop "$WEBXR_CONTAINER" 2>/dev/null || true
        docker rm "$WEBXR_CONTAINER" 2>/dev/null || true
    fi

    if docker ps -a --format '{{.Names}}' | grep -q "$CLOUDFLARED_CONTAINER"; then
        log "${YELLOW}Stopping and removing $CLOUDFLARED_CONTAINER...${NC}"
        docker stop "$CLOUDFLARED_CONTAINER" 2>/dev/null || true
        docker rm "$CLOUDFLARED_CONTAINER" 2>/dev/null || true
    fi

    # Down any existing compose setup
    # More thorough cleanup: remove orphans and volumes associated with the compose file
    $DOCKER_COMPOSE -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" down --remove-orphans --volumes 2>/dev/null || true

    log "${GREEN}✓ Cleanup complete${NC}"

    return 0
}

start_containers() {
    section "Starting Containers"

    # Get Docker Compose command
    DOCKER_COMPOSE=$(get_docker_compose_cmd)

    # Start containers
    log "${YELLOW}Starting containers...${NC}"
    # Let Docker Compose load the .env file automatically
    $DOCKER_COMPOSE -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" up -d

    log "${GREEN}✓ Containers started${NC}"

    return 0
}

check_application_readiness() {
    section "Checking Application Readiness"

    local max_attempts=60
    local attempt=1
    local wait_secs=2

    log "${YELLOW}Waiting for application to be ready...${NC}"

    while [ "$attempt" -le "$max_attempts" ]; do
        # Check if containers are running
        if ! docker ps --format '{{.Names}}' | grep -q "$WEBXR_CONTAINER"; then
            log "${YELLOW}Attempt $attempt/$max_attempts: Container not running${NC}"
            sleep "$wait_secs"
            attempt=$((attempt + 1))
            continue
        fi

        # Check if HTTP endpoint is accessible
        if ! curl -s http://localhost:4000/health >/dev/null; then
            log "${YELLOW}Attempt $attempt/$max_attempts: HTTP endpoint not ready${NC}"
            sleep "$wait_secs"
            attempt=$((attempt + 1))
            continue
        fi

        # All checks passed
        log "${GREEN}✓ Application is ready${NC}"
        return 0
    done

    log "${RED}Error: Application failed to start properly${NC}"
    return 1
}

###############################################################################
# MAIN EXECUTION
###############################################################################

main() {
    section "Starting Production Deployment"

    # Change to project root
    cd "$PROJECT_ROOT"

    # Docker Compose will automatically load .env from the project root
    if [ ! -f .env ]; then
        log "${YELLOW}Warning: .env file not found in project root. CLOUDFLARE_TUNNEL_TOKEN and other variables may be missing.${NC}"
    fi

    # Run validation checks
    check_dependencies || exit 1
    check_environment_file || exit 1
    check_gpu_availability || true  # Non-fatal
    check_ragflow_network || exit 1

    # Build steps
    check_ptx_status || true  # Non-fatal, just sets flags
    # build_client call removed
    clean_existing_containers || exit 1
    build_docker_images || exit 1

    # Deployment
    start_containers || exit 1
    check_application_readiness || exit 1

    # Final status
    section "Deployment Complete"
    log "${GREEN}🚀 Application is running in production mode!${NC}"

    # Show resource usage
    log "\n${YELLOW}Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

    # Show endpoints
    log "\n${YELLOW}Endpoints:${NC}"
    echo "HTTP:      http://localhost:4000"
    echo "WebSocket: wss://localhost:4000/wss"

    # Show useful commands
    log "\n${YELLOW}Useful Commands:${NC}"
    DOCKER_COMPOSE=$(get_docker_compose_cmd)
    echo "View logs:    $DOCKER_COMPOSE -f $DOCKER_COMPOSE_FILE logs -f"
    echo "Stop:         $DOCKER_COMPOSE -f $DOCKER_COMPOSE_FILE down"
    echo "Restart:      $DOCKER_COMPOSE -f $DOCKER_COMPOSE_FILE restart"

    # Show logs
    log "\n${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
    $DOCKER_COMPOSE -f "$DOCKER_COMPOSE_FILE" logs -f &

    wait
}

# Execute main function
main
