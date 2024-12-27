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
NC='\033[0m'

# Function to log messages with timestamps
log() {
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check pnpm security
check_pnpm_security() {
    log "${YELLOW}Running pnpm security audit...${NC}"
    
    # Run pnpm audit and capture the output
    local audit_output=$(pnpm audit 2>&1)
    local audit_exit=$?
    
    # Count critical vulnerabilities, ensuring we have a valid integer
    local critical_count
    critical_count=$(echo "$audit_output" | grep -i "critical" | grep -o '[0-9]\+ vulnerabilities' | awk '{print $1}')
    critical_count=${critical_count:-0}  # Default to 0 if empty
    
    echo "$audit_output"
    
    if [ "$critical_count" -gt 0 ]; then
        log "${RED}Found $critical_count critical vulnerabilities!${NC}"
        return 1
    elif [ "$audit_exit" -ne 0 ]; then
        log "${YELLOW}Found non-critical vulnerabilities${NC}"
    else
        log "${GREEN}No critical vulnerabilities found${NC}"
    fi
    return 0
}

# Function to check TypeScript compilation
check_typescript() {
    log "${YELLOW}Running TypeScript type check...${NC}"
    if ! pnpm run type-check; then
        log "${RED}TypeScript check failed${NC}"
        return 1
    fi
    log "${GREEN}TypeScript check passed${NC}"
    return 0
}

# Function to check Rust security
check_rust_security() {
    log "${YELLOW}Running cargo audit...${NC}"
    
    # Run cargo audit and capture the output
    local audit_output=$(cargo audit 2>&1)
    local audit_exit=$?
    
    # Count critical vulnerabilities, ensuring we have a valid integer
    local critical_count
    critical_count=$(echo "$audit_output" | grep -i "critical" | wc -l)
    critical_count=${critical_count:-0}  # Default to 0 if empty
    
    echo "$audit_output"
    
    if [ "$critical_count" -gt 0 ]; then
        log "${RED}Found $critical_count critical vulnerabilities!${NC}"
        return 1
    elif [ "$audit_exit" -ne 0 ]; then
        log "${YELLOW}Found non-critical vulnerabilities${NC}"
    else
        log "${GREEN}No critical vulnerabilities found${NC}"
    fi
    return 0
}

# Function to read settings from TOML file
read_settings() {
    local settings_file="$PROJECT_ROOT/settings.toml"
    # Extract domain and port from settings.toml
    export DOMAIN=$(grep "domain = " "$settings_file" | cut -d'"' -f2)
    export PORT=$(grep "port = " "$settings_file" | awk '{print $3}')
    
    if [ -z "$DOMAIN" ] || [ -z "$PORT" ]; then
        log "${YELLOW}Warning: DOMAIN or PORT not set in settings.toml. Using defaults.${NC}"
        DOMAIN=${DOMAIN:-"localhost"}
        PORT=${PORT:-4000}
    fi
}

# Function to check system resources
check_system_resources() {
    log "${YELLOW}Checking GPU availability...${NC}"
    if ! command -v nvidia-smi &> /dev/null; then
        log "${YELLOW}Warning: nvidia-smi not found${NC}"
        return 0
    fi
    
    # Check GPU memory
    local gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)
    echo "$gpu_info"
    
    # Check if any GPU has enough memory (at least 4GB free)
    local has_enough_memory=false
    while IFS=, read -r used total; do
        used=$(echo "$used" | tr -d ' MiB')
        total=$(echo "$total" | tr -d ' MiB')
        free=$((total - used))
        if [ "$free" -gt 4096 ]; then
            has_enough_memory=true
            break
        fi
    done <<< "$gpu_info"
    
    if [ "$has_enough_memory" = false ]; then
        log "${YELLOW}Warning: No GPU with sufficient free memory (need at least 4GB)${NC}"
    fi
    return 0
}

# Function to check Docker setup
check_docker() {
    if ! command -v docker &> /dev/null; then
        log "${YELLOW}Warning: Docker is not installed${NC}"
        DOCKER_COMPOSE="echo 'Docker not installed'"
        return 0
    fi

    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif docker-compose version &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        log "${YELLOW}Warning: Docker Compose not found${NC}"
        DOCKER_COMPOSE="echo 'Docker Compose not found'"
    fi
    return 0
}

# Function to verify client directory structure
verify_client_structure() {
    log "${YELLOW}Verifying client directory structure...${NC}"
    
    local required_files=(
        "$PROJECT_ROOT/client/index.html"
        "$PROJECT_ROOT/client/index.ts"
        "$PROJECT_ROOT/client/core/types.ts"
        "$PROJECT_ROOT/client/core/constants.ts"
        "$PROJECT_ROOT/client/core/utils.ts"
        "$PROJECT_ROOT/client/core/logger.ts"
        "$PROJECT_ROOT/client/websocket/websocketService.ts"
        "$PROJECT_ROOT/client/rendering/scene.ts"
        "$PROJECT_ROOT/client/rendering/nodes.ts"
        "$PROJECT_ROOT/client/rendering/textRenderer.ts"
        "$PROJECT_ROOT/client/state/settings.ts"
        "$PROJECT_ROOT/client/state/graphData.ts"
        "$PROJECT_ROOT/client/state/defaultSettings.ts"
        "$PROJECT_ROOT/client/xr/xrSessionManager.ts"
        "$PROJECT_ROOT/client/xr/xrInteraction.ts"
        "$PROJECT_ROOT/client/xr/xrTypes.ts"
        "$PROJECT_ROOT/client/platform/platformManager.ts"
        "$PROJECT_ROOT/client/tsconfig.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log "${RED}Error: Required file $file not found${NC}"
            return 1
        fi
    done
    
    log "${GREEN}Client directory structure verified${NC}"
    return 0
}

# Function to clean up existing processes
cleanup_existing_processes() {
    log "${YELLOW}Cleaning up...${NC}"
    
    # Save logs before cleanup if there was a failure
    if [ -n "${SAVE_LOGS:-}" ]; then
        local log_dir="$PROJECT_ROOT/logs/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$log_dir"
        $DOCKER_COMPOSE logs --no-color > "$log_dir/docker-compose.log"
        log "${YELLOW}Logs saved to $log_dir${NC}"
    fi
    
    # Stop and remove all containers from the compose project
    $DOCKER_COMPOSE down --remove-orphans --timeout 30
    
    # Clean up any orphaned containers
    for container in "logseq-xr-webxr" "cloudflared-tunnel"; do
        if docker ps -a | grep -q "$container"; then
            log "Removing container $container..."
            docker rm -f "$container" || true
        fi
    done

    # Clean up ports
    for port in $PORT 4000 3001; do
        if netstat -tuln | grep -q ":$port "; then
            local pid=$(lsof -ti ":$port")
            if [ ! -z "$pid" ]; then
                log "Killing process using port $port (PID: $pid)"
                kill -15 $pid 2>/dev/null || kill -9 $pid
            fi
        fi
    done
    
    # Clean up old volumes and images
    log "Cleaning up Docker resources..."
    docker volume ls -q | grep "logseqXR" | xargs -r docker volume rm
    docker image prune -f
    
    sleep 2
}

# Function to check RAGFlow network availability
check_ragflow_network() {
    log "${YELLOW}Checking RAGFlow network availability...${NC}"
    if ! docker network ls | grep -q "docker_ragflow"; then
        log "${RED}Error: RAGFlow network (docker_ragflow) not found${NC}"
        log "${YELLOW}Please ensure RAGFlow is running in ../ragflow/docker${NC}"
        log "${YELLOW}You can check the network with: docker network ls${NC}"
        return 1
    fi
    log "${GREEN}RAGFlow network is available${NC}"
    return 0
}

# Function to check application readiness
check_application_readiness() {
    local max_attempts=60
    local attempt=1
    local wait=2

    log "${YELLOW}Checking application readiness...${NC}"
    
    # Install websocat if not available
    if ! command -v websocat &> /dev/null; then
        log "${YELLOW}Installing websocat for WebSocket testing...${NC}"
        if command -v cargo &> /dev/null; then
            cargo install websocat
        else
            log "${RED}Error: Neither websocat nor cargo found. Cannot test WebSocket connection.${NC}"
            return 1
        fi
    fi

    while [ $attempt -le $max_attempts ]; do
        local ready=true
        local status_msg=""

        # Check HTTP endpoint
        if ! timeout 5 curl -s http://localhost:4000/ >/dev/null; then
            ready=false
            status_msg="HTTP endpoint not ready"
        fi

        # Check WebSocket endpoint
        if [ "$ready" = true ]; then
            log "${YELLOW}Testing WebSocket connection...${NC}"
            if ! timeout 5 websocat "ws://localhost:4000/wss" > /dev/null 2>&1 <<< '{"type":"ping"}'; then
                ready=false
                status_msg="WebSocket endpoint not ready"
            fi
        fi

        # Optional RAGFlow connectivity check
        if [ "$ready" = true ]; then
            if timeout 5 curl -s http://ragflow-server/v1/health >/dev/null; then
                log "${GREEN}RAGFlow service is accessible${NC}"
            else
                log "${YELLOW}Note: RAGFlow service is not accessible - some features will be limited${NC}"
            fi
        fi

        if [ "$ready" = true ]; then
            log "${GREEN}All services are ready${NC}"
            return 0
        fi

        log "${YELLOW}Attempt $attempt/$max_attempts: $status_msg${NC}"
        
        if [ $attempt -eq $((max_attempts/2)) ]; then
            log "${YELLOW}Still waiting for services. Recent logs:${NC}"
            $DOCKER_COMPOSE logs --tail=20
        fi

        sleep $wait
        attempt=$((attempt + 1))
    done

    log "${RED}Application failed to become ready. Dumping logs...${NC}"
    SAVE_LOGS=1
    $DOCKER_COMPOSE logs
    return 1
}

# Function to handle exit
handle_exit() {
    log "\n${YELLOW}Exiting immediately without cleanup${NC}"
    kill -9 $$
}

# Set up trap for immediate exit
trap handle_exit INT TERM

# Change to project root directory
cd "$PROJECT_ROOT"

# Check environment
if [ ! -f .env ]; then
    log "${YELLOW}Warning: .env file not found in $PROJECT_ROOT${NC}"
fi

# Source .env file if it exists
if [ -f .env ]; then
    set -a
    source .env || log "${YELLOW}Warning: Error sourcing .env file${NC}"
    set +a
fi

# Read settings from TOML
read_settings

# Initial setup
check_docker
check_system_resources

# Verify client structure
verify_client_structure || log "${YELLOW}Warning: Client structure verification failed${NC}"

# Run security checks
log "\n${YELLOW}Running security checks...${NC}"
check_pnpm_security || true
check_typescript || log "${YELLOW}Warning: TypeScript check failed${NC}"
check_rust_security || true

cleanup_existing_processes

# Check RAGFlow network before starting
check_ragflow_network || log "${YELLOW}Warning: RAGFlow network not available${NC}"

# Build and start services
log "${YELLOW}Building and starting services...${NC}"
$DOCKER_COMPOSE build --pull --no-cache
$DOCKER_COMPOSE up -d

# Check application readiness
check_application_readiness || log "${YELLOW}Warning: Application may not have started properly${NC}"

# Print final status
log "\n${GREEN}🚀 Services are running!${NC}"

log "\nResource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

log "\nEndpoints:"
echo "HTTP:      http://localhost:4000"
echo "WebSocket: ws://localhost:4000/wss"

log "\nCommands:"
echo "logs:    $DOCKER_COMPOSE logs -f"
echo "stop:    $DOCKER_COMPOSE down"
echo "restart: $DOCKER_COMPOSE restart"

# Keep script running to show logs
log "\n${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
$DOCKER_COMPOSE logs -f &

# Wait for signal
wait
