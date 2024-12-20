#!/bin/bash

set -e

# Determine script location and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check pnpm security
check_pnpm_security() {
    echo -e "${YELLOW}Running pnpm security audit...${NC}"
    
    # Run pnpm audit and capture the output
    local audit_output=$(pnpm audit 2>&1)
    local audit_exit=$?
    
    # Count critical vulnerabilities, ensuring we have a valid integer
    local critical_count
    critical_count=$(echo "$audit_output" | grep -i "critical" | grep -o '[0-9]\+ vulnerabilities' | awk '{print $1}')
    critical_count=${critical_count:-0}  # Default to 0 if empty
    
    echo "$audit_output"
    
    if [ "$critical_count" -gt 0 ]; then
        echo -e "${RED}Found $critical_count critical vulnerabilities!${NC}"
        return 1
    elif [ "$audit_exit" -ne 0 ]; then
        echo -e "${YELLOW}Found non-critical vulnerabilities${NC}"
    else
        echo -e "${GREEN}No critical vulnerabilities found${NC}"
    fi
    return 0
}

# Function to check TypeScript compilation
check_typescript() {
    echo -e "${YELLOW}Running TypeScript type check...${NC}"
    if ! pnpm run type-check; then
        echo -e "${RED}TypeScript check failed${NC}"
        return 1
    fi
    echo -e "${GREEN}TypeScript check passed${NC}"
    return 0
}

# Function to check Rust security
check_rust_security() {
    echo -e "${YELLOW}Running cargo audit...${NC}"
    
    # Run cargo audit and capture the output
    local audit_output=$(cargo audit 2>&1)
    local audit_exit=$?
    
    # Count critical vulnerabilities, ensuring we have a valid integer
    local critical_count
    critical_count=$(echo "$audit_output" | grep -i "critical" | wc -l)
    critical_count=${critical_count:-0}  # Default to 0 if empty
    
    echo "$audit_output"
    
    if [ "$critical_count" -gt 0 ]; then
        echo -e "${RED}Found $critical_count critical vulnerabilities!${NC}"
        return 1
    elif [ "$audit_exit" -ne 0 ]; then
        echo -e "${YELLOW}Found non-critical vulnerabilities${NC}"
    else
        echo -e "${GREEN}No critical vulnerabilities found${NC}"
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
        echo -e "${RED}Error: DOMAIN or PORT not set in settings.toml. Please check your configuration.${NC}"
        exit 1
    fi
}

# Function to check system resources
check_system_resources() {
    echo -e "${YELLOW}Checking GPU availability...${NC}"
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found${NC}"
        exit 1
    fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

# Function to check Docker setup
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi

    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif docker-compose version &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        echo -e "${RED}Error: Docker Compose not found${NC}"
        exit 1
    fi
}

# Function to verify client directory structure
verify_client_structure() {
    echo -e "${YELLOW}Verifying client directory structure...${NC}"
    
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
            echo -e "${RED}Error: Required file $file not found${NC}"
            return 1
        fi
    done
    
    echo -e "${GREEN}Client directory structure verified${NC}"
    return 0
}

# Function to clean up existing processes
cleanup_existing_processes() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Stop and remove all containers from the compose project
    $DOCKER_COMPOSE down --remove-orphans
    
    # Explicitly remove containers if they still exist
    if docker ps -a | grep -q "logseq-xr-webxr"; then
        docker rm -f logseq-xr-webxr
    fi
    if docker ps -a | grep -q "cloudflared-tunnel"; then
        docker rm -f cloudflared-tunnel
    fi

    # Clean up port if in use
    if netstat -tuln | grep -q ":$PORT "; then
        local pid=$(lsof -ti ":$PORT")
        if [ ! -z "$pid" ]; then
            kill -9 $pid
        fi
    fi
    
    sleep 2
}

# Function to check container health
check_container_health() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Checking container health...${NC}"
    while [ $attempt -le $max_attempts ]; do
        # Check if container is running
        if ! docker ps | grep -q "logseq-xr-${service}"; then
            echo -e "${RED}Container is not running${NC}"
            return 1
        fi

        # Check container health status directly
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "logseq-xr-${service}")
        
        if [ "$health_status" = "healthy" ]; then
            echo -e "${GREEN}Container is healthy${NC}"
            return 0
        fi

        if (( attempt % 10 == 0 )); then
            echo -e "${YELLOW}Recent logs:${NC}"
            $DOCKER_COMPOSE logs --tail=10 $service
        fi

        echo "Health check attempt $attempt/$max_attempts... (status: $health_status)"
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Container failed to become healthy${NC}"
    $DOCKER_COMPOSE logs --tail=20 $service
    return 1
}

# Function to check application readiness
check_application_readiness() {
    local max_attempts=60
    local attempt=1

    echo -e "${YELLOW}Checking application readiness...${NC}"
    while [ $attempt -le $max_attempts ]; do
        # Check HTTP endpoint
        if ! timeout 5 curl -s http://localhost:4000/ >/dev/null; then
            echo "HTTP check attempt $attempt/$max_attempts..."
            sleep 2
            attempt=$((attempt + 1))
            continue
        fi

        # Check WebSocket endpoint using websocat if available
        if command -v websocat &> /dev/null; then
            echo "Testing WebSocket connection..."
            if timeout 5 websocat "ws://localhost:4000/wss" > /dev/null 2>&1 <<< '{"type":"ping"}'; then
                echo -e "${GREEN}Application is ready (HTTP + WebSocket)${NC}"
                return 0
            else
                echo "WebSocket check failed, retrying..."
            fi
        else
            echo -e "${YELLOW}websocat not found, skipping WebSocket check${NC}"
            echo -e "${GREEN}Application is ready (HTTP only)${NC}"
            return 0
        fi

        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Application failed to become ready${NC}"
    return 1
}

# Function to ensure cloudflared is running and healthy
ensure_cloudflared() {
    local max_attempts=3
    local attempt=1
    local success=false

    while [ $attempt -le $max_attempts ] && [ "$success" = false ]; do
        echo -e "\n${YELLOW}Checking cloudflared status (Attempt $attempt/$max_attempts)...${NC}"
        
        if ! docker ps | grep -q cloudflared-tunnel; then
            echo -e "${YELLOW}Cloudflared tunnel not running, starting it...${NC}"
            $DOCKER_COMPOSE up -d cloudflared
            sleep 10
        fi

        # Check container health status
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' cloudflared-tunnel 2>/dev/null || echo "unknown")
        
        if [ "$health_status" = "healthy" ]; then
            echo -e "${GREEN}Cloudflared tunnel is healthy${NC}"
            success=true
            break
        fi

        # Validate ingress configuration
        echo -e "${YELLOW}Validating cloudflared ingress configuration...${NC}"
        if ! docker exec cloudflared-tunnel cloudflared tunnel ingress validate; then
            echo -e "${RED}Ingress validation failed${NC}"
            if [ $attempt -lt $max_attempts ]; then
                echo -e "${YELLOW}Restarting cloudflared...${NC}"
                $DOCKER_COMPOSE restart cloudflared
                sleep 10
                attempt=$((attempt + 1))
                continue
            else
                echo -e "${RED}Failed to validate cloudflared configuration after $max_attempts attempts${NC}"
                return 1
            fi
        fi

        attempt=$((attempt + 1))
    done

    if [ "$success" = true ]; then
        return 0
    else
        return 1
    fi
}

# Change to project root directory
cd "$PROJECT_ROOT"

# Check environment
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found in $PROJECT_ROOT${NC}"
    exit 1
fi

# Source .env file
set -a
source .env
set +a

# Read settings from TOML
read_settings

# Initial setup
check_docker
check_system_resources

# Verify client structure
if ! verify_client_structure; then
    echo -e "${RED}Client structure verification failed${NC}"
    exit 1
fi

# Run security checks
echo -e "\n${YELLOW}Running security checks...${NC}"
check_pnpm_security || true
check_typescript || exit 1
check_rust_security || true

cleanup_existing_processes

# Clean up old resources
echo -e "${YELLOW}Cleaning up old resources...${NC}"
docker volume ls -q | grep "logseqXR" | xargs -r docker volume rm
docker image prune -f

# Build and start services
echo -e "${YELLOW}Building and starting services...${NC}"
$DOCKER_COMPOSE build --pull # consider --- no-cache
$DOCKER_COMPOSE up -d

# Check health and readiness
if ! check_container_health "webxr"; then
    echo -e "${RED}Startup failed${NC}"
    exit 1
fi

if ! check_application_readiness; then
    echo -e "${RED}Startup failed${NC}"
    $DOCKER_COMPOSE logs --tail=50 webxr
    exit 1
fi

# Ensure cloudflared is running and healthy
if ! ensure_cloudflared; then
    echo -e "${RED}Failed to ensure cloudflared is running and healthy${NC}"
    exit 1
fi

# Print final status
echo -e "\n${GREEN}🚀 Services are running!${NC}"

echo -e "\nResource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo -e "\nEndpoints:"
echo "HTTP:      http://localhost:4000"
echo "WebSocket: ws://localhost:4000/wss"

echo -e "\nCommands:"
echo "logs:    $DOCKER_COMPOSE logs -f"
echo "stop:    $DOCKER_COMPOSE down"
echo "restart: $DOCKER_COMPOSE restart"

# Handle Ctrl+C gracefully
cleanup_and_exit() {
    echo -e "\n${YELLOW}Received shutdown signal. Cleaning up...${NC}"
    $DOCKER_COMPOSE down
    exit 0
}

trap cleanup_and_exit INT TERM

# Keep script running to show logs
echo -e "\n${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
$DOCKER_COMPOSE logs -f &

# Wait for signal
wait
