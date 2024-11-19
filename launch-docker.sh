#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to read settings from TOML file
read_settings() {
    # Extract domain and port from settings.toml
    export DOMAIN=$(grep "domain = " settings.toml | cut -d'"' -f2)
    export PORT=$(grep "port = " settings.toml | awk '{print $3}')
    
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

# Function to clean up existing processes
cleanup_existing_processes() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    $DOCKER_COMPOSE down --remove-orphans >/dev/null 2>&1

    if netstat -tuln | grep -q ":$PORT "; then
        local pid=$(lsof -t -i:"$PORT")
        if [ ! -z "$pid" ]; then
            kill -9 $pid >/dev/null 2>&1
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
        if ! docker ps | grep -q "logseq-xr-${service}-1"; then
            echo -e "${RED}Container is not running${NC}"
            return 1
        fi

        # Check container logs for successful startup
        if docker logs "logseq-xr-${service}-1" 2>&1 | grep -q "Port 3000 is available"; then
            echo -e "${GREEN}Container startup completed${NC}"
            # Add delay to allow nginx to fully initialize
            sleep 5
            return 0
        fi

        if (( attempt % 10 == 0 )); then
            echo -e "${YELLOW}Recent logs:${NC}"
            $DOCKER_COMPOSE logs --tail=10 $service
        fi

        echo "Health check attempt $attempt/$max_attempts..."
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
        if timeout 5 curl -s http://localhost:4000/ >/dev/null; then
            echo -e "${GREEN}Application is ready${NC}"
            return 0
        fi
        echo "Readiness check attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Application failed to become ready${NC}"
    return 1
}

# Function to test endpoints
test_endpoints() {
    echo -e "\n${YELLOW}Testing endpoints...${NC}"
    
    # Test local index endpoint
    echo -e "\nTesting local index endpoint..."
    local_index_response=$(curl -s -w "\nHTTP Status: %{http_code}\n" http://localhost:4000/)
    if [ $? -eq 0 ] && [ ! -z "$local_index_response" ]; then
        http_status_local=$(echo "$local_index_response" | grep "HTTP Status" | awk '{print $3}')
        echo "$local_index_response" | sed '/HTTP Status/d'
        echo -e "${GREEN}Local index endpoint: OK${NC} (HTTP Status: $http_status_local)"
    else
        echo -e "${RED}Local index endpoint: Failed${NC}"
        return 1
    fi
}

# Function to check cloudflared tunnel connectivity
check_cloudflared_connectivity() {
    echo -e "\n${YELLOW}Checking Cloudflared tunnel connectivity...${NC}"
    
    # Check if cloudflared container is running
    if ! docker ps | grep -q cloudflared-tunnel; then
        echo -e "${RED}Cloudflared tunnel container is not running${NC}"
        return 1
    fi

    # Wait for tunnel connections with timeout
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for tunnel connections to establish..."
    while [ $attempt -le $max_attempts ]; do
        # Check for active tunnel connections
        local connection_count=$(docker logs cloudflared-tunnel 2>&1 | grep -c "Registered tunnel connection")
        local error_count=$(docker logs cloudflared-tunnel 2>&1 | grep -c "no more connections active and exiting")
        
        if [ $connection_count -gt 0 ] && [ $error_count -eq 0 ]; then
            echo -e "${GREEN}Tunnel connections established successfully${NC}"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Failed to establish tunnel connections${NC}"
    return 1
}

# Check environment
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
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
cleanup_existing_processes

# Clean up old resources
echo -e "${YELLOW}Cleaning up old resources...${NC}"
docker volume ls -q | grep "logseqXR" | xargs -r docker volume rm >/dev/null 2>&1
docker image prune -f >/dev/null 2>&1

# Ensure data directory exists
mkdir -p data/markdown

# Build and start services
echo -e "${YELLOW}Building and starting services...${NC}"
$DOCKER_COMPOSE build --pull --no-cache
$DOCKER_COMPOSE up -d

# Check health and readiness
if ! check_container_health "webxr-graph"; then
    echo -e "${RED}Startup failed${NC}"
    exit 1
fi

if ! check_application_readiness; then
    echo -e "${RED}Startup failed${NC}"
    $DOCKER_COMPOSE logs --tail=50 webxr-graph
    exit 1
fi

# Test endpoints
test_endpoints

# Validate and restart cloudflared tunnel
echo -e "\n${YELLOW}Validating Cloudflared ingress configuration...${NC}"
docker exec cloudflared-tunnel cloudflared tunnel ingress validate
if [ $? -ne 0 ]; then
    echo -e "${RED}Cloudflared ingress validation failed. Please check your config.yml.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Restarting Cloudflared tunnel...${NC}"
if docker ps | grep -q cloudflared-tunnel; then
    docker restart cloudflared-tunnel
    # Add delay to allow cloudflared to fully initialize
    sleep 5
else
    echo -e "${RED}Cloudflared tunnel container is not running. Starting it now.${NC}"
    $DOCKER_COMPOSE up -d cloudflared
    sleep 5
fi

# Check cloudflared connectivity
if ! check_cloudflared_connectivity; then
    echo -e "${RED}Cloudflared tunnel connectivity check failed${NC}"
    echo -e "${YELLOW}Showing recent cloudflared logs:${NC}"
    docker logs --tail 50 cloudflared-tunnel
    exit 1
fi

# Print final status
echo -e "\n${GREEN}🚀 Services are running!${NC}"

echo -e "\nResource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo -e "\nCommands:"
echo "logs:    $DOCKER_COMPOSE logs -f"
echo "stop:    $DOCKER_COMPOSE down"
echo "restart: $DOCKER_COMPOSE restart"

# Keep script running to show logs
echo -e "\n${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
$DOCKER_COMPOSE logs -f