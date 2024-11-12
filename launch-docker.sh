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
        echo -e "${RED}Error: Could not read domain or port from settings.toml${NC}"
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
        local status=$($DOCKER_COMPOSE ps -q $service)
        if [ ! -z "$status" ]; then
            local health=$(docker inspect --format='{{.State.Health.Status}}' $status)
            if [[ "$health" == "healthy" ]]; then
                echo -e "${GREEN}Container is healthy${NC}"
                return 0
            fi
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
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Checking application readiness...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if timeout 5 curl -s http://localhost:$PORT/health >/dev/null; then
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

# Function to check tunnel readiness
check_tunnel_readiness() {
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Checking tunnel readiness...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if timeout 5 curl -s https://www.$DOMAIN/health >/dev/null; then
            echo -e "${GREEN}Tunnel is ready${NC}"
            return 0
        fi
        echo "Tunnel check attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Tunnel failed to become ready${NC}"
    $DOCKER_COMPOSE logs --tail=20 tunnel
    return 1
}

# Function to test endpoints
test_endpoints() {
    echo -e "\n${YELLOW}Testing endpoints...${NC}"
    
    # Test local /health endpoint
    echo "Testing local /health endpoint..."
    local_health_response=$(curl -s -w "\nHTTP Status: %{http_code}\n" http://localhost:$PORT/health)
    if [ $? -eq 0 ] && [ ! -z "$local_health_response" ]; then
        echo -e "${GREEN}Local /health endpoint: OK${NC}"
        echo "Response:"
        echo "$local_health_response"
    else
        echo -e "${RED}Local /health endpoint: Failed${NC}"
        return 1
    fi
    
    # Test local index endpoint
    echo -e "\nTesting local index endpoint..."
    local_index_response=$(curl -s -w "\nHTTP Status: %{http_code}\n" http://localhost:$PORT/)
    if [ $? -eq 0 ] && [ ! -z "$local_index_response" ]; then
        http_status_local=$(echo "$local_index_response" | grep "HTTP Status" | awk '{print $3}')
        echo "$local_index_response" | sed '/HTTP Status/d'
        echo -e "${GREEN}Local index endpoint: OK${NC} (HTTP Status: $http_status_local)"
    else
        echo -e "${RED}Local index endpoint: Failed${NC}"
        return 1
    fi
    
    # Test Cloudflare tunnel /health endpoint
    echo -e "\nTesting Cloudflare tunnel /health endpoint..."
    tunnel_health_response=$(curl -s -w "\nHTTP Status: %{http_code}\n" https://www.$DOMAIN/health)
    if [ $? -eq 0 ] && [ ! -z "$tunnel_health_response" ]; then
        http_status_tunnel_health=$(echo "$tunnel_health_response" | grep "HTTP Status" | awk '{print $3}')
        echo "$tunnel_health_response" | sed '/HTTP Status/d'
        echo -e "${GREEN}Cloudflare tunnel /health endpoint: OK${NC} (HTTP Status: $http_status_tunnel_health)"
    else
        echo -e "${RED}Cloudflare tunnel /health endpoint: Failed${NC}"
        echo "Note: The application is still accessible locally"
    fi

    # Test Cloudflare tunnel index endpoint
    echo -e "\nTesting Cloudflare tunnel index endpoint..."
    tunnel_index_response=$(curl -s -w "\nHTTP Status: %{http_code}\n" https://www.$DOMAIN/)
    if [ $? -eq 0 ] && [ ! -z "$tunnel_index_response" ]; then
        http_status_tunnel_index=$(echo "$tunnel_index_response" | grep "HTTP Status" | awk '{print $3}')
        echo "$tunnel_index_response" | sed '/HTTP Status/d'
        if [ "$http_status_tunnel_index" -eq 200 ]; then
            echo -e "${GREEN}Cloudflare tunnel index endpoint: OK${NC} (HTTP Status: $http_status_tunnel_index)"
        else
            echo -e "${RED}Cloudflare tunnel index endpoint: Failed${NC} (HTTP Status: $http_status_tunnel_index)"
            echo "Note: The application is still accessible locally"
        fi
    else
        echo -e "${RED}Cloudflare tunnel index endpoint: Failed${NC}"
        echo "Note: The application is still accessible locally"
    fi
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
    exit 1
fi

# Check tunnel readiness
if ! check_tunnel_readiness; then
    echo -e "${RED}Tunnel failed to establish${NC}"
    exit 1
fi

# Test endpoints
test_endpoints

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
