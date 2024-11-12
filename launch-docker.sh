#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

    if netstat -tuln | grep -q ":4000 "; then
        local pid=$(lsof -t -i:4000)
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
        if timeout 5 curl -s http://localhost:4000/health >/dev/null; then
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
    
    # Test local endpoint
    echo "Testing local endpoint..."
    local_response=$(curl -s http://localhost:4000/health)
    if [ $? -eq 0 ] && [ ! -z "$local_response" ]; then
        echo -e "${GREEN}Local endpoint: OK${NC}"
        echo "Response: $local_response"
    else
        echo -e "${RED}Local endpoint: Failed${NC}"
        return 1
    fi
    
    # Test Cloudflare tunnel endpoint
    echo -e "\nTesting Cloudflare tunnel endpoint..."
    tunnel_response=$(curl -s --max-time 10 https://visionflow.info/health)
    if [ $? -eq 0 ] && [ ! -z "$tunnel_response" ]; then
        echo -e "${GREEN}Cloudflare tunnel: OK${NC}"
        echo "Response: $tunnel_response"
    else
        echo -e "${RED}Cloudflare tunnel: Failed${NC}"
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
