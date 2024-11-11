#!/bin/bash

set -e

# Define the container name
CONTAINER_NAME="logseqXR"
echo "Launching $CONTAINER_NAME with Cloudflare Tunnel..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check system resources
check_system_resources() {
    echo -e "${YELLOW}Checking system resources...${NC}"
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    if [ $cpu_cores -lt 20 ]; then
        echo -e "${RED}Warning: Less than 20 CPU cores available ($cpu_cores cores)${NC}"
        echo "Consider adjusting CPU limits in docker-compose.yml"
    fi
    
    # Check available memory
    local mem_available=$(free -g | awk '/^Mem:/ {print $7}')
    if [ $mem_available -lt 68 ]; then
        echo -e "${RED}Warning: Less than 68GB RAM available ($mem_available GB)${NC}"
        echo "Consider adjusting memory limits in docker-compose.yml"
    fi
    
    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Warning: nvidia-smi not found. GPU support may not be available.${NC}"
    else
        local gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
        if [ $gpu_count -eq 0 ]; then
            echo -e "${RED}Error: No NVIDIA GPUs detected${NC}"
            exit 1
        fi
        echo -e "${GREEN}Found $gpu_count NVIDIA GPU(s)${NC}"
        nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader
    fi
}

# Function to check Docker setup
check_docker() {
    echo -e "${YELLOW}Checking Docker setup...${NC}"
    
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

# Function to check container health
check_container_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Checking $service health...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if $DOCKER_COMPOSE ps $service | grep -q "running"; then
            local health=$($DOCKER_COMPOSE ps $service | grep -o "(healthy)\|unhealthy" || echo "no health check")
            if [[ $health == "(healthy)" || $health == "no health check" ]]; then
                echo -e "${GREEN}$service is running and healthy${NC}"
                return 0
            fi
        fi
        echo "Attempt $attempt/$max_attempts: Waiting for $service..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}Error: $service failed to start properly${NC}"
    return 1
}

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found. Please create a .env file with the necessary environment variables.${NC}"
    echo "You can use .env_template as a reference."
    exit 1
fi

# Source .env file
set -a
source .env
set +a

# Check for Cloudflare Tunnel token
if [ -z "$TUNNEL_TOKEN" ]; then
    echo -e "${RED}Error: TUNNEL_TOKEN environment variable not set in .env file.${NC}"
    echo "Please add your Cloudflare Tunnel token to the .env file:"
    echo "TUNNEL_TOKEN=your-tunnel-token"
    exit 1
fi

# Run initial checks
check_docker
check_system_resources

# Create Docker network if it doesn't exist
docker network create logseq-net 2>/dev/null || true

# Stop and remove existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
$DOCKER_COMPOSE down --remove-orphans

# Ensure all related containers are removed
echo "Cleaning up any lingering containers..."
docker rm -f logseqxr-tunnel logseqXR-tunnel 2>/dev/null || true
docker rm -f $(docker ps -a | grep 'logseq' | awk '{print $1}') 2>/dev/null || true

# Clean up old images
echo -e "${YELLOW}Cleaning up old images...${NC}"
docker image prune -f

# Ensure data directory exists
mkdir -p data/markdown

# Build and start services
echo -e "${YELLOW}Building and starting services...${NC}"
$DOCKER_COMPOSE build --pull --no-cache
$DOCKER_COMPOSE up -d

# Check health of services
check_container_health "webxr-graph"
check_container_health "cloudflared"

# Print resource usage
echo ""
echo "🖥️  Current Resource Usage:"
echo "==========================="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo ""
echo -e "${GREEN}🚀 Services are now running!${NC}"
echo ""
echo "📊 Status Dashboard:"
echo "- Main application: Running on port 8080"
echo "- Cloudflare Tunnel: Active"
echo "- GPU Support: Enabled for main application"
echo ""
echo "📝 Useful Commands:"
echo "- View all logs:           $DOCKER_COMPOSE logs -f"
echo "- View app logs:          $DOCKER_COMPOSE logs -f webxr-graph"
echo "- View tunnel logs:       $DOCKER_COMPOSE logs -f cloudflared"
echo "- Check GPU usage:        nvidia-smi"
echo "- Stop all services:      $DOCKER_COMPOSE down"
echo "- Restart all services:   $DOCKER_COMPOSE restart"
echo ""
echo "🌐 Access:"
echo "The application will be accessible through your Cloudflare Tunnel domain"
echo "Check the Cloudflare Zero Trust Dashboard for your tunnel status"
echo ""
echo -e "${YELLOW}🔍 Monitoring logs...${NC}"
$DOCKER_COMPOSE logs -f
