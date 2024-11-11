#!/bin/bash

set -e

# Define the container name
CONTAINER_NAME="logseqXR"
echo "Launching $CONTAINER_NAME with Cloudflare Tunnel..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check docker compose version
check_docker_compose() {
    if command_exists "docker compose"; then
        echo "Using Docker Compose V2"
        DOCKER_COMPOSE="docker compose"
    elif command_exists "docker-compose"; then
        echo "Using Docker Compose V1"
        DOCKER_COMPOSE="docker-compose"
    else
        echo "Error: Docker Compose not found"
        exit 1
    fi
}

# Check for required commands
if ! command_exists "docker"; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check docker compose version
check_docker_compose

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create a .env file with the necessary environment variables."
    echo "You can use .env_template as a reference."
    exit 1
fi

# Source .env file to get environment variables
set -a
source .env
set +a

# Check for required environment variables
if [ -z "$TUNNEL_TOKEN" ]; then
    echo "Error: TUNNEL_TOKEN environment variable not set in .env file."
    echo "Please add your Cloudflare Tunnel token to the .env file:"
    echo "TUNNEL_TOKEN=your-tunnel-token"
    exit 1
fi

# Check if nvidia-smi exists (GPU support)
if command_exists "nvidia-smi"; then
    echo "NVIDIA GPU detected, enabling GPU support"
    export DOCKER_GPU_FLAG="--gpus all"
else
    echo "No NVIDIA GPU detected, running without GPU support"
    export DOCKER_GPU_FLAG=""
fi

# Ensure data/markdown directory exists
mkdir -p data/markdown

# Stop and remove existing containers
echo "Stopping existing containers..."
$DOCKER_COMPOSE down --remove-orphans

# Clean up any old containers with the same name
docker ps -a | grep $CONTAINER_NAME && docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

# Build the Docker images
echo "Building Docker images..."
$DOCKER_COMPOSE build --no-cache

# Start the containers
echo "Starting containers..."
$DOCKER_COMPOSE up -d

# Function to check container health
check_container_health() {
    local container=$1
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $container to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if docker ps | grep -q $container; then
            echo "$container is running"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $container is not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "Error: $container failed to start properly"
    return 1
}

# Check health of main containers
check_container_health "${CONTAINER_NAME}"
check_container_health "cloudflared"

# If we get here, everything started successfully
echo ""
echo "🚀 Services are now running!"
echo ""
echo "📊 Application Status:"
echo "- Main application: Running on port 8080"
echo "- Cloudflare Tunnel: Active"
echo ""
echo "📝 Useful Commands:"
echo "- View all logs:           $DOCKER_COMPOSE logs -f"
echo "- View app logs:          $DOCKER_COMPOSE logs -f webxr-graph"
echo "- View tunnel logs:       $DOCKER_COMPOSE logs -f cloudflared"
echo "- Stop all services:      $DOCKER_COMPOSE down"
echo "- Restart all services:   $DOCKER_COMPOSE restart"
echo ""
echo "🌐 Access:"
echo "The application will be accessible through your Cloudflare Tunnel domain"
echo "Check the Cloudflare Zero Trust Dashboard for your tunnel status"
echo ""
echo "🔍 Monitoring logs..."
$DOCKER_COMPOSE logs -f
