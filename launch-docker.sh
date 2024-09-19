#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create a .env file with the required environment variables."
    exit 1
fi

# Option to check for rebuild flag
REBUILD=false
if [ "$1" == "--rebuild" ]; then
    REBUILD=true
fi

# Stop and remove the existing container if it exists
echo "Stopping and removing existing containers..."
docker-compose down

# Optional rebuild
if [ "$REBUILD" == true ]; then
    echo "Rebuilding the Docker containers..."
    docker-compose build --no-cache
else
    echo "Skipping rebuild. Starting the containers..."
fi

# Start the container in detached mode
docker-compose up -d

# Wait for the containers to be healthy (optional)
echo "Waiting for containers to be healthy..."
docker-compose ps

echo "WebXR Graph Visualization container has been launched successfully!"
