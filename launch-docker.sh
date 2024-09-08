#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create a .env file with the required environment variables."
    exit 1
fi

# Stop and remove the existing container if it exists
docker-compose down

# Build and start the new container
docker-compose up --build -d

echo "WebXR Graph Visualization container has been launched successfully!"