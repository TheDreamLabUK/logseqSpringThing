#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create a .env file with the required environment variables."
    exit 1
fi

# Stop and remove the existing container if it exists
echo "Stopping and removing existing containers..."
docker stop logseqXR &>/dev/null
docker rm logseqXR &>/dev/null

# Build and start the new container with maximal debug
echo "Building and starting new container with maximal debug..."
docker build -t logseq-xr-image .
docker run -d --name logseqXR \
    --env-file .env \
    -p 8443:8443 \
    --gpus all \
    -e RUST_LOG=debug \
    -e RUST_BACKTRACE=1 \
    logseq-xr-image

# Wait for the container to start
echo "Waiting for the container to start..."
sleep 5

# Check if the container is running
if [ "$(docker ps -q -f name=logseqXR)" ]; then
    echo "Container logseqXR is now running with maximal debug."
    echo "You can view the logs with: docker logs -f logseqXR"
else
    echo "Error: Container logseqXR failed to start. Please check the logs."
    exit 1
fi
