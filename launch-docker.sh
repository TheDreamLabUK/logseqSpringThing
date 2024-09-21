#!/bin/bash

# Stop and remove existing container
docker stop logseqXR || true
docker rm logseqXR || true

# Build the Docker image
docker build -t logseq-xr-image .

# Run the Docker container with GPU 0 enabled, correct environment variables, and volume mounts
docker run -d --name logseqXR \
  --gpus "device=0" \
  -v "$(pwd)/data:/app/data" \
  -p 8081:8080 \
  -e PORT=8080 \
  -e BIND_ADDRESS=0.0.0.0 \
  logseq-xr-image
