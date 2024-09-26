#!/bin/bash

# Stop and remove existing container, including associated volumes
docker stop logseqXR || true
docker rm -v logseqXR || true

# Remove any dangling volumes (optional)
# docker volume prune -f

# Build the Docker image
docker build --no-cache -t logseq-xr-image .

# Run the Docker container with GPU 0 enabled, correct environment variables, and volume mounts
docker run -d --name logseqXR \
  --gpus "device=0" \
  -v "$(pwd)/data:/app/data" \
  -p 8443:8443 \
  --env-file .env \
  logseq-xr-image

echo "Docker container is now running."
echo "Access the application at https://192.168.0.51:8443"
echo "WebSocket should be available at https://://192.168.0.51:8443/ws"
echo "Note: You may see a security warning in your browser due to the self-signed certificate. This is expected for local development."

# Display container logs
echo "Container logs:"
docker logs -f logseqXR
