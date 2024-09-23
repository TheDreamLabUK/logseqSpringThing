#!/bin/bash

# Stop and remove existing container
docker stop logseqXR || true
docker rm logseqXR || true

# Build the Docker image
docker build --no-cache -t logseq-xr-image .

# Run the Docker container with GPU 0 enabled, correct environment variables, and volume mounts
docker run -d --name logseqXR \
  --gpus "device=0" \
  -v "$(pwd)/data:/app/data" \
  -p 8443:8443 \
  -e PORT=8080 \
  -e BIND_ADDRESS=0.0.0.0 \
  -e RUST_LOG=debug \
  -e USE_HTTPS=true \
  logseq-xr-image

echo "Docker container is now running."
echo "Access the application at https://192.168.0.51:8443"
echo "WebSocket should be available at wss://192.168.0.51:8443/"
echo "Note: You may see a security warning in your browser due to the self-signed certificate. This is expected for local development."

# Display container logs
echo "Container logs:"
docker logs -f logseqXR
