#!/bin/bash

set -e

# Define the container name
CONTAINER_NAME="logseqXR"
echo $CONTAINER_NAME

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create a .env file with the necessary environment variables."
    echo "You can use .env_template as a reference."
    exit 1
fi

# Check for domain configuration
if [ -z "$DOMAIN" ]; then
    echo "Warning: DOMAIN environment variable not set in .env file."
    echo "The application will run with self-signed certificates."
    echo "To use Let's Encrypt SSL certificates, add these to your .env file:"
    echo "DOMAIN=your-domain.com"
    echo "CERTBOT_EMAIL=your-email@example.com"
    echo ""
fi

# Stop and remove existing container, including associated volumes
docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
docker rm -v $CONTAINER_NAME >/dev/null 2>&1 || true

# Build the Docker image
echo "Building Docker image..."
if ! docker build -t logseq-xr-image .; then
    echo "Docker build failed. Please check the error messages above."
    exit 1
fi

# Ensure data/markdown directory exists
mkdir -p data/markdown

# Run the Docker container with GPU support
echo "Running Docker container..."
if ! docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -v "$(pwd)/data/markdown:/app/data/markdown" \
    -v "$(pwd)/settings.toml:/app/settings.toml" \
    -p 80:80 \
    -p 8443:8443 \
    --env-file .env \
    logseq-xr-image; then
    echo "Failed to start Docker container. Please check the error messages above."
    exit 1
fi

echo "Docker container is now running."

if [ ! -z "$DOMAIN" ]; then
    echo "The application will be available at:"
    echo "https://$DOMAIN:8443"
    echo ""
    echo "Let's Encrypt certificate will be automatically requested for $DOMAIN"
    echo "Please ensure your domain's DNS is properly configured and ports 80 and 8443 are accessible."
else
    echo "Access the application at https://localhost:8443"
    echo "Note: You will see a security warning due to the self-signed certificate."
fi

echo ""
echo "To view the certificate setup progress, use:"
echo "docker logs -f $CONTAINER_NAME"

# Display container logs
echo "Container logs:"
docker logs -f $CONTAINER_NAME
