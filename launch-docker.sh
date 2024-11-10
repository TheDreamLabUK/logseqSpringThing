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

# Stop and remove existing container, including associated volumes
docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
docker rm -v $CONTAINER_NAME >/dev/null 2>&1 || true

# Create SSL directory and generate certificates if they don't exist
mkdir -p ssl

# Create OpenSSL config file with SAN
cat > ssl/openssl.cnf << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn
x509_extensions = v3_req

[dn]
C = US
ST = State
L = City
O = Organization
CN = localhost

[req_ext]
subjectAltName = @alt_names

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = 192.168.0.51
EOF

# Generate self-signed SSL certificate with SAN
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/nginx-selfsigned.key \
  -out ssl/nginx-selfsigned.crt \
  -config ssl/openssl.cnf \
  -extensions v3_req

# Build the Docker image
echo "Building Docker image..."
if ! docker build -t logseq-xr-image .; then
    echo "Docker build failed. Please check the error messages above."
    exit 1
fi

# Ensure data/markdown directory exists
mkdir -p data/markdown

# Run the Docker container with GPU 0 enabled, correct environment variables, and volume mounts
echo "Running Docker container..."
if ! docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -v "$(pwd)/data/markdown:/app/data/markdown" \
    -v "$(pwd)/settings.toml:/app/settings.toml" \
    -v "$(pwd)/ssl:/etc/nginx/ssl" \
    -p 8443:8443 \
    --env-file .env \
    logseq-xr-image; then
    echo "Failed to start Docker container. Please check the error messages above."
    exit 1
fi

echo "Docker container is now running."
echo "Access the application at https://localhost:8443"
echo "WebSocket should be available at wss://localhost:8443/ws"
echo "Note: You may see a security warning in your browser due to the self-signed certificate. This is expected for local development."

# Display container logs
echo "Container logs:"
docker logs -f $CONTAINER_NAME
