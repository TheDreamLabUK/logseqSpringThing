#!/bin/bash

# Trap signals early to ensure proper cleanup
trap "echo 'Stopping services...'; kill $RUST_PID; exit" INT TERM

# Function to stop existing services
stop_existing_services() {
    echo "Checking for existing services..."

    # Stop existing Rust server
    if pgrep -f "cargo run" > /dev/null; then
        echo "Stopping existing Rust server..."
        pkill -f "cargo run"
    fi

    # Wait a moment to ensure processes have stopped
    sleep 2
}

# Function to generate SSL certificates
generate_ssl_certificates() {
    if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
        echo "Generating SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
        chmod 664 key.pem
        chmod 644 cert.pem
    else
        echo "SSL certificates already exist."
    fi
}

# Stop existing services
stop_existing_services

# Generate SSL certificates
generate_ssl_certificates

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
fi

# Set the port for the Rust server
export PORT=8443

# Start Rust server
echo "Starting Rust server on port $PORT..."
RUST_BACKTRACE=1 cargo run --release &
RUST_PID=$!

echo "Rust server is now running on https://localhost:$PORT"
echo "This server handles both HTTPS and WebSocket connections."
echo "Press Ctrl+C to stop the service."

# Wait for the Rust server to finish
wait $RUST_PID
