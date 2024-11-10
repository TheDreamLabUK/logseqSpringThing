# Stage 1: Build the Frontend
FROM node:latest AS frontend-builder

WORKDIR /app

# Copy package files, vite config, and the public directory
COPY package.json pnpm-lock.yaml vite.config.js ./
COPY data/public ./data/public

# Install pnpm globally
RUN npm install -g pnpm

# Clean PNPM store and install dependencies
RUN pnpm install

# Build the frontend (this will output to /app/data/dist)
RUN pnpm run build

# Stage 2: Build the Rust Backend
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS backend-builder

# Install necessary dependencies for building Rust applications
RUN apt-get update && apt-get install -y \
    build-essential \
    gnupg2 \
    curl \
    libssl-dev \
    pkg-config \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    libegl1-mesa-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the default toolchain to stable
RUN rustup default stable

WORKDIR /usr/src/app

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Copy the source code and settings
COPY src ./src
COPY settings.toml ./settings.toml

# Build the Rust application in release mode for optimized performance
RUN cargo build --release

# Stage 3: Create the Final Image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary runtime dependencies, nginx, certbot, and Python 3.10
RUN apt-get update && apt-get install -y \
    curl \
    libssl3 \
    nginx \
    openssl \
    libvulkan1 \
    libegl1-mesa \
    libasound2 \
    software-properties-common \
    certbot \
    python3-certbot-nginx \
    cron \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data/public/dist /app/data/markdown /app/src /app/data/piper /etc/letsencrypt

# Create an empty metadata.json file
RUN mkdir -p /app/data/markdown && touch /app/data/markdown/metadata.json && echo "{}" > /app/data/markdown/metadata.json

# Copy the local piper voice model
COPY data/piper/en_GB-northern_english_male-medium.onnx /app/data/piper/en_GB-northern_english_male-medium.onnx

# Copy the built Rust binary from the backend-builder stage
COPY --from=backend-builder /usr/src/app/target/release/webxr-graph /app/webxr-graph

# Copy settings.toml from the builder stage
COPY --from=backend-builder /usr/src/app/settings.toml /app/settings.toml

# Copy the built frontend files from the frontend-builder stage
COPY --from=frontend-builder /app/data/dist /app/data/public/dist

# Copy the generate_audio.py script
COPY src/generate_audio.py /app/src/generate_audio.py

# Set up a persistent volume for Markdown files to ensure data persistence
VOLUME ["/app/data/markdown"]

# Create directory for Let's Encrypt certificates
VOLUME ["/etc/letsencrypt"]

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Ensure proper permissions for nginx and application directories
RUN chown -R www-data:www-data /var/lib/nginx /app

# Create Python virtual environment and install Piper TTS
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip, install wheel, and then install Piper TTS and its dependencies
RUN pip install --upgrade pip wheel && \
    pip install --upgrade piper-phonemize==1.1.0 && \
    pip install --upgrade piper-tts==1.2.0 onnxruntime-gpu

# Set up automatic certificate renewal
RUN echo "0 12 * * * root /usr/bin/certbot renew --quiet" > /etc/cron.d/certbot-renew
RUN chmod 0644 /etc/cron.d/certbot-renew

# Expose HTTP and HTTPS ports
EXPOSE 80 8443

# Create startup script for certificate management and service startup
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to request Let'\''s Encrypt certificate\n\
setup_ssl() {\n\
    if [ -z "$DOMAIN" ]; then\n\
        echo "Error: DOMAIN environment variable not set"\n\
        exit 1\n\
    fi\n\
    \n\
    # Check if certificate already exists\n\
    if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then\n\
        echo "Requesting Let'\''s Encrypt certificate for $DOMAIN..."\n\
        certbot --nginx \\\n\
            --non-interactive \\\n\
            --agree-tos \\\n\
            --email ${CERTBOT_EMAIL:-"admin@example.com"} \\\n\
            -d $DOMAIN \\\n\
            --redirect\n\
    else\n\
        echo "Certificate for $DOMAIN already exists"\n\
    fi\n\
}\n\
\n\
# Ensure metadata.json exists\n\
if [ ! -f /app/data/markdown/metadata.json ]; then\n\
    echo "{}" > /app/data/markdown/metadata.json\n\
fi\n\
\n\
# Start nginx\n\
nginx\n\
\n\
# Start cron for certificate renewal\n\
service cron start\n\
\n\
# Set up SSL if domain is provided\n\
if [ ! -z "$DOMAIN" ]; then\n\
    setup_ssl\n\
fi\n\
\n\
# Start the Rust application\n\
exec /app/webxr-graph' > /app/start.sh && chmod +x /app/start.sh

# Set the command to run the startup script
CMD ["/app/start.sh"]
