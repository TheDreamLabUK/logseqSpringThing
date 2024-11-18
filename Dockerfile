# Stage 1: Frontend Build
FROM node:23.1.0-slim AS frontend-builder

WORKDIR /app

# Copy package files, vite config, and the public directory
COPY package.json pnpm-lock.yaml vite.config.js ./
COPY data/public ./data/public

# Configure npm and build
ENV NPM_CONFIG_PREFIX=/home/node/.npm-global
ENV PATH=/home/node/.npm-global/bin:$PATH
RUN mkdir -p /home/node/.npm-global && \
    chown -R node:node /app /home/node/.npm-global && \
    npm config set prefix /home/node/.npm-global

USER node
RUN npm install -g pnpm && \
    pnpm install --frozen-lockfile && \
    pnpm run build

# Stage 2: Rust Dependencies Cache
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS rust-deps-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    libegl1-mesa-dev \
    libasound2-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.82.0
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/app

# Copy only Cargo.toml and Cargo.lock first
COPY Cargo.toml Cargo.lock ./

# Create dummy src/main.rs to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn add(a: i32, b: i32) -> i32 { a + b }" > src/lib.rs && \
    cargo build --release && \
    rm src/*.rs && \
    rm -f target/release/deps/webxr_graph* target/release/webxr-graph*

# Stage 3: Rust Application Build
FROM rust-deps-builder AS rust-builder

# Copy actual source code
COPY src ./src
COPY settings.toml ./settings.toml

# Build the application
RUN cargo build --release

# Stage 4: Python Dependencies
FROM python:3.10.12-slim AS python-builder

WORKDIR /app

# Create virtual environment and install dependencies
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip==23.3.1 wheel==0.41.3 && \
    pip install --no-cache-dir \
    piper-phonemize==1.1.0 \
    piper-tts==1.2.0 \
    onnxruntime-gpu==1.16.3

# Stage 5: Final Runtime Image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUST_LOG=info \
    RUST_BACKTRACE=0 \
    PORT=3000 \
    BIND_ADDRESS=0.0.0.0

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libssl3 \
    nginx \
    libvulkan1 \
    libegl1-mesa \
    libasound2 \
    python3.10-minimal \
    python3.10-venv \
    ca-certificates \
    mesa-vulkan-drivers \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/*

# Create nginx group and non-root user
RUN groupadd -r nginx -g 101 && \
    groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -G nginx -u 1000 -m -d /home/appuser appuser

# Set up directory structure
WORKDIR /app

# Create required directories
RUN mkdir -p /app/data/public/dist \
             /app/data/markdown \
             /app/data/runtime \
             /app/src \
             /app/data/piper && \
    chown -R appuser:appuser /app

# Copy Python virtual environment
COPY --from=python-builder --chown=appuser:appuser /app/venv /app/venv

# Copy built artifacts
COPY --from=rust-builder --chown=appuser:appuser /usr/src/app/target/release/webxr-graph /app/
COPY --from=rust-builder --chown=appuser:appuser /usr/src/app/settings.toml /app/
COPY --from=frontend-builder --chown=appuser:appuser /app/data/public/dist /app/data/public/dist

# Copy configuration and scripts
COPY --chown=appuser:appuser src/generate_audio.py /app/src/
COPY --chown=root:root nginx.conf /etc/nginx/nginx.conf

# Create and configure startup script with proper permissions
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Function to log messages with timestamps\n\
log() {\n\
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"\n\
}\n\
\n\
# Function to setup nginx directories\n\
setup_nginx_dirs() {\n\
    log "Setting up nginx directories..."\n\
    # Create required directories\n\
    mkdir -p /var/lib/nginx/client_temp \\\n\
             /var/lib/nginx/proxy_temp \\\n\
             /var/lib/nginx/fastcgi_temp \\\n\
             /var/lib/nginx/uwsgi_temp \\\n\
             /var/lib/nginx/scgi_temp\n\
\n\
    # Create and set permissions for log files\n\
    touch /var/log/nginx/error.log \\\n\
          /var/log/nginx/access.log\n\
\n\
    log "Nginx directories setup complete"\n\
}\n\
\n\
# Function to check if a port is available\n\
wait_for_port() {\n\
    local port=$1\n\
    local retries=60\n\
    local wait=5\n\
    while ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/0.0.0.0/$port" 2>/dev/null && [ $retries -gt 0 ]; do\n\
        log "Waiting for port $port to become available... ($retries retries left)"\n\
        sleep $wait\n\
        retries=$((retries-1))\n\
    done\n\
    if [ $retries -eq 0 ]; then\n\
        log "Timeout waiting for port $port"\n\
        return 1\n\
    fi\n\
    log "Port $port is available"\n\
    return 0\n\
}\n\
\n\
# Function to check RAGFlow connectivity\n\
check_ragflow() {\n\
    log "Checking RAGFlow connectivity..."\n\
    if curl -s -f --max-time 5 "http://ragflow-server/v1/" > /dev/null; then\n\
        log "RAGFlow server is reachable"\n\
        return 0\n\
    else\n\
        log "Warning: Cannot reach RAGFlow server"\n\
        return 1\n\
    fi\n\
}\n\
\n\
# Setup nginx directories\n\
setup_nginx_dirs\n\
\n\
# Wait for RAGFlow to be available\n\
log "Waiting for RAGFlow server..."\n\
retries=24\n\
while ! check_ragflow && [ $retries -gt 0 ]; do\n\
    log "Retrying RAGFlow connection... ($retries attempts left)"\n\
    retries=$((retries-1))\n\
    sleep 5\n\
done\n\
\n\
if [ $retries -eq 0 ]; then\n\
    log "Failed to connect to RAGFlow server after multiple attempts"\n\
    exit 1\n\
fi\n\
\n\
# Start nginx\n\
log "Starting nginx..."\n\
nginx -t && nginx\n\
if [ $? -ne 0 ]; then\n\
    log "Failed to start nginx"\n\
    exit 1\n\
fi\n\
log "nginx started successfully"\n\
\n\
# Start the Rust backend\n\
log "Starting webxr-graph..."\n\
exec /app/webxr-graph\n\
' > /app/start.sh && \
    chown appuser:appuser /app/start.sh && \
    chmod 755 /app/start.sh

# Add security labels
LABEL org.opencontainers.image.source="https://github.com/yourusername/logseq-xr" \
      org.opencontainers.image.description="LogseqXR WebXR Graph Visualization" \
      org.opencontainers.image.licenses="MIT" \
      security.capabilities="cap_net_bind_service" \
      security.privileged="false" \
      security.allow-privilege-escalation="false"

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4000/ || exit 1

# Start application
ENTRYPOINT ["/app/start.sh"]
