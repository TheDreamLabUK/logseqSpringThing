# Stage 1: Frontend Build
FROM node:23.1.0-slim AS frontend-builder

WORKDIR /app

# Copy package files, vite config, and the public directory
COPY package.json pnpm-lock.yaml vite.config.js ./
COPY data/public ./data/public

# Install pnpm and build
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
    # Build dependencies only
    cargo build --release && \
    # Remove the dummy source files but keep the dependencies
    rm src/*.rs && \
    # Remove the target/release/deps files that depend on the dummy source
    rm -f target/release/deps/webxr_graph* target/release/webxr-graph*

# Stage 3: Rust Application Build
FROM rust-deps-builder AS rust-builder

# Copy actual source code
COPY src ./src
COPY settings.toml ./settings.toml

# Build the application, reusing cached dependencies
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

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV RUST_LOG=debug
ENV RUST_BACKTRACE=1
ENV PORT=3000
ENV BIND_ADDRESS=0.0.0.0

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

# Set up directory structure
WORKDIR /app
RUN mkdir -p /app/data/public/dist /app/data/markdown /app/src /app/data/piper && \
    mkdir -p /app/nginx/client_temp \
             /app/nginx/proxy_temp \
             /app/nginx/fastcgi_temp \
             /app/nginx/uwsgi_temp \
             /app/nginx/scgi_temp && \
    touch /app/nginx/error.log && \
    touch /app/nginx/access.log

# Copy Python virtual environment
COPY --from=python-builder /app/venv /app/venv

# Copy built artifacts
COPY --from=rust-builder /usr/src/app/target/release/webxr-graph /app/
COPY --from=rust-builder /usr/src/app/settings.toml /app/
COPY --from=frontend-builder /app/data/public/dist /app/data/public/dist

# Copy configuration and scripts
COPY src/generate_audio.py /app/src/
COPY nginx.conf /etc/nginx/nginx.conf

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to log messages with timestamps\n\
log() {\n\
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"\n\
}\n\
\n\
# Function to check if a port is available\n\
wait_for_port() {\n\
    local port=$1\n\
    local retries=200\n\
    local wait=3\n\
    while ! nc -z 0.0.0.0 $port && [ $retries -gt 0 ]; do\n\
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
# Start the Rust backend first\n\
log "Starting webxr-graph..."\n\
/app/webxr-graph > /tmp/webxr.log 2>&1 &\n\
APP_PID=$!\n\
\n\
# Wait for the backend to be ready\n\
log "Waiting for application to be ready..."\n\
if ! wait_for_port $PORT; then\n\
    log "Application failed to start. Backend logs:"\n\
    cat /tmp/webxr.log\n\
    if [ -n "$APP_PID" ] && ps -p $APP_PID > /dev/null; then\n\
        kill $APP_PID\n\
    fi\n\
    exit 1\n\
fi\n\
\n\
# Check if the backend health endpoint is responding\n\
log "Checking backend health endpoint..."\n\
if ! curl -s -f http://localhost:$PORT/health; then\n\
    log "Backend health check failed. Backend logs:"\n\
    cat /tmp/webxr.log\n\
    if [ -n "$APP_PID" ] && ps -p $APP_PID > /dev/null; then\n\
        kill $APP_PID\n\
    fi\n\
    exit 1\n\
fi\n\
log "Backend health check passed"\n\
\n\
# Start nginx after backend is ready\n\
log "Starting nginx..."\n\
nginx -t && nginx\n\
if [ $? -ne 0 ]; then\n\
    log "Failed to start nginx"\n\
    if [ -n "$APP_PID" ] && ps -p $APP_PID > /dev/null; then\n\
        kill $APP_PID\n\
    fi\n\
    exit 1\n\
fi\n\
log "nginx started successfully"\n\
\n\
# Monitor both nginx and the backend\n\
while true; do\n\
    if ! ps -p $APP_PID > /dev/null; then\n\
        log "Backend process died. Logs:"\n\
        cat /tmp/webxr.log\n\
        nginx -s stop\n\
        exit 1\n\
    fi\n\
    if ! curl -s -f http://localhost:$PORT/health > /dev/null; then\n\
        log "Backend health check failed. Logs:"\n\
        cat /tmp/webxr.log\n\
        nginx -s stop\n\
        exit 1\n\
    fi\n\
    sleep 5\n\
done' > /app/start.sh && \
    chmod +x /app/start.sh

# Health check with increased start period and interval
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start application
ENTRYPOINT ["/app/start.sh"]
