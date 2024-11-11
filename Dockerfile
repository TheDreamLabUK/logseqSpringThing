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

# Stage 2: Rust Backend Build
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS backend-builder

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

# Copy the Cargo files and source code
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY settings.toml ./settings.toml

# Build the Rust application
RUN cargo build --release

# Stage 3: Python Dependencies
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

# Stage 4: Final Runtime Image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Create non-root user with same UID as typical host user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/false -m appuser

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/*

# Set up directory structure and permissions
WORKDIR /app
RUN mkdir -p /app/data/public/dist /app/data/markdown /app/src /app/data/piper && \
    # Create nginx temp directories
    mkdir -p /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp && \
    # Set permissions for app directories
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    # Set permissions for nginx temp directories
    chown -R appuser:appuser /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp && \
    chmod -R 755 /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp

# Copy Python virtual environment
COPY --from=python-builder --chown=appuser:appuser /app/venv /app/venv

# Copy built artifacts
COPY --from=backend-builder --chown=appuser:appuser /usr/src/app/target/release/webxr-graph /app/
COPY --from=backend-builder --chown=appuser:appuser /usr/src/app/settings.toml /app/
COPY --from=frontend-builder --chown=appuser:appuser /app/data/dist /app/data/public/dist

# Copy configuration and scripts
COPY --chown=appuser:appuser src/generate_audio.py /app/src/
COPY --chown=appuser:appuser nginx.conf /etc/nginx/nginx.conf

# Create startup script with proper permissions
RUN echo '#!/bin/bash\nset -e\n\n# Start nginx and the application\nnginx\nexec /app/webxr-graph' > /app/start.sh && \
    chmod +x /app/start.sh && \
    chown appuser:appuser /app/start.sh

# Switch to non-root user
USER appuser

# Health check with increased start period and interval
HEALTHCHECK --interval=30s --timeout=10s --start-period=200s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start application
ENTRYPOINT ["/app/start.sh"]
