# Stage 1: Frontend Build
FROM node:20-slim AS frontend-builder

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.14.2

# Copy package files and configuration
COPY package.json pnpm-lock.yaml ./
COPY tsconfig.json tsconfig.node.json vite.config.ts ./
COPY client ./client

# Create data/public directory for build output
RUN mkdir -p data/public

# Install dependencies and build
RUN pnpm install --frozen-lockfile && \
    pnpm run build

# Stage 2: Rust Dependencies Cache
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS rust-deps-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    libegl1-mesa-dev \
    libasound2-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.82.0
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/app

# Copy Cargo files and entire src directory for proper module resolution
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build dependencies and application
RUN cargo build --release

# Stage 3: Python Dependencies
FROM python:3.10.12-slim AS python-builder

WORKDIR /app

# Create virtual environment and install dependencies
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python packages
RUN pip install --upgrade pip==23.3.1 wheel==0.41.3 && \
    pip install \
    piper-phonemize==1.1.0 \
    piper-tts==1.2.0 \
    onnxruntime-gpu==1.16.3

# Stage 4: Final Runtime Image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:${PATH}" \
    NVIDIA_DRIVER_CAPABILITIES=all \
    RUST_LOG=info \
    RUST_BACKTRACE=0 \
    PORT=4000 \
    BIND_ADDRESS=0.0.0.0 \
    NODE_ENV=production \
    DOMAIN=localhost

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libssl3 \
    nginx \
    libegl1-mesa \
    libasound2 \
    python3.10-minimal \
    python3.10-venv \
    ca-certificates \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    netcat-openbsd \
    gettext-base \
    net-tools \
    iproute2 \
    procps \
    lsof \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/*

# Create a non-root user for running the application
RUN groupadd -g 1000 webxr && \
    useradd -u 1000 -g webxr -d /app webxr

# Set up nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf.template
RUN envsubst '${DOMAIN}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf && \
    rm /etc/nginx/nginx.conf.template && \
    chown -R webxr:webxr /etc/nginx/nginx.conf && \
    chmod 644 /etc/nginx/nginx.conf

# Set up nginx directories and permissions
RUN mkdir -p /var/lib/nginx/client_temp \
             /var/lib/nginx/proxy_temp \
             /var/lib/nginx/fastcgi_temp \
             /var/lib/nginx/uwsgi_temp \
             /var/lib/nginx/scgi_temp \
             /var/log/nginx \
             /var/run/nginx \
             /var/cache/nginx && \
    chown -R webxr:webxr /var/lib/nginx \
                         /var/log/nginx \
                         /var/run/nginx \
                         /var/cache/nginx \
                         /etc/nginx && \
    chmod -R 755 /var/lib/nginx \
                 /var/log/nginx \
                 /var/run/nginx \
                 /var/cache/nginx \
                 /etc/nginx && \
    touch /var/log/nginx/error.log \
          /var/log/nginx/access.log \
          /var/run/nginx/nginx.pid && \
    chmod 666 /var/log/nginx/*.log \
              /var/run/nginx/nginx.pid

# Set up directory structure and permissions
WORKDIR /app

# Create required directories with proper permissions
RUN mkdir -p /app/data/public/dist \
             /app/data/markdown \
             /app/data/runtime \
             /app/src/utils \
             /app/data/piper \
             /tmp/runtime && \
    chown -R webxr:webxr /app /tmp/runtime && \
    chmod -R 755 /app /tmp/runtime

# Copy Python virtual environment
COPY --from=python-builder /app/venv /app/venv
RUN chown -R webxr:webxr /app/venv

# Copy built artifacts
COPY --from=rust-deps-builder /usr/src/app/target/release/webxr /app/
COPY settings.toml /app/
COPY src/utils/compute_forces.ptx /app/compute_forces.ptx
COPY --from=frontend-builder /app/data/public/dist /app/data/public/dist

# Copy configuration and scripts
COPY src/generate_audio.py /app/src/
COPY scripts/start.sh /app/start.sh

# Set proper permissions for copied files
RUN chown -R webxr:webxr /app && \
    chmod 755 /app/start.sh && \
    chmod 644 /app/settings.toml && \
    chmod -R g+w /app

# Switch to non-root user
USER webxr

# Add security labels
LABEL org.opencontainers.image.source="https://github.com/yourusername/logseq-xr" \
      org.opencontainers.image.description="LogseqXR WebXR Graph Visualization" \
      org.opencontainers.image.licenses="MIT" \
      security.capabilities="cap_net_bind_service" \
      security.privileged="false" \
      security.allow-privilege-escalation="false"

# Expose port
EXPOSE 4000

# Start application
ENTRYPOINT ["/app/start.sh"]
