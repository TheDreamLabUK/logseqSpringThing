# Stage 1: Frontend Build
FROM node:20-slim AS frontend-builder

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.14.2

# Copy package files and configuration
COPY package.json pnpm-lock.yaml ./
COPY tsconfig.json tsconfig.node.json vite.config.ts ./
COPY index.html ./
COPY client ./client

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

# Copy only Cargo.toml and Cargo.lock first
COPY Cargo.toml Cargo.lock ./

# Create dummy src/main.rs to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn add(a: i32, b: i32) -> i32 { a + b }" > src/lib.rs && \
    cargo build --release && \
    rm src/*.rs && \
    rm -f target/release/deps/webxr* target/release/webxr*

# Stage 3: Rust Application Build
FROM rust-deps-builder AS rust-builder

# Copy actual source code and PTX file
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
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=2 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUST_LOG=info \
    RUST_BACKTRACE=0 \
    PORT=4000 \
    BIND_ADDRESS=0.0.0.0 \
    NODE_ENV=production

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
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/*

# Set up nginx directories and permissions
RUN mkdir -p /var/lib/nginx/client_temp \
             /var/lib/nginx/proxy_temp \
             /var/lib/nginx/fastcgi_temp \
             /var/lib/nginx/uwsgi_temp \
             /var/lib/nginx/scgi_temp \
             /var/log/nginx \
             /var/run/nginx \
             /var/cache/nginx && \
    chmod -R 777 /var/lib/nginx \
                 /var/log/nginx \
                 /var/run/nginx \
                 /var/cache/nginx \
                 /etc/nginx && \
    touch /var/log/nginx/error.log \
          /var/log/nginx/access.log && \
    chmod 666 /var/log/nginx/*.log && \
    touch /var/run/nginx/nginx.pid && \
    chmod 666 /var/run/nginx/nginx.pid

# Set up directory structure
WORKDIR /app

# Create required directories with root permissions
RUN mkdir -p /app/data/public/dist \
             /app/data/markdown \
             /app/data/runtime \
             /app/src/utils \
             /app/data/piper \
             /tmp/runtime && \
    chmod -R 777 /app /tmp/runtime

# Copy Python virtual environment
COPY --from=python-builder /app/venv /app/venv

# Copy built artifacts
COPY --from=rust-builder /usr/src/app/target/release/webxr /app/
COPY --from=rust-builder /usr/src/app/settings.toml /app/
COPY --from=rust-builder /usr/src/app/src/utils/compute_forces.ptx /app/compute_forces.ptx
COPY --from=frontend-builder /app/dist /app/data/public/dist

# Copy configuration and scripts
COPY src/generate_audio.py /app/src/
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /app/start.sh
RUN chmod 755 /app/start.sh && \
    chmod 644 /etc/nginx/nginx.conf

# Add security labels
LABEL org.opencontainers.image.source="https://github.com/yourusername/logseq-xr" \
      org.opencontainers.image.description="LogseqXR WebXR Graph Visualization" \
      org.opencontainers.image.licenses="MIT" \
      security.capabilities="cap_net_bind_service" \
      security.privileged="false" \
      security.allow-privilege-escalation="false"

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4000/ || exit 1

# Start application
ENTRYPOINT ["/app/start.sh"]
