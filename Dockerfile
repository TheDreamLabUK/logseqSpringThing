# Stage 1: Frontend Build
FROM node:20-slim AS frontend-builder

WORKDIR /app

# Copy package files and configuration
COPY package.json package-lock.json ./
COPY tsconfig.json tsconfig.node.json vite.config.ts ./
COPY client ./client

# Create data/public directory for build output
RUN mkdir -p data/public

# Install dependencies and build
RUN npm ci && \
    npm run build

# Stage 2: Rust Dependencies Cache
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS rust-deps-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    libegl1-mesa-dev \
    libasound2-dev \
    ca-certificates \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Rust with better error handling
RUN curl --retry 5 --retry-delay 2 --retry-connrefused https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.82.0
ENV PATH="/root/.cargo/bin:${PATH}"

# Configure cargo for better network resilience
RUN mkdir -p ~/.cargo && \
    echo '[source.crates-io]' >> ~/.cargo/config.toml && \
    echo 'registry = "https://github.com/rust-lang/crates.io-index"' >> ~/.cargo/config.toml && \
    echo 'replace-with = "ustc"' >> ~/.cargo/config.toml && \
    echo '[source.ustc]' >> ~/.cargo/config.toml && \
    echo 'registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"' >> ~/.cargo/config.toml && \
    echo '[net]' >> ~/.cargo/config.toml && \
    echo 'retry = 10' >> ~/.cargo/config.toml && \
    echo 'timeout = 120' >> ~/.cargo/config.toml && \
    echo 'git-fetch-with-cli = true' >> ~/.cargo/config.toml

WORKDIR /usr/src/app

# Copy Cargo files first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Install git and set GIT_HASH
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create dummy src directory and build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    GIT_HASH=$(git rev-parse HEAD || echo "development") \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    CARGO_HTTP_TIMEOUT=120 \
    CARGO_HTTP_CHECK_REVOKE=false \
    cargo build --release --features gpu --jobs $(nproc) || \
    (sleep 2 && GIT_HASH=$(git rev-parse HEAD || echo "development") CARGO_HTTP_MULTIPLEXING=false cargo build --release --jobs $(nproc)) || \
    (sleep 5 && GIT_HASH=$(git rev-parse HEAD || echo "development") CARGO_HTTP_MULTIPLEXING=false cargo build --release --jobs 1)

# Copy the real source code and build
COPY src ./src

RUN GIT_HASH=$(git rev-parse HEAD || echo "development") \
    cargo build --release --features gpu --jobs $(nproc) || \
    (sleep 2 && GIT_HASH=$(git rev-parse HEAD || echo "development") cargo build --release --jobs $(nproc)) || \
    (sleep 5 && GIT_HASH=$(git rev-parse HEAD || echo "development") cargo build --release --jobs 1)

# Stage 3: Final Runtime Image
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

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
    jq \
    wget \
    && wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq \
    && chmod +x /usr/bin/yq \
    && wget https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl -O /usr/bin/websocat \
    && chmod +x /usr/bin/websocat \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/*

# Create non-root user
RUN groupadd -r webxr && useradd -r -g webxr webxr

# Create necessary directories
RUN mkdir -p /app/data/public/dist && \
    mkdir -p /app/src/utils && \
    chown -R webxr:webxr /app

# Switch to non-root user
USER webxr

# Copy built artifacts
COPY --from=rust-deps-builder /usr/src/app/target/release/webxr /app/
COPY src/utils/compute_forces.ptx /app/src/utils/compute_forces.ptx
COPY --from=frontend-builder /app/data/public/dist /app/data/public/dist

# Copy start script
COPY scripts/start.sh /app/start.sh

# Set proper permissions
USER root
RUN chown -R webxr:webxr /app && \
    chmod 755 /app/start.sh && \
    chmod -R g+w /app && \
    chmod 644 /app/src/utils/compute_forces.ptx
# Settings file is mounted via docker-compose, no need to touch/chmod here

USER webxr

EXPOSE 4000

CMD ["/app/start.sh"]
