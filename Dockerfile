# Stage 1: Build the Frontend
FROM node:latest AS frontend-builder

# Set working directory
WORKDIR /app/frontend

# Copy frontend package files
COPY data/public/package.json data/public/pnpm-lock.yaml ./

# Install pnpm globally
RUN npm install -g pnpm

# Install frontend dependencies
RUN pnpm install

# Copy frontend source files
COPY data/public ./data/public

# Build the frontend
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
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the default Rust toolchain to stable
RUN rustup default stable

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the Cargo.toml and Cargo.lock files to leverage Docker cache
COPY Cargo.toml Cargo.lock ./

# Copy the source code
COPY src ./src

# Copy settings.toml
COPY settings.toml ./

# Build the Rust application in release mode for optimized performance
RUN cargo build --release

# Stage 3: Create the Final Image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install necessary runtime dependencies and nginx
RUN apt-get update && apt-get install -y \
    curl \
    libssl3 \
    nginx \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the built Rust binary from the backend-builder stage
COPY --from=backend-builder /usr/src/app/target/release/webxr-graph /app/webxr-graph

# Copy the built frontend files from the frontend-builder stage
COPY --from=frontend-builder /app/frontend/dist /app/data/public/dist

# Copy settings.toml from the backend-builder stage
COPY --from=backend-builder /usr/src/app/settings.toml /app/settings.toml

# Set up a persistent volume for Markdown files to ensure data persistence
VOLUME ["/app/data/markdown"]

# Create directory for SSL certificates
RUN mkdir -p /etc/nginx/ssl

# Generate self-signed SSL certificate in the correct location
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/selfsigned.key \
    -out /etc/nginx/ssl/selfsigned.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=192.168.0.51"

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Ensure proper permissions for nginx
RUN chown -R www-data:www-data /var/lib/nginx

# Expose HTTPS port
EXPOSE 8443

# Create a startup script that runs nginx and the Rust application
RUN echo '#!/bin/bash\nset -e\nnginx\nexec /app/webxr-graph' > /app/start.sh && chmod +x /app/start.sh

# Set the command to run the startup script
CMD ["/app/start.sh"]