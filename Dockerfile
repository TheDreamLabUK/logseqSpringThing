# Stage 1: Build the Frontend
FROM node:latest AS frontend-builder

WORKDIR /app/frontend

# Copy package files and vite config before installing dependencies
COPY package.json pnpm-lock.yaml ./
COPY vite.config.js ./

# Copy the public assets
COPY data/public ./data/public  

# Install pnpm globally
RUN npm install -g pnpm

# Clean PNPM store and install dependencies
RUN pnpm store clean && pnpm install --force

# Build the frontend (this will output the 'dist' folder)
RUN pnpm run build

# Stage 2: Final Build and Serve
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install necessary runtime dependencies and nginx
RUN apt-get update && apt-get install -y \
    curl \
    libssl3 \
    nginx \
    openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built Rust binary from the backend-builder stage (you should have a backend-builder stage in your setup)
COPY --from=backend-builder /usr/src/app/target/release/webxr-graph /app/webxr-graph

# Copy the built frontend files (dist folder) into the public directory in the final container
COPY --from=frontend-builder /app/frontend/dist /app/data/public/dist

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
