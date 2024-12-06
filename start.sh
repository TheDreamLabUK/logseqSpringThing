#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check if a port is available
wait_for_port() {
    local port=$1
    local retries=60
    local wait=5
    while ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/0.0.0.0/$port" 2>/dev/null && [ $retries -gt 0 ]; do
        log "Waiting for port $port to become available... ($retries retries left)"
        sleep $wait
        retries=$((retries-1))
    done
    if [ $retries -eq 0 ]; then
        log "Timeout waiting for port $port"
        return 1
    fi
    log "Port $port is available"
    return 0
}

# Function to check RAGFlow connectivity
check_ragflow() {
    log "Checking RAGFlow connectivity..."
    if curl -s -f --max-time 5 "http://ragflow-server/v1/" > /dev/null; then
        log "RAGFlow server is reachable"
        return 0
    else
        log "Warning: Cannot reach RAGFlow server"
        return 1
    fi
}

# Function to verify production build
verify_build() {
    log "Verifying production build..."
    if [ ! -d "/app/data/public/dist" ]; then
        log "Error: Production build directory not found"
        return 1
    fi
    
    if [ ! -f "/app/data/public/dist/index.html" ]; then
        log "Error: Production build index.html not found"
        return 1
    fi
    
    log "Production build verified"
    return 0
}

# Set up runtime environment
setup_runtime() {
    log "Setting up runtime environment..."

    # Set up XDG_RUNTIME_DIR with proper permissions
    export XDG_RUNTIME_DIR="/tmp/runtime"
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    chown "$(id -u)" "$XDG_RUNTIME_DIR"

    # Set up Vulkan ICD and layer paths
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d

    # Verify GPU is available
    if ! command -v nvidia-smi &> /dev/null; then
        log "Error: nvidia-smi not found. GPU support is required."
        return 1
    fi

    # Check GPU is accessible
    if ! nvidia-smi &> /dev/null; then
        log "Error: Cannot access NVIDIA GPU. Check device is properly passed to container."
        return 1
    fi

    # Verify Vulkan support
    if ! command -v vulkaninfo &> /dev/null; then
        log "Error: Vulkan tools not found. Vulkan support is required."
        return 1
    fi

    # Test Vulkan initialization
    if ! vulkaninfo &> /dev/null; then
        log "Error: Failed to initialize Vulkan. Check GPU drivers and permissions."
        return 1
    fi

    log "Runtime environment configured successfully"
    return 0
}

# Set up runtime environment first
if ! setup_runtime; then
    log "Failed to set up runtime environment"
    exit 1
fi

# Wait for RAGFlow to be available
log "Waiting for RAGFlow server..."
retries=24
while ! check_ragflow && [ $retries -gt 0 ]; do
    log "Retrying RAGFlow connection... ($retries attempts left)"
    retries=$((retries-1))
    sleep 5
done

if [ $retries -eq 0 ]; then
    log "Failed to connect to RAGFlow server after multiple attempts"
    exit 1
fi

# Verify production build
if ! verify_build; then
    log "Failed to verify production build"
    exit 1
fi

# Update nginx configuration with environment variables
log "Configuring nginx..."
cat > /etc/nginx/nginx.conf << 'EOF'
user root;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 20M;
    
    # Compression
    gzip on;
    gzip_disable "msie6";
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Map to detect WebSocket upgrade
    map $http_upgrade $connection_upgrade {
        default upgrade;
        ''      close;
    }

    upstream backend {
        server localhost:3000;
        keepalive 32;
    }

    server {
        listen 4000;
        server_name localhost;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN";
        add_header X-XSS-Protection "1; mode=block";
        add_header X-Content-Type-Options "nosniff";
        add_header Referrer-Policy "strict-origin-when-cross-origin";
        add_header Content-Security-Policy "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: ws: wss:; connect-src 'self' ws: wss: http: https:;";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Root directory for static files
        root /app/data/public/dist;
        
        # SPA configuration
        location / {
            try_files $uri $uri/ /index.html;
            expires -1;
            add_header Cache-Control 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0';
        }
        
        # Static file caching
        location /assets {
            expires 1y;
            add_header Cache-Control "public, no-transform";
        }
        
        # API proxy
        location /api {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
        
        # WebSocket proxy with proper upgrade handling
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
            proxy_buffering off;
            proxy_cache off;
        }
    }
}
EOF

# Start nginx
log "Starting nginx..."
nginx -t && nginx
if [ $? -ne 0 ]; then
    log "Failed to start nginx"
    exit 1
fi
log "nginx started successfully"

# Start the Rust backend
log "Starting webxr..."
exec /app/webxr
