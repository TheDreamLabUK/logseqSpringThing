# Docker Deployment Guide

This guide details the deployment process for LogseqXR using Docker and Docker Compose.

## Prerequisites

- Docker Engine 24.0+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (for GPU support)
- Cloudflare account (for tunnel setup)

## Directory Structure

```
logseq-xr/
├── .env                    # Environment variables
├── docker-compose.yml      # Service definitions
├── Dockerfile             # Multi-stage build definition
├── nginx.conf            # Nginx configuration
├── config.yml            # Cloudflare tunnel config
├── settings.yaml         # Application settings
└── data/                 # Mounted data volumes
    ├── markdown/         # Markdown files
    ├── metadata/         # Graph metadata
    ├── piper/           # TTS models
    └── public/          # Built frontend files
```

## Configuration Files

### 1. Environment Variables (.env)
```env
# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repo_name
GITHUB_BASE_PATH=path/to/markdown/files

# Cloudflare Configuration
TUNNEL_TOKEN=your_cloudflare_tunnel_token

# Application Settings
RUST_LOG=info
RUST_BACKTRACE=1
NODE_ENV=production
DOMAIN=your.domain.com

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=0
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# User Configuration
UID=1000
GID=1000
```

### 2. Docker Compose Configuration
```yaml
name: logseq-xr

services:
  webxr:
    build: .
    image: logseq-xr-image:latest
    container_name: logseq-xr-webxr
    read_only: false
    networks:
      ragflow:
        aliases:
          - logseq-xr-webxr
          - webxr-client
    deploy:
      resources:
        limits:
          cpus: '16.0'
          memory: 64G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    expose:
      - "4000"
    ports:
      - "4000:4000"
    environment:
      - RUST_LOG=info
      - RUST_BACKTRACE=1
      - BIND_ADDRESS=0.0.0.0
      - PORT=3001
      - NGINX_PORT=4000
      - NODE_ENV=production
    volumes:
      - ./data/markdown:/app/data/markdown
      - ./data/metadata:/app/data/metadata
      - ./data/piper:/app/data/piper
      - ./client:/app/client
      - type: bind
        source: ${PWD}/settings.yaml
        target: /app/settings.yaml
        read_only: false
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 4G
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:4000/ || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  cloudflared:
    image: cloudflare/cloudflared:latest
    container_name: cloudflared-tunnel
    networks:
      ragflow:
        aliases:
          - cloudflared
    volumes:
      - ./config.yml:/etc/cloudflared/config.yml:ro
    command: tunnel --config /etc/cloudflared/config.yml run
    environment:
      - TUNNEL_TOKEN=${TUNNEL_TOKEN}
```

### 3. Nginx Configuration
```nginx
pid /var/run/nginx/nginx.pid;
error_log /var/log/nginx/error.log debug;

events {
    worker_connections 1024;
    multi_accept on;
    use epoll;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    charset utf-8;

    # WebSocket configuration
    map $http_upgrade $connection_upgrade {
        default upgrade;
        ''      close;
    }

    upstream backend {
        server 127.0.0.1:3001;
        keepalive 32;
    }

    server {
        listen 4000 default_server;
        root /app/data/public/dist;

        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options SAMEORIGIN;
        add_header X-XSS-Protection "1; mode=block";
        add_header Content-Security-Policy "default-src 'self'";

        # WebSocket endpoint
        location /wss {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_read_timeout 3600s;
        }

        # API endpoints
        location /api {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_buffering on;
            proxy_buffer_size 256k;
        }

        # Static files
        location / {
            try_files $uri $uri/ /index.html =404;
            expires 1h;
        }
    }
}
```

## Deployment Steps

### 1. Initial Setup
```bash
# Clone repository
git clone https://github.com/yourusername/logseq-xr.git
cd logseq-xr

# Copy and configure environment
cp .env_template .env
nano .env

# Create data directories
mkdir -p data/{markdown,metadata,piper,public}
```

### 2. Build and Deploy
```bash
# Build and start services
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Check service health
docker ps
docker-compose ps
```

### 3. Verify Deployment
```bash
# Check HTTP endpoint
curl http://localhost:4000/health

# Test WebSocket connection
websocat ws://localhost:4000/wss
```

## Monitoring & Maintenance

### Health Checks
The Docker Compose configuration includes health checks that monitor:
- HTTP endpoint availability
- WebSocket connectivity
- Resource usage

### Resource Monitoring
```bash
# Monitor container resources
docker stats logseq-xr-webxr

# Check logs
docker-compose logs -f

# Check nginx logs
docker exec logseq-xr-webxr tail -f /var/log/nginx/error.log
```

### Backup & Recovery
```bash
# Backup data volumes
docker run --rm \
  -v logseq-xr_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/data-backup.tar.gz /data

# Restore from backup
docker run --rm \
  -v logseq-xr_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/data-backup.tar.gz -C /
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check GPU visibility
   docker exec logseq-xr-webxr nvidia-smi
   ```

2. **WebSocket Connection Failed**
   - Check nginx logs
   - Verify proxy settings
   - Ensure Cloudflare tunnel is running

3. **Performance Issues**
   - Monitor resource usage
   - Check for memory leaks
   - Verify GPU utilization

### Recovery Procedures

1. **Service Recovery**
   ```bash
   # Restart services
   docker-compose restart

   # Rebuild and restart
   docker-compose up --build -d
   ```

2. **Data Recovery**
   ```bash
   # Stop services
   docker-compose down

   # Restore data
   cp -r backup/data/* data/

   # Restart services
   docker-compose up -d
   ```

## Security Considerations

1. **Container Security**
   - Run containers as non-root
   - Use read-only root filesystem
   - Implement resource limits

2. **Network Security**
   - Use Cloudflare tunnel for secure access
   - Implement proper firewall rules
   - Enable security headers

3. **Data Security**
   - Regular backups
   - Encrypted storage
   - Proper permission management

## Related Documentation
- [Development Setup](../development/setup.md)
- [Configuration Guide](../development/configuration.md)
- [Performance Tuning](../technical/performance.md)