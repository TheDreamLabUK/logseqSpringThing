#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Target production container
CONTAINER_NAME="logseq-spring-thing-webxr"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Enhanced diagnostics
check_process_status() {
    echo -e "\n${YELLOW}Checking processes inside container:${NC}"
    docker exec ${CONTAINER_NAME} ps aux || echo "Could not check processes"
    
    # Use ss instead of netstat (more commonly available)
    echo -e "\n${YELLOW}Checking listening ports inside container:${NC}"
    docker exec ${CONTAINER_NAME} ss -tulpn || echo "Could not check ports"
    
    # Check webxr process specifically
    echo -e "\n${YELLOW}Checking webxr process:${NC}"
    docker exec ${CONTAINER_NAME} pgrep -a webxr || echo "No webxr process found"
    
    # Enhanced log checking
    echo -e "\n${YELLOW}Checking Rust server logs:${NC}"
    docker exec ${CONTAINER_NAME} bash -c 'for f in /app/webxr.log /app/*.log; do if [ -f "$f" ]; then echo "=== $f ==="; tail -n 20 "$f"; fi; done' 2>/dev/null || echo "No logs found"
    # Check Nginx logs
    echo -e "\n${YELLOW}Checking Nginx Access Logs:${NC}"
    docker exec ${CONTAINER_NAME} tail -n 20 /var/log/nginx/access.log 2>/dev/null || echo "No Nginx access logs found or accessible"
    echo -e "\n${YELLOW}Checking Nginx Error Logs:${NC}"
    docker exec ${CONTAINER_NAME} tail -n 20 /var/log/nginx/error.log 2>/dev/null || echo "No Nginx error logs found or accessible"
}

test_endpoints() {
    echo -e "${YELLOW}Testing basic connectivity...${NC}"
    
    # Check container status
    echo -e "\n${YELLOW}Container Status:${NC}"
    docker ps | grep ${CONTAINER_NAME}
    
    # Check container logs with timestamp
    echo -e "\n${YELLOW}Recent Container Logs:${NC}"
    docker logs --timestamps ${CONTAINER_NAME} --tail 50
    
    # Test root endpoint on host port
    echo -e "\n${YELLOW}Testing Root Endpoint (localhost:4000/):${NC}"
    curl -v http://localhost:4000/ 2>&1 || echo -e "${RED}Failed to connect to root endpoint on localhost:4000${NC}"
    
    # Test Production Endpoint (Root)
    echo -e "\n${YELLOW}Testing Production Endpoint (Root - https://www.visionflow.info/):${NC}"
    curl -v --connect-timeout 10 https://www.visionflow.info/ 2>&1 || echo -e "${RED}Failed to connect to Production Root Endpoint${NC}"

    # Production Health endpoint test removed
    # Test Internal Docker Network (cloudflared -> webxr)
    echo -e "\n${YELLOW}Testing Internal Network (cloudflared -> webxr:4000/health):${NC}"
    # Check if cloudflared container exists first
    if docker ps -q -f name=cloudflared-tunnel > /dev/null; then
        # Try using wget (often available in minimal images) inside the cloudflared container
        # Test root path instead of /health
        docker exec cloudflared-tunnel wget --spider --timeout=5 -q http://webxr:4000/ || echo -e "${RED}Failed to connect from cloudflared to webxr:4000/ using wget${NC}"
    else
        echo -e "${YELLOW}Skipping internal network test: cloudflared-tunnel container not running.${NC}"
    fi
}

check_container_health() {
    echo -e "\n${YELLOW}Container Details:${NC}"
    docker inspect ${CONTAINER_NAME} | grep -A 20 "State"
    
    # Add GPU check
    echo -e "\n${YELLOW}GPU Status:${NC}"
    docker exec ${CONTAINER_NAME} nvidia-smi || echo "Could not access GPU"
}

# Restart logic removed (this script is for testing, not management)

# Main execution
echo -e "${GREEN}Starting comprehensive diagnostics...${NC}"
test_endpoints
check_process_status
check_container_health

# Provide next steps
echo -e "\n${YELLOW}Diagnostic Summary:${NC}"
echo "1. Host Port 4000 (Root /): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:4000/ 2>/dev/null || echo "Failed")"
echo "2. Production Root (https://www.visionflow.info/): $(curl -s -o /dev/null -w "%{http_code}" https://www.visionflow.info/ 2>/dev/null || echo "Failed")"
# Production Health check removed
echo "3. Internal Network (cloudflared -> webxr /): $(docker exec cloudflared-tunnel wget --spider --timeout=5 -q http://webxr:4000/ >/dev/null 2>&1 && echo "Success" || echo "Failed/Skipped")"
echo "5. Container Status: $(docker inspect --format='{{.State.Status}}' ${CONTAINER_NAME} 2>/dev/null || echo "Not Found")"
