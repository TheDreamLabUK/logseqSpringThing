#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER_NAME="logseq_spring_thing_webxr"
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
}

test_connectivity() {
    echo -e "${YELLOW}Testing basic connectivity...${NC}"
    
    # Check container status
    echo -e "\n${YELLOW}Container Status:${NC}"
    docker ps | grep ${CONTAINER_NAME}
    
    # Check container logs with timestamp
    echo -e "\n${YELLOW}Recent Container Logs:${NC}"
    docker logs --timestamps ${CONTAINER_NAME} --tail 50
    
    # Test health endpoint specifically
    echo -e "\n${YELLOW}Testing health endpoint:${NC}"
    curl -v http://localhost:4000/health 2>&1 || echo -e "${RED}Failed to connect to health endpoint${NC}"
    
    # Test Vite dev server root
    echo -e "\n${YELLOW}Testing Vite dev server:${NC}"
    curl -v http://localhost:3001/ 2>&1 || echo -e "${RED}Failed to connect to Vite server${NC}"
}

check_container_health() {
    echo -e "\n${YELLOW}Container Details:${NC}"
    docker inspect ${CONTAINER_NAME} | grep -A 20 "State"
    
    # Add GPU check
    echo -e "\n${YELLOW}GPU Status:${NC}"
    docker exec ${CONTAINER_NAME} nvidia-smi || echo "Could not access GPU"
}

# Restart container if needed
restart_if_needed() {
    if ! curl -s http://localhost:4000/health >/dev/null; then
        echo -e "${YELLOW}Services not responding, attempting restart...${NC}"
        docker restart ${CONTAINER_NAME}
        sleep 10  # Wait for services to initialize
    fi
}

# Main execution
echo -e "${GREEN}Starting comprehensive diagnostics...${NC}"
restart_if_needed
test_connectivity
check_process_status
check_container_health

# Provide next steps
echo -e "\n${YELLOW}Diagnostic Summary:${NC}"
echo "1. API Server (port 4000): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:4000/health 2>/dev/null || echo "Failed")"
echo "2. Vite Server (port 3001): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/ 2>/dev/null || echo "Failed")"
echo "3. Container Status: $(docker inspect --format='{{.State.Status}}' ${CONTAINER_NAME})"
