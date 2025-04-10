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
    
    echo -e "\n${YELLOW}Checking listening ports inside container:${NC}"
    docker exec ${CONTAINER_NAME} netstat -tulpn || echo "Could not check ports"
    
    echo -e "\n${YELLOW}Checking Rust server logs:${NC}"
    docker exec ${CONTAINER_NAME} tail -n 20 /app/webxr.log 2>/dev/null || echo "No Rust server log found"
}

test_connectivity() {
    echo -e "${YELLOW}Testing basic connectivity...${NC}"
    
    # Check container status
    echo -e "\n${YELLOW}Container Status:${NC}"
    docker ps | grep ${CONTAINER_NAME}
    
    # Check container logs
    echo -e "\n${YELLOW}Recent Container Logs:${NC}"
    docker logs ${CONTAINER_NAME} --tail 20
    
    # Test endpoints using curl with verbose output
    echo -e "\n${YELLOW}Testing endpoints with verbose output:${NC}"
    
    echo -e "\nTesting port 4000 (API server):"
    curl -v http://localhost:4000/ 2>&1 || echo -e "${RED}Failed to connect to port 4000${NC}"
    
    echo -e "\nTesting port 3001 (Vite dev server):"
    curl -v http://localhost:3001/ 2>&1 || echo -e "${RED}Failed to connect to port 3001${NC}"
}

check_container_health() {
    echo -e "\n${YELLOW}Container Details:${NC}"
    docker inspect ${CONTAINER_NAME} | grep -A 20 "State"
}

# Main execution
echo -e "${GREEN}Starting comprehensive diagnostics...${NC}"
test_connectivity
check_process_status
check_container_health

# Provide next steps
echo -e "\n${YELLOW}Diagnostic Summary:${NC}"
echo "1. API Server (port 4000): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:4000/ 2>/dev/null || echo "Failed")"
echo "2. Vite Server (port 3001): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/ 2>/dev/null || echo "Failed")"
echo "3. Container Status: $(docker inspect --format='{{.State.Status}}' ${CONTAINER_NAME})"
