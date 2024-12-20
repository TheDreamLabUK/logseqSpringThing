#!/bin/bash

# Enable error reporting
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CONTAINER_NAME="logseq-xr-webxr"
BACKEND_PORT=3001
NGINX_PORT=4000
PUBLIC_DOMAIN="www.visionflow.info"
RAGFLOW_NETWORK="docker_ragflow"
TIMEOUT=5

# Function to log messages
log() {
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

# Function to safely execute docker commands with timeout
docker_exec() {
    timeout $TIMEOUT docker exec "$CONTAINER_NAME" $@ 2>&1 || echo "Command timed out after ${TIMEOUT}s"
}

# Function to check if port is open
check_port() {
    local host="$1"
    local port="$2"
    timeout $TIMEOUT bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null
    return $?
}

# Function to test endpoint and show response
test_endpoint() {
    local url="$1"
    local description="$2"
    local extra_opts="${3:-}"
    local expected_content="${4:-}"
    
    log "${BLUE}Testing $description...${NC}"
    log "URL: $url"
    
    # First check if port is open
    local port=$(echo "$url" | sed -n 's/.*:\([0-9]\+\).*/\1/p')
    local host=$(echo "$url" | sed -n 's/.*\/\/\([^:\/]*\).*/\1/p')
    
    if [ -n "$port" ] && ! check_port "$host" "$port"; then
        log "${RED}✗ Port $port is not open on $host${NC}"
        return 1
    fi
    
    # Then try the request
    local response
    if [[ -n "$extra_opts" ]]; then
        response=$(curl -v -m $TIMEOUT -s $extra_opts "$url" 2>&1)
    else
        response=$(curl -v -m $TIMEOUT -s "$url" 2>&1)
    fi
    local status=$?
    
    if [ $status -eq 0 ]; then
        log "${GREEN}✓ $description successful${NC}"
        log "Response: $response"
        
        # Check for expected content if provided
        if [ -n "$expected_content" ] && ! echo "$response" | grep -q "$expected_content"; then
            log "${RED}✗ Expected content not found: $expected_content${NC}"
            return 1
        fi
        
        return 0
    else
        log "${RED}✗ $description failed (status: $status)${NC}"
        log "Response: $response"
        return 1
    fi
}

# Function to check Nginx logs
check_nginx_logs() {
    log "${BLUE}Checking Nginx logs...${NC}"
    docker_exec tail -n 50 /var/log/nginx/error.log || true
    docker_exec tail -n 50 /var/log/nginx/access.log || true
}

# Function to check static files
check_static_files() {
    log "${BLUE}Checking static files in container...${NC}"
    docker_exec ls -la /app/client || true
    docker_exec cat /app/client/index.html || true
}

# Function to test backend health
test_backend() {
    log "\n${BLUE}=== Testing Internal Backend (Port $BACKEND_PORT) ===${NC}"
    local failed=0
    
    # Check if container is running
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        log "${RED}Container $CONTAINER_NAME is not running${NC}"
        docker ps
        return 1
    fi
    
    # Test internal endpoints
    local response=$(docker_exec curl -s "http://localhost:$BACKEND_PORT/api/graph/data")
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        log "${GREEN}✓ Backend /api/graph/data accessible${NC}"
        log "Response: $response"
    else
        log "${RED}✗ Backend /api/graph/data failed${NC}"
        ((failed++))
    fi
    
    response=$(docker_exec curl -s "http://localhost:$BACKEND_PORT/api/graph/data/paginated?page=0&page_size=10")
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        log "${GREEN}✓ Backend /api/graph/data/paginated accessible${NC}"
        log "Response: $response"
    else
        log "${RED}✗ Backend /api/graph/data/paginated failed${NC}"
        ((failed++))
    fi
    
    return $failed
}

# Function to test nginx
test_nginx() {
    log "\n${BLUE}=== Testing Nginx Proxy (Port $NGINX_PORT) ===${NC}"
    local failed=0
    
    # Check if nginx is running
    if ! docker_exec pgrep nginx > /dev/null; then
        log "${RED}Nginx is not running in container${NC}"
        return 1
    fi
    
    # Check nginx config
    log "Checking Nginx configuration..."
    docker_exec nginx -t || true
    
    # Check static files
    check_static_files
    
    # Test static file serving
    test_endpoint "http://localhost:$NGINX_PORT/" "Nginx static files" "" "<!DOCTYPE html>" || ((failed++))
    test_endpoint "http://localhost:$NGINX_PORT/index.html" "Nginx index.html" "" "<!DOCTYPE html>" || ((failed++))
    
    # Test API endpoint
    test_endpoint "http://localhost:$NGINX_PORT/api/graph/data" "Nginx API proxy" || ((failed++))
    
    # Check logs if there were failures
    if [ $failed -gt 0 ]; then
        check_nginx_logs
    fi
    
    return $failed
}

# Function to test network
test_network() {
    log "\n${BLUE}=== Testing RAGFlow Network ===${NC}"
    local failed=0
    
    # Get container IP
    local ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER_NAME")
    if [ -z "$ip" ]; then
        log "${RED}Failed to get container IP${NC}"
        return 1
    fi
    log "Container IP: $ip"
    
    # Test network connectivity
    test_endpoint "http://$ip:$NGINX_PORT/api/graph/data" "Network API connectivity" || ((failed++))
    test_endpoint "http://$ip:$NGINX_PORT/" "Network static files" "" "<!DOCTYPE html>" || ((failed++))
    
    # Test DNS resolution
    local dns_response=$(docker run --rm --network "$RAGFLOW_NETWORK" alpine nslookup webxr-client)
    if [ $? -eq 0 ]; then
        log "${GREEN}✓ DNS resolution working${NC}"
        log "DNS Response: $dns_response"
    else
        log "${RED}✗ DNS resolution failed${NC}"
        log "DNS Response: $dns_response"
        ((failed++))
    fi
    
    return $failed
}

# Function to test public URL
test_public() {
    log "\n${BLUE}=== Testing Public URL ===${NC}"
    local failed=0
    
    # Test HTTPS endpoint
    test_endpoint "https://$PUBLIC_DOMAIN/api/graph/data" "Public API" "-k" || ((failed++))
    
    # Test static files
    test_endpoint "https://$PUBLIC_DOMAIN/" "Public static files" "-k" "<!DOCTYPE html>" || ((failed++))
    test_endpoint "https://$PUBLIC_DOMAIN/index.html" "Public index.html" "-k" "<!DOCTYPE html>" || ((failed++))
    
    return $failed
}

# Main execution
main() {
    log "${YELLOW}Starting comprehensive endpoint tests...${NC}"
    local total_failed=0
    
    # Run tests in order
    test_backend
    local backend_failed=$?
    ((total_failed += backend_failed))
    
    test_nginx
    local nginx_failed=$?
    ((total_failed += nginx_failed))
    
    test_network
    local network_failed=$?
    ((total_failed += network_failed))
    
    test_public
    local public_failed=$?
    ((total_failed += public_failed))
    
    # Print summary
    echo
    log "${YELLOW}Test Summary:${NC}"
    echo "Backend Tests: $([ $backend_failed -eq 0 ] && echo "${GREEN}PASS${NC}" || echo "${RED}FAIL ($backend_failed failed)${NC}")"
    echo "Nginx Tests: $([ $nginx_failed -eq 0 ] && echo "${GREEN}PASS${NC}" || echo "${RED}FAIL ($nginx_failed failed)${NC}")"
    echo "Network Tests: $([ $network_failed -eq 0 ] && echo "${GREEN}PASS${NC}" || echo "${RED}FAIL ($network_failed failed)${NC}")"
    echo "Public URL Tests: $([ $public_failed -eq 0 ] && echo "${GREEN}PASS${NC}" || echo "${RED}FAIL ($public_failed failed)${NC}")"
    
    if [ $total_failed -eq 0 ]; then
        log "${GREEN}All tests passed successfully!${NC}"
        exit 0
    else
        log "${RED}${total_failed} tests failed${NC}"
        exit 1
    fi
}

# Run main function
main
