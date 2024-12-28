#!/bin/bash

# Enable error reporting and strict mode
set -euo pipefail
IFS=$'\n\t'

# Color setup using tput
if [ -t 1 ]; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    BOLD=$(tput bold)
    NC=$(tput sgr0)
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    BOLD=""
    NC=""
fi

# Configuration
CONTAINER_NAME="logseq-xr-webxr"
BACKEND_PORT=3001
NGINX_PORT=4000
PUBLIC_DOMAIN="www.visionflow.info"
TIMEOUT=5
WEBSOCKET_TIMEOUT=10

# Test environments
declare -A ENDPOINTS=(
    ["internal"]="http://localhost:$NGINX_PORT"
    ["container"]="http://127.0.0.1:$BACKEND_PORT"
    ["ragflow"]="http://logseq-xr-webxr:$BACKEND_PORT"
    ["external"]="https://$PUBLIC_DOMAIN"
)

# API endpoints to test
declare -A API_ENDPOINTS=(
    ["graph_data"]="/api/graph/data"
    ["graph_update"]="/api/graph/update"
    ["graph_paginated"]="/api/graph/data/paginated"
    ["settings_root"]="/api/settings"
    ["settings_update"]="/api/settings/update"
    ["settings_visualization"]="/api/settings/visualization"
    ["websocket_control"]="/api/websocket/control"
    ["files"]="/api/files"
)

# Function to log messages with timestamp
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    printf "%s[%s]%s %s\n" "${BOLD}" "${timestamp}" "${NC}" "$1"
}

# Function to log success
log_success() {
    log "${GREEN}✓ $1${NC}"
}

# Function to log error
log_error() {
    log "${RED}✗ $1${NC}"
}

# Function to log info
log_info() {
    log "${BLUE}$1${NC}"
}

# Function to log header
log_header() {
    printf "\n%s%s===%s %s %s===%s\n" "${BOLD}" "${YELLOW}" "${NC}" "$1" "${YELLOW}" "${NC}"
}

# Function to pretty print JSON
pretty_json() {
    if command -v jq >/dev/null 2>&1; then
        echo "$1" | jq '.'
    else
        echo "$1"
    fi
}

# Function to test endpoint and show response
test_endpoint() {
    local url="$1"
    local description="$2"
    local method="${3:-GET}"
    local data="${4:-}"
    local extra_opts="${5:-}"
    
    log_info "Testing $method $description..."
    log_info "URL: $url"
    
    # Use docker exec for container endpoints
    if [[ "$url" == *"127.0.0.1"* ]] || [[ "$url" == *"localhost"* ]]; then
        local internal_url="http://localhost:4000${url#*:3001}"
        log_info "Testing internal URL: $internal_url"
        
        # Only capture status code, discard response body
        local http_code
        if [[ -n "$data" ]]; then
            http_code=$(docker exec $CONTAINER_NAME curl -X $method -m $TIMEOUT -s -o /dev/null -w "%{http_code}" \
                -H "Accept: application/json" -H "Content-Type: application/json" \
                -d "$data" $extra_opts "$internal_url")
        else
            http_code=$(docker exec $CONTAINER_NAME curl -X $method -m $TIMEOUT -s -o /dev/null -w "%{http_code}" \
                -H "Accept: application/json" $extra_opts "$internal_url")
        fi
    else
        # For external endpoints, use regular curl
        local http_code
        if [[ -n "$data" ]]; then
            http_code=$(curl -X $method -m $TIMEOUT -s -o /dev/null -w "%{http_code}" \
                -H "Accept: application/json" -H "Content-Type: application/json" \
                -d "$data" $extra_opts "$url")
        else
            http_code=$(curl -X $method -m $TIMEOUT -s -o /dev/null -w "%{http_code}" \
                -H "Accept: application/json" $extra_opts "$url")
        fi
    fi
    
    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        log_success "$description successful (HTTP $http_code)"
        return 0
    else
        log_error "$description failed (HTTP $http_code)"
        return 1
    fi
}

# Function to test WebSocket connection
test_websocket() {
    local url="$1"
    local description="$2"
    
    log_info "Testing WebSocket connection to $url..."
    
    # Use docker exec for container endpoints
    if [[ "$url" == *"127.0.0.1"* ]] || [[ "$url" == *"localhost"* ]]; then
        local internal_url="http://localhost:4000${url#*:3001}"
        log_info "Testing internal WebSocket URL: $internal_url"
        
        # Test WebSocket upgrade
        local response=$(docker exec $CONTAINER_NAME curl -i -N \
            -H "Connection: Upgrade" \
            -H "Upgrade: websocket" \
            -H "Host: localhost:4000" \
            -H "Origin: http://localhost:4000" \
            -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
            -H "Sec-WebSocket-Version: 13" \
            "$internal_url" 2>&1)
            
        # Check for successful upgrade (101) or normal success (200)
        if echo "$response" | grep -q "HTTP/1.1 101\|HTTP/1.1 200"; then
            log_success "$description successful (WebSocket upgrade)"
            return 0
        else
            local status=$(echo "$response" | grep -oP "HTTP/1.1 \K[0-9]+" || echo "000")
            log_error "$description failed (HTTP $status)"
            return 1
        fi
    else
        # For external endpoints, just check if it's accessible
        local http_code=$(curl -m $WEBSOCKET_TIMEOUT -s -o /dev/null -w "%{http_code}" "$url")
        if [[ "$http_code" =~ ^2[0-9][0-9]$ ]] || [[ "$http_code" == "400" ]] || [[ "$http_code" == "426" ]]; then
            log_success "$description accessible (HTTP $http_code)"
            return 0
        else
            log_error "$description failed (HTTP $http_code)"
            return 1
        fi
    fi
}

# Function to test environment
test_environment() {
    local env="$1"
    local base_url="${ENDPOINTS[$env]}"
    local failed=0
    
    log_header "Testing $env environment ($base_url)"
    
    # Test graph endpoints
    test_endpoint "$base_url${API_ENDPOINTS[graph_data]}" "$env full graph data" || ((failed++))
    test_endpoint "$base_url${API_ENDPOINTS[graph_paginated]}" "$env paginated graph data" || ((failed++))
    test_endpoint "$base_url${API_ENDPOINTS[graph_update]}" "$env graph update" "POST" '{"nodes": [], "edges": []}' || ((failed++))
    
    # Test settings endpoints
    test_endpoint "$base_url${API_ENDPOINTS[settings_root]}" "$env all settings" || ((failed++))
    test_endpoint "$base_url${API_ENDPOINTS[settings_visualization]}" "$env visualization settings" || ((failed++))
    test_endpoint "$base_url${API_ENDPOINTS[websocket_control]}" "$env WebSocket control API" || ((failed++))
    
    # Test settings endpoints for each category
    local categories=(
        "system.network"
        "system.websocket"
        "system.security"
        "system.debug"
        "visualization.animations"
        "visualization.ar"
        "visualization.audio"
        "visualization.bloom"
        "visualization.edges"
        "visualization.hologram"
        "visualization.labels"
        "visualization.nodes"
        "visualization.physics"
        "visualization.rendering"
    )
    
    for category in "${categories[@]}"; do
        # Test category endpoint
        test_endpoint "$base_url/api/settings/$category" "$env $category settings" || ((failed++))
        
        # Test individual setting update
        local setting="enabled"
        test_endpoint "$base_url/api/settings/$category/$setting" "$env get $category.$setting" || ((failed++))
        test_endpoint "$base_url/api/settings/$category/$setting" "$env update $category.$setting" "PUT" '{"value": true}' || ((failed++))
    done
    
    # Test WebSocket endpoints
    local ws_protocol="ws"
    [[ "$base_url" == https://* ]] && ws_protocol="wss"
    local ws_base_url="${base_url/http:/$ws_protocol:}"
    local ws_base_url="${ws_base_url/https:/$ws_protocol:}"
    
    test_websocket "$ws_base_url/wss" "$env Binary Protocol" || ((failed++))
    test_endpoint "$base_url/api/visualization/settings/websocket" "$env WebSocket Settings" || ((failed++))
    
    return $failed
}

# Function to test container internal endpoints
test_container_endpoints() {
    local failed=0
    log_header "Testing container internal endpoints"
    
    # Test graph data endpoints
    log_info "Testing graph data endpoints..."
    test_endpoint "${ENDPOINTS[container]}${API_ENDPOINTS[graph_data]}" "Container internal graph data endpoint" || ((failed++))
    test_endpoint "${ENDPOINTS[container]}${API_ENDPOINTS[graph_paginated]}" "Container internal paginated graph data endpoint" || ((failed++))
    
    # Test settings endpoints - Note: These are expected to fail with 500 error
    log_info "Testing settings endpoints (expected to fail with 500 error)..."
    local status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
        -H "Accept: application/json" "http://localhost:4000/api/settings")
    if [[ "$status" == "500" ]]; then
        log_info "Settings endpoint returned expected 500 error (server-side settings disabled)"
    else
        log_error "Settings endpoint returned unexpected status (HTTP $status)"
        ((failed++))
    fi
    
    # Test WebSocket endpoint
    log_info "Testing WebSocket endpoint..."
    test_websocket "${ENDPOINTS[container]}/wss" "Container internal WebSocket endpoint" || ((failed++))
    
    # Test file endpoints
    log_info "Testing file endpoints..."
    test_endpoint "${ENDPOINTS[container]}${API_ENDPOINTS[files]}" "Container internal files endpoint" || ((failed++))
    
    if [ $failed -eq 0 ]; then
        log_success "All container internal endpoints passed"
    else
        log_error "$failed container internal endpoint tests failed"
    fi
    
    return $failed
}

# Function to test backend directly
test_backend_directly() {
    local failed=0
    log_header "Testing backend directly"
    
    # Test backend on port 3001
    log_info "Testing backend endpoints on port 3001..."
    
    # Test graph data endpoints
    local graph_endpoints=(
        "/api/graph/data"
        "/api/graph/data/paginated"
    )
    
    for endpoint in "${graph_endpoints[@]}"; do
        log_info "Testing $endpoint..."
        local status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
            -H "Accept: application/json" "http://localhost:3001$endpoint")
        if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
            log_success "Backend $endpoint successful (HTTP $status)"
        else
            log_error "Backend $endpoint failed (HTTP $status)"
            ((failed++))
        fi
    done
    
    # Test settings endpoints - Note: 500 errors are expected here
    local settings_endpoints=(
        "/api/settings"
        "/api/settings/"
        "/api/settings/visualization"
        "/api/settings/websocket"
        "/api/settings/system"
        "/api/settings/all"
    )
    
    for endpoint in "${settings_endpoints[@]}"; do
        log_info "Testing $endpoint (expected to return 500)..."
        status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
            -H "Accept: application/json" "http://localhost:3001$endpoint")
        if [[ "$status" == "500" ]]; then
            log_info "Settings endpoint $endpoint returned expected 500 error (server-side settings disabled)"
        else
            log_error "Settings endpoint $endpoint returned unexpected status (HTTP $status)"
            ((failed++))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        log_success "All backend tests passed"
    else
        log_error "$failed backend tests failed"
    fi
    
    return $failed
}

# Function to test RAGFlow network connectivity
test_ragflow_connectivity() {
    local failed=0
    log_header "Testing RAGFlow network connectivity"
    
    # Test RAGFlow server connectivity
    docker exec $CONTAINER_NAME curl -s -f -H "Accept: application/json" "http://ragflow-server/v1/" > /dev/null || {
        log_error "RAGFlow server connectivity failed"
        ((failed++))
    }
    
    # Test Redis connectivity
    docker exec $CONTAINER_NAME curl -s -H "Accept: application/json" "http://ragflow-redis:6379/ping" > /dev/null || {
        log_error "RAGFlow Redis connectivity failed"
        ((failed++))
    }
    
    # Test MySQL connectivity (just check if port is open)
    docker exec $CONTAINER_NAME timeout 1 bash -c "cat < /dev/null > /dev/tcp/ragflow-mysql/3306" 2>/dev/null || {
        log_error "RAGFlow MySQL connectivity failed"
        ((failed++))
    }
    
    # Test Elasticsearch connectivity
    docker exec $CONTAINER_NAME curl -s -f -H "Accept: application/json" "http://ragflow-es-01:9200/_cluster/health" > /dev/null || {
        log_error "RAGFlow Elasticsearch connectivity failed"
        ((failed++))
    }
    
    if [ $failed -eq 0 ]; then
        log_success "All RAGFlow network connectivity tests passed"
    else
        log_error "$failed RAGFlow network connectivity tests failed"
    fi
    
    return $failed
}

# Function to check container logs
check_container_logs() {
    local container_name="$1"
    local lines="${2:-50}"
    
    log_header "Container Logs ($container_name)"
    
    # Show container status
    local status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null)
    local health=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null)
    local started=$(docker inspect -f '{{.State.StartedAt}}' "$container_name" 2>/dev/null)
    local networks=$(docker inspect -f '{{range $net, $conf := .NetworkSettings.Networks}}{{$net}} {{end}}' "$container_name" 2>/dev/null)
    
    log_info "Status: $status"
    log_info "Health: $health"
    log_info "Started At: $started"
    log_info "Networks: $networks"
    
    # Show recent health check logs, filtering out large JSON responses
    log_info "Recent health check logs:"
    docker logs "$container_name" 2>&1 | grep -v '{"nodes":\[{' | grep -v '"edges":\[{' | tail -n "$lines"
    
    # Show recent container logs
    log_info "Last $lines lines of container logs:"
    docker logs "$container_name" 2>&1 | grep -v '{"nodes":\[{' | grep -v '"edges":\[{' | grep "\[" | tail -n "$lines"
}

# Function to check nginx config
check_nginx_config() {
    log_header "Nginx Configuration"
    
    # Check nginx config inside container
    docker exec $CONTAINER_NAME nginx -T 2>/dev/null | grep -A 10 "location /api" || {
        log_error "Failed to get nginx configuration"
        return 1
    }
    
    # Check nginx process
    docker exec $CONTAINER_NAME ps aux | grep "[n]ginx" || {
        log_error "Nginx process not running"
        return 1
    }
    
    return 0
}

# Function to wait for webxr to be ready
wait_for_webxr() {
    local max_attempts=30
    local attempt=1
    local delay=2
    
    log_info "Waiting for webxr container to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker logs $CONTAINER_NAME 2>&1 | grep -q "Frontend is healthy"; then
            log_success "webxr container is ready"
            return 0
        fi
        log_info "Attempt $attempt/$max_attempts: webxr not ready yet, waiting ${delay}s..."
        sleep $delay
        ((attempt++))
    done
    
    log_error "Timed out waiting for webxr container to be ready"
    docker logs $CONTAINER_NAME
    return 1
}

# Function to test Cloudflare tunnel
test_cloudflare_tunnel() {
    log_header "Testing Cloudflare Tunnel"
    
    local failed=0
    local tunnel_id=""
    local tunnel_name=""
    local tunnel_hostname=""
    
    # Check if cloudflared is running
    if ! docker ps | grep -q "cloudflared-tunnel"; then
        log_error "Cloudflared tunnel container is not running"
        return 1
    fi
    
    # Check tunnel status
    log_info "Checking tunnel status..."
    if ! docker exec cloudflared-tunnel cloudflared tunnel info 2>/dev/null | grep -q "Active"; then
        log_error "Cloudflared tunnel is not active"
        ((failed++))
    else
        log_success "Cloudflared tunnel is active"
    fi
    
    # Test tunnel connectivity
    log_info "Testing tunnel connectivity..."
    if ! curl -s -o /dev/null -w "%{http_code}" "https://$PUBLIC_DOMAIN" | grep -q "200"; then
        log_error "Cannot reach public domain through tunnel"
        ((failed++))
    else
        log_success "Public domain is accessible through tunnel"
    fi
    
    return $failed
}

# Main function
main() {
    local total_failed=0
    
    # Wait for services to be ready
    wait_for_webxr || exit 1
    
    # Run tests
    test_container_endpoints
    ((total_failed+=$?))
    
    test_backend_directly
    ((total_failed+=$?))
    
    test_ragflow_connectivity
    ((total_failed+=$?))
    
    test_cloudflare_tunnel
    ((total_failed+=$?))
    
    # Check container logs for errors
    check_container_logs
    ((total_failed+=$?))
    
    # Check nginx config
    check_nginx_config
    ((total_failed+=$?))
    
    # Final summary
    if [ $total_failed -eq 0 ]; then
        log_success "All tests passed successfully!"
        exit 0
    else
        log_error "$total_failed tests failed"
        exit 1
    fi
}

# Make script executable
chmod +x "$0"

# Run main function
main
