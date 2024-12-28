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
    ["nginx"]="http://localhost:$NGINX_PORT"
    ["backend"]="http://127.0.0.1:$BACKEND_PORT"
    ["ragflow"]="http://localhost:3001"
    ["external"]="https://$PUBLIC_DOMAIN"
)

# Store results for comparison
declare -A TEST_RESULTS

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

# Function to test WebSocket connection with detailed response
test_websocket() {
    local url="$1"
    local description="$2"
    local env="$3"
    
    log_info "Testing WebSocket connection to $url..."
    
    # Convert http/https to ws/wss
    local ws_url="${url/http:/ws:}"
    ws_url="${ws_url/https:/wss:}"
    ws_url="${ws_url}/wss"
    
    log_info "WebSocket URL: $ws_url"
    
    # Use websocat for testing if available
    if command -v websocat >/dev/null 2>&1; then
        local response
        if [[ "$url" == *"localhost"* ]] || [[ "$url" == *"127.0.0.1"* ]]; then
            response=$(docker exec $CONTAINER_NAME timeout $WEBSOCKET_TIMEOUT websocat "$ws_url" 2>&1)
        else
            response=$(timeout $WEBSOCKET_TIMEOUT websocat "$ws_url" 2>&1)
        fi
        
        local status=$?
        if [ $status -eq 0 ] || [ $status -eq 124 ]; then
            log_success "$description successful (WebSocket connection established)"
            TEST_RESULTS["${env}_ws"]="success"
            return 0
        else
            log_error "$description failed (WebSocket connection failed)"
            TEST_RESULTS["${env}_ws"]="failed"
            return 1
        fi
    else
        # Fallback to curl for upgrade request
        local response
        if [[ "$url" == *"localhost"* ]] || [[ "$url" == *"127.0.0.1"* ]]; then
            response=$(docker exec $CONTAINER_NAME curl -i -N \
                -H "Connection: Upgrade" \
                -H "Upgrade: websocket" \
                -H "Host: ${url#*://}" \
                -H "Origin: $url" \
                -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
                -H "Sec-WebSocket-Version: 13" \
                "$url/wss" 2>&1)
        else
            response=$(curl -i -N \
                -H "Connection: Upgrade" \
                -H "Upgrade: websocket" \
                -H "Host: ${url#*://}" \
                -H "Origin: $url" \
                -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
                -H "Sec-WebSocket-Version: 13" \
                "$url/wss" 2>&1)
        fi
        
        if echo "$response" | grep -q "HTTP/1.1 101\|HTTP/1.1 200"; then
            log_success "$description successful (WebSocket upgrade)"
            TEST_RESULTS["${env}_ws"]="success"
            return 0
        else
            local status=$(echo "$response" | grep -oP "HTTP/1.1 \K[0-9]+" || echo "000")
            log_error "$description failed (HTTP $status)"
            TEST_RESULTS["${env}_ws"]="failed"
            log_error "Response: $response"
            return 1
        fi
    fi
}

# Function to test environment with detailed response capture
test_environment() {
    local env="$1"
    local base_url="${ENDPOINTS[$env]}"
    local failed=0
    
    log_header "Testing $env environment ($base_url)"
    
    # Store base URL for comparison
    TEST_RESULTS["${env}_base_url"]="$base_url"
    
    # Test graph endpoints with response capture
    for endpoint in "graph_data" "graph_paginated"; do
        local response
        local status
        if [[ "$base_url" == *"localhost"* ]] || [[ "$base_url" == *"127.0.0.1"* ]]; then
            response=$(docker exec $CONTAINER_NAME curl -s -w "\n%{http_code}" \
                -H "Accept: application/json" \
                "$base_url${API_ENDPOINTS[$endpoint]}")
        else
            response=$(curl -s -w "\n%{http_code}" \
                -H "Accept: application/json" \
                "$base_url${API_ENDPOINTS[$endpoint]}")
        fi
        
        status=$(echo "$response" | tail -n1)
        response=$(echo "$response" | head -n-1)
        
        if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
            log_success "$env $endpoint successful (HTTP $status)"
            TEST_RESULTS["${env}_${endpoint}"]="success"
            TEST_RESULTS["${env}_${endpoint}_response"]="$response"
        else
            log_error "$env $endpoint failed (HTTP $status)"
            TEST_RESULTS["${env}_${endpoint}"]="failed"
            TEST_RESULTS["${env}_${endpoint}_error"]="$response"
            ((failed++))
        fi
    done
    
    # Test graph update endpoint
    local update_response
    if [[ "$base_url" == *"localhost"* ]] || [[ "$base_url" == *"127.0.0.1"* ]]; then
        update_response=$(docker exec $CONTAINER_NAME curl -s -w "\n%{http_code}" \
            -H "Accept: application/json" -H "Content-Type: application/json" \
            -X POST -d '{"nodes": [], "edges": []}' \
            "$base_url${API_ENDPOINTS[graph_update]}")
    else
        update_response=$(curl -s -w "\n%{http_code}" \
            -H "Accept: application/json" -H "Content-Type: application/json" \
            -X POST -d '{"nodes": [], "edges": []}' \
            "$base_url${API_ENDPOINTS[graph_update]}")
    fi
    
    local update_status=$(echo "$update_response" | tail -n1)
    update_response=$(echo "$update_response" | head -n-1)
    
    if [[ "$update_status" =~ ^2[0-9][0-9]$ ]]; then
        log_success "$env graph update successful (HTTP $update_status)"
        TEST_RESULTS["${env}_graph_update"]="success"
    else
        log_error "$env graph update failed (HTTP $update_status)"
        TEST_RESULTS["${env}_graph_update"]="failed"
        ((failed++))
    fi
    
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
    
    # Test WebSocket endpoints with detailed response
    test_websocket "$base_url" "$env Binary Protocol" "$env" || ((failed++))
    
    # Test WebSocket settings endpoint
    local ws_settings_response
    if [[ "$base_url" == *"localhost"* ]] || [[ "$base_url" == *"127.0.0.1"* ]]; then
        ws_settings_response=$(docker exec $CONTAINER_NAME curl -s -w "\n%{http_code}" \
            -H "Accept: application/json" \
            "$base_url/api/visualization/settings/websocket")
    else
        ws_settings_response=$(curl -s -w "\n%{http_code}" \
            -H "Accept: application/json" \
            "$base_url/api/visualization/settings/websocket")
    fi
    
    local ws_settings_status=$(echo "$ws_settings_response" | tail -n1)
    ws_settings_response=$(echo "$ws_settings_response" | head -n-1)
    
    if [[ "$ws_settings_status" =~ ^2[0-9][0-9]$ ]]; then
        log_success "$env WebSocket Settings successful (HTTP $ws_settings_status)"
        TEST_RESULTS["${env}_ws_settings"]="success"
        TEST_RESULTS["${env}_ws_settings_response"]="$ws_settings_response"
    else
        log_error "$env WebSocket Settings failed (HTTP $ws_settings_status)"
        TEST_RESULTS["${env}_ws_settings"]="failed"
        TEST_RESULTS["${env}_ws_settings_error"]="$ws_settings_response"
        ((failed++))
    fi
    
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

# Function to compare environment results
compare_environments() {
    log_header "Environment Comparison"
    
    # Compare nginx vs external responses
    log_info "Comparing nginx (localhost:4000) vs external (www.visionflow.info) responses:"
    
    # Compare WebSocket results
    log_info "\nWebSocket Comparison:"
    printf "%-20s %-15s %-15s\n" "Endpoint" "Nginx" "External"
    printf "%-20s %-15s %-15s\n" "WebSocket" "${TEST_RESULTS[nginx_ws]:-N/A}" "${TEST_RESULTS[external_ws]:-N/A}"
    printf "%-20s %-15s %-15s\n" "WS Settings" "${TEST_RESULTS[nginx_ws_settings]:-N/A}" "${TEST_RESULTS[external_ws_settings]:-N/A}"
    
    # Compare REST endpoints
    log_info "\nREST Endpoint Comparison:"
    printf "%-20s %-15s %-15s\n" "Endpoint" "Nginx" "External"
    for endpoint in "graph_data" "graph_paginated" "graph_update"; do
        printf "%-20s %-15s %-15s\n" "$endpoint" "${TEST_RESULTS[nginx_${endpoint}]:-N/A}" "${TEST_RESULTS[external_${endpoint}]:-N/A}"
    done
    
    # Compare RAGFlow responses
    log_info "\nRAGFlow (port 3001) Comparison:"
    printf "%-20s %-15s\n" "Endpoint" "Status"
    for endpoint in "graph_data" "graph_paginated" "graph_update"; do
        printf "%-20s %-15s\n" "$endpoint" "${TEST_RESULTS[ragflow_${endpoint}]:-N/A}"
    done
}

# Main function
main() {
    local total_failed=0
    
    # Wait for services to be ready
    wait_for_webxr || exit 1
    
    # Test each environment
    for env in "nginx" "backend" "ragflow" "external"; do
        test_environment "$env"
        ((total_failed+=$?))
    done
    
    # Additional tests
    test_ragflow_connectivity
    ((total_failed+=$?))
    
    test_cloudflare_tunnel
    ((total_failed+=$?))
    
    # Compare results
    compare_environments
    
    # Check container logs for errors
    check_container_logs $CONTAINER_NAME
    
    # Check nginx config
    check_nginx_config
    
    # Final summary
    if [ $total_failed -eq 0 ]; then
        log_success "All tests passed successfully!"
        exit 0
    else
        log_error "$total_failed tests failed"
        log_info "See environment comparison above for details on differences between localhost:4000 and www.visionflow.info"
        exit 1
    fi
}

# Make script executable
chmod +x "$0"

# Run main function
main
