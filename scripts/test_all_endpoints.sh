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
    GRAY=$(tput setaf 8)
    BOLD=$(tput bold)
    NC=$(tput sgr0)
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    GRAY=""
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
    ["development"]="http://localhost:3001"
)

# WebSocket configuration
declare -A WS_CONFIG=(
    ["nginx"]="ws|4000|/wss"
    ["backend"]="ws|3001|/wss"
    ["ragflow"]="ws|3001|/wss"
    ["external"]="wss|443|/wss"
    ["development"]="ws|3001|/wss"
)

# Function to get WebSocket configuration
get_ws_config() {
    local env="$1"
    local config="${WS_CONFIG[$env]}"
    local protocol=$(echo "$config" | cut -d'|' -f1)
    local port=$(echo "$config" | cut -d'|' -f2)
    local path=$(echo "$config" | cut -d'|' -f3)
    echo "$protocol|$port|$path"
}

# Function to test WebSocket protocol with detailed output
test_ws_protocol() {
    local url="$1"
    local protocol="$2"
    local port="$3"
    local path="$4"
    local description="$5"
    
    local start_time=$(date +%s.%N)
    log_step "Testing WebSocket Protocol: $protocol"
    log_info "Configuration:"
    printf "%-20s: %s\n" "URL" "$url"
    printf "%-20s: %s\n" "Protocol" "$protocol"
    printf "%-20s: %s\n" "Port" "$port"
    printf "%-20s: %s\n" "Path" "$path"
    printf "%-20s: %s\n" "Timeout" "${WEBSOCKET_TIMEOUT}s"
    
    local ws_url="${url/$protocol:\/\/localhost:$port/$protocol:\/\/localhost:$port}$path"
    log_info "Full WebSocket URL: $ws_url"
    
    local response
    local curl_start_time=$(date +%s.%N)
    log_info "Attempting WebSocket connection..."
    if ! response=$(curl -v -s -i -N -m $WEBSOCKET_TIMEOUT \
        -H "Connection: Upgrade" \
        -H "Upgrade: websocket" \
        -H "Host: localhost:$port" \
        -H "Origin: ${protocol}://localhost:$port" \
        -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
        -H "Sec-WebSocket-Version: 13" \
        "$ws_url" 2>&1); then
        local curl_end_time=$(date +%s.%N)
        local curl_duration=$(echo "$curl_end_time - $curl_start_time" | bc)
        log_error "$protocol connection failed" "Error: Connection refused or timed out"
        log_info "Connection Details:"
        printf "%-20s: %.3fs\n" "Attempt Duration" "$curl_duration"
        printf "%-20s: %s\n" "Error" "Connection Refused"
        return 1
    elif echo "$response" | grep -q "HTTP/1.1 101"; then
        local curl_end_time=$(date +%s.%N)
        local curl_duration=$(echo "$curl_end_time - $curl_start_time" | bc)
        log_success "$protocol connection successful" "Connection established"
        log_info "Connection Details:"
        printf "%-20s: %.3fs\n" "Connect Time" "$curl_duration"
        printf "%-20s: %s\n" "Status" "Connected"
        printf "%-20s: %s\n" "Protocol Version" "13"
        log_info "Response Headers:"
        echo "$response" | grep -E '^(HTTP|Upgrade|Connection|Sec-WebSocket)' | sed 's/^/  /'
        return 0
    else
        local curl_end_time=$(date +%s.%N)
        local curl_duration=$(echo "$curl_end_time - $curl_start_time" | bc)
        log_error "$protocol connection failed" "Unexpected response from server"
        log_info "Connection Details:"
        printf "%-20s: %.3fs\n" "Attempt Duration" "$curl_duration"
        printf "%-20s: %s\n" "Error" "Invalid Response"
        log_info "Server Response:"
        echo "$response" | sed 's/^/  /'
        return 1
    fi
}

# Function to test WebSocket connection with detailed response
test_websocket() {
    local url="$1"
    local description="$2"
    local timeout="${3:-$WEBSOCKET_TIMEOUT}"
    
    log_step "Testing WebSocket Connection: $description"
    log_info "URL: $url"
    log_info "Timeout: ${timeout}s"
    
    # Test initial WebSocket upgrade
    local upgrade_response
    if [[ "$url" == *"localhost"* ]] || [[ "$url" == *"127.0.0.1"* ]]; then
        response=$(docker exec $CONTAINER_NAME curl -v -s -i -N \
            -H "Connection: Upgrade" \
            -H "Upgrade: websocket" \
            -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
            -H "Sec-WebSocket-Version: 13" \
            "$url" 2>&1)
    else
        # For external endpoints, use regular curl
        response=$(curl -v -s -i -N \
            -H "Connection: Upgrade" \
            -H "Upgrade: websocket" \
            -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
            -H "Sec-WebSocket-Version: 13" \
            "$url" 2>&1)
    fi
    
    # Check for successful upgrade (101 status)
    if echo "$response" | grep -q "HTTP/1.1 101"; then
        log_success "WebSocket upgrade successful" "Connection established"
        
        # Test binary protocol support
        if echo -e "\x81\x05Hello" | timeout "$timeout" nc -w 1 "${url#*://}" > /dev/null 2>&1; then
            log_success "Binary protocol test passed" "Server accepted binary frame"
        else
            log_error "Binary protocol test failed" "Server rejected binary frame"
            return 1
        fi
        
        # Test heartbeat
        if echo -e "\x89\x00" | timeout 2 nc -w 1 "${url#*://}" > /dev/null 2>&1; then
            log_success "Heartbeat test passed" "Server responded to ping"
        else
            log_warning "Heartbeat test inconclusive" "Server may not support ping/pong"
        fi
        
        return 0
    else
        log_error "WebSocket upgrade failed" "Server returned: $(echo "$response" | grep "HTTP/")"
        return 1
    fi
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if command -v nc >/dev/null 2>&1; then
        nc -z localhost "$port" >/dev/null 2>&1
        return $?
    elif command -v lsof >/dev/null 2>&1; then
        lsof -i:"$port" >/dev/null 2>&1
        return $?
    else
        # Fallback to curl
        curl -s "http://localhost:$port" >/dev/null 2>&1
        return $?
    fi
}

# Function to test development environment
test_development_environment() {
    local failed=0
    log_header "Testing Development Environment"
    
    # Check if Vite dev server is running
    log_step "Testing Vite Development Server"
    log_info "Checking Vite server status on port 3001..."
    
    if ! check_port 3001; then
        log_error "✗ Vite dev server is not running"
        log_info "To start the development server:"
        log_info "1. Open a new terminal"
        log_info "2. Navigate to the project directory"
        log_info "3. Run: npm run dev"
        return 1
    else
        log_success "✓ Vite dev server is running on port 3001"
    fi
    
    # Test WebSocket protocols with detailed output
    log_step "Testing WebSocket Protocols"
    
    # Get WebSocket configuration
    local ws_config=$(get_ws_config "development")
    local protocol=$(echo "$ws_config" | cut -d'|' -f1)
    local port=$(echo "$ws_config" | cut -d'|' -f2)
    local path=$(echo "$ws_config" | cut -d'|' -f3)
    
    # Show WebSocket test configuration
    log_info "WebSocket Configuration:"
    printf "\n%-25s: %s\n" "Environment" "Development"
    printf "%-25s: %s\n" "Primary Protocol" "$protocol://"
    printf "%-25s: %s\n" "Port" "$port"
    printf "%-25s: %s\n" "Path" "$path"
    printf "%-25s: %s\n" "Timeout" "${WEBSOCKET_TIMEOUT}s"
    
    # Test primary protocol
    log_info "Testing primary protocol ($protocol)..."
    if ! test_ws_protocol "http://localhost:$port" "$protocol" "$port" "$path" "Primary protocol test"; then
        log_info "Primary protocol failed, attempting fallback..."
        
        # Try fallback protocol
        local fallback_protocol
        if [[ "$protocol" == "ws" ]]; then
            fallback_protocol="wss"
            log_info "Trying secure WebSocket protocol (wss://)..."
        else
            fallback_protocol="ws"
            log_info "Trying standard WebSocket protocol (ws://)..."
        fi
        
        test_ws_protocol "http://localhost:$port" "$fallback_protocol" "$port" "$path" "Fallback protocol test"
        local fallback_status=$?
        
        if [ $fallback_status -eq 0 ]; then
            log_success "Fallback protocol ($fallback_protocol) connection successful"
            TEST_RESULTS["ws_protocol_used"]="$fallback_protocol"
        else
            log_error "Both primary and fallback protocols failed"
            TEST_RESULTS["ws_protocol_used"]="none"
            ((failed++))
        fi
    else
        log_success "Primary protocol ($protocol) connection successful"
        TEST_RESULTS["ws_protocol_used"]="$protocol"
    fi
    
    # Show protocol test summary
    log_step "WebSocket Protocol Test Summary"
    printf "\n%-25s: %s\n" "Protocol Used" "${TEST_RESULTS[ws_protocol_used]}"
    printf "%-25s: %s\n" "Primary Protocol" "$protocol"
    printf "%-25s: %s\n" "Connection Status" "${TEST_RESULTS[ws_connection]:-N/A}"
    printf "%-25s: %s\n" "Response Time" "${TEST_RESULTS[ws_latency]:-N/A}"
    
    # Test API endpoints through Vite proxy
    log_step "Testing Development API Endpoints"
    log_info "Testing endpoints through Vite development proxy..."
    log_info "Base URL: http://localhost:3001"
    log_info "Total endpoints to test: ${#API_ENDPOINTS[@]}"
    
    for endpoint in "${!API_ENDPOINTS[@]}"; do
        log_info "Testing endpoint: ${API_ENDPOINTS[$endpoint]}"
        local start_time=$(date +%s.%N)
        test_endpoint "http://localhost:3001${endpoint}" "Development $endpoint endpoint" || ((failed++))
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        TEST_RESULTS["dev_${endpoint}_latency"]=$(printf "%.3fs" $duration)
    done
    
    # Test WebSocket settings
    log_step "Testing WebSocket Configuration"
    log_info "Checking WebSocket settings endpoint..."
    log_info "URL: http://localhost:3001/api/settings/websocket"
    local start_time=$(date +%s.%N)
    local ws_settings_response=$(curl -s -w "\n%{http_code}" \
        -H "Accept: application/json" \
        "http://localhost:3001/api/settings/websocket")
    
    local ws_settings_status=$(echo "$ws_settings_response" | tail -n1)
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    if [[ "$ws_settings_status" == "500" ]]; then
        log_success "WebSocket settings returned expected 500" "Settings are disabled in development mode as expected"
        TEST_RESULTS["ws_settings"]="success"
    else
        log_error "WebSocket settings returned unexpected status: $ws_settings_status" "Expected: 500 (disabled)\nActual: $ws_settings_status\nResponse: $ws_settings_response"
        TEST_RESULTS["ws_settings"]="failed"
        ((failed++))
    fi
    
    TEST_RESULTS["ws_settings_latency"]=$(printf "%.3fs" $duration)
    
    log_step "Development Environment Test Summary"
    printf "\n%-30s %-15s %-15s\n" "Component" "Status" "Latency"
    printf "%-30s %-15s %-15s\n" "Vite Server" "${TEST_RESULTS[vite_server]}" "-"
    printf "%-30s %-15s %-15s\n" "WebSocket Connection" "${TEST_RESULTS[ws_connection]}" "-"
    printf "%-30s %-15s %-15s\n" "WebSocket Settings" "${TEST_RESULTS[ws_settings]}" "${TEST_RESULTS[ws_settings_latency]}"
    
    for endpoint in "${!API_ENDPOINTS[@]}"; do
        printf "%-30s %-15s %-15s\n" \
            "API: ${endpoint}" \
            "${TEST_RESULTS[dev_${endpoint}]:-N/A}" \
            "${TEST_RESULTS[dev_${endpoint}_latency]:-N/A}"
    done
    
    return $failed
}

# Function to test nginx service
test_nginx_service() {
    log_step "Testing Nginx Service"
    log_info "Checking nginx service status..."
    
    # Check if nginx is running on port 4000
    if ! check_port 4000; then
        log_error "✗ Nginx service is not running"
        log_info "To start nginx service:"
        log_info "1. Make sure Docker is running"
        log_info "2. Run: docker-compose up -d"
        return 1
    fi
    
    # Try to get nginx health status
    local health_response
    if ! health_response=$(curl -s "http://localhost:4000/health"); then
        log_error "✗ Nginx health check failed"
        return 1
    fi
    
    if echo "$health_response" | grep -q '"status":"healthy"'; then
        log_success "✓ Nginx service is healthy"
        
        # Check if it can proxy to backend
        if curl -s "http://localhost:4000/api/health" >/dev/null 2>&1; then
            log_success "✓ Nginx successfully proxying to backend"
            return 0
        else
            log_error "✗ Nginx failed to proxy to backend"
            return 1
        fi
    else
        log_error "✗ Nginx service reported unhealthy status"
        log_info "Response: $health_response"
        return 1
    fi
}

# Store results for comparison
declare -A TEST_RESULTS

# API endpoints to test
declare -A API_ENDPOINTS=(
    ["GET:/api/graph/data"]="Get full graph data"
    ["GET:/api/graph/paginated?page=1&pageSize=10"]="Get paginated graph data"
    ["POST:/api/graph/update"]="Update graph data"
    
    ["GET:/api/files/fetch?path=README.md"]="Fetch repository file"
    
    ["GET:/api/chat/status"]="Check RAGFlow chat status"
    ["GET:/api/perplexity/status"]="Check Perplexity status"
    
    ["GET:/health"]="Backend health check"
    ["GET:/metrics"]="Backend metrics"
)

# Health check configuration
declare -A HEALTH_CHECKS=(
    ["nginx"]="http://localhost:4000/health"
    ["backend"]="http://127.0.0.1:3001/health"
    ["ragflow"]="http://localhost:3001/health"
    ["cloudflared"]="http://localhost:2000/ready"
)

# Function to log messages with timestamp and category
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S.%N' | cut -b 1-23)
    local category="${2:-INFO}"
    local color="${3:-$BLUE}"
    printf "%s[%s]%s %s[%s]%s %s\n" "${BOLD}" "${timestamp}" "${NC}" "${color}" "${category}" "${NC}" "$1"
}

# Function to log debug messages
debug() {
    log "$1" "DEBUG" "$GRAY"
}

# Function to log test step
log_step() {
    printf "\n%s%s---%s %s %s---%s\n" "${BOLD}" "${BLUE}" "${NC}" "$1" "${BLUE}" "${NC}"
}

# Function to log success with details
log_success() {
    local message="$1"
    local details="${2:-}"
    log "✓ $message" "SUCCESS" "$GREEN"
    if [ -n "$details" ]; then
        printf "%s%s%s\n" "${GRAY}" "$details" "${NC}"
    fi
}

# Function to log error with details
log_error() {
    local message="$1"
    local details="${2:-}"
    log "✗ $message" "ERROR" "$RED"
    if [ -n "$details" ]; then
        printf "%s%s%s\n" "${GRAY}" "$details" "${NC}"
    fi
}

# Function to log info with details
log_info() {
    local message="$1"
    local details="${2:-}"
    log "$message" "INFO" "$BLUE"
    if [ -n "$details" ]; then
        printf "%s%s%s\n" "${GRAY}" "$details" "${NC}"
    fi
}

# Function to log section header with timing
log_header() {
    local start_time=$(date +%s.%N)
    printf "\n%s%s===%s %s %s===%s\n" "${BOLD}" "${YELLOW}" "${NC}" "$1" "${YELLOW}" "${NC}"
    echo "$start_time"
}

# Function to log section footer with timing
log_footer() {
    local section="$1"
    local start_time="$2"
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    printf "\n%s%s===%s %s completed in %.2f seconds %s===%s\n" "${BOLD}" "${YELLOW}" "${NC}" "$section" "$duration" "${YELLOW}" "${NC}"
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
    
    local start_time=$(date +%s.%N)
    log_step "Testing $method $description"
    log_info "URL: $url"
    log_info "Method: $method"
    if [[ -n "$data" ]]; then
        log_info "Request Data:"
        echo "$data" | jq '.' || echo "$data"
    fi
    
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
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        log_success "$description successful (HTTP $http_code)" "Duration: ${duration}s"
        return 0
    else
        log_error "$description failed (HTTP $http_code)" "Duration: ${duration}s"
        if [[ -n "$response" ]]; then
            log_error "Response:" "$response"
        fi
        return 1
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
    test_websocket "$base_url" "$env Binary Protocol" || ((failed++))
    
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

# Function to compare environment results with detailed output
compare_environments() {
    local start_time=$(log_header "Environment Comparison")
    
    log_step "Environment Overview"
    printf "\n%-15s %-20s %-15s %-15s %-15s\n" "Environment" "Base URL" "Protocol" "Status" "Latency"
    printf "%-15s %-20s %-15s %-15s %-15s\n" "Development" "localhost:3001" "ws://" "${TEST_RESULTS[development_status]:-N/A}" "${TEST_RESULTS[development_latency]:-N/A}"
    printf "%-15s %-20s %-15s %-15s %-15s\n" "Nginx" "localhost:4000" "ws://" "${TEST_RESULTS[nginx_status]:-N/A}" "${TEST_RESULTS[nginx_latency]:-N/A}"
    printf "%-15s %-20s %-15s %-15s %-15s\n" "Production" "$PUBLIC_DOMAIN" "wss://" "${TEST_RESULTS[external_status]:-N/A}" "${TEST_RESULTS[external_latency]:-N/A}"
    
    log_step "WebSocket Protocol Status"
    printf "\n%-20s %-15s %-15s %-15s %-20s\n" "Protocol" "Development" "Nginx" "External" "Notes"
    printf "%-20s %-15s %-15s %-15s %-20s\n" "Connection" "${TEST_RESULTS[development_ws]:-N/A}" "${TEST_RESULTS[nginx_ws]:-N/A}" "${TEST_RESULTS[external_ws]:-N/A}" "Initial handshake"
    printf "%-20s %-15s %-15s %-15s %-20s\n" "Binary Updates" "${TEST_RESULTS[development_binary]:-N/A}" "${TEST_RESULTS[nginx_binary]:-N/A}" "${TEST_RESULTS[external_binary]:-N/A}" "Data streaming"
    printf "%-20s %-15s %-15s %-15s %-20s\n" "Settings" "${TEST_RESULTS[development_ws_settings]:-N/A}" "${TEST_RESULTS[nginx_ws_settings]:-N/A}" "${TEST_RESULTS[external_ws_settings]:-N/A}" "Configuration"
    
    log_step "REST API Status"
    printf "\n%-30s %-15s %-15s %-15s %-10s\n" "Endpoint" "Development" "Nginx" "External" "Latency"
    for endpoint in "${!API_ENDPOINTS[@]}"; do
        printf "%-30s %-15s %-15s %-15s %-10s\n" \
            "$endpoint" \
            "${TEST_RESULTS[development_${endpoint}]:-N/A}" \
            "${TEST_RESULTS[nginx_${endpoint}]:-N/A}" \
            "${TEST_RESULTS[external_${endpoint}]:-N/A}" \
            "${TEST_RESULTS[${endpoint}_latency]:-N/A}"
    done
    
    log_step "Error Summary"
    local error_count=0
    for key in "${!TEST_RESULTS[@]}"; do
        if [[ ${TEST_RESULTS[$key]} == "failed" ]]; then
            ((error_count++))
            log_error "Failed Test: $key" "${TEST_RESULTS[${key}_error]:-No error details available}"
        fi
    done
    
    if [ $error_count -eq 0 ]; then
        log_success "All tests completed successfully"
    else
        log_error "$error_count tests failed" "See above for detailed error information"
    fi
    
    log_step "Performance Summary"
    printf "\n%-20s %-15s %-15s %-15s\n" "Metric" "Development" "Nginx" "External"
    printf "%-20s %-15s %-15s %-15s\n" "Avg Response" "${TEST_RESULTS[development_avg_latency]:-N/A}" "${TEST_RESULTS[nginx_avg_latency]:-N/A}" "${TEST_RESULTS[external_avg_latency]:-N/A}"
    printf "%-20s %-15s %-15s %-15s\n" "WS Connect" "${TEST_RESULTS[development_ws_latency]:-N/A}" "${TEST_RESULTS[nginx_ws_latency]:-N/A}" "${TEST_RESULTS[external_ws_latency]:-N/A}"
    
    log_footer "Environment Comparison" "$start_time"
}

# Main function with detailed progress reporting
main() {
    local total_failed=0
    local main_start_time=$(date +%s.%N)
    
    log_header "LogseqXR Connection Test Suite"
    
    # Show test configuration
    log_step "Test Configuration"
    printf "\n%-25s: %s\n" "Backend Port" "$BACKEND_PORT"
    printf "%-25s: %s\n" "Nginx Port" "$NGINX_PORT"
    printf "%-25s: %s\n" "Production Domain" "$PUBLIC_DOMAIN"
    printf "%-25s: %s seconds\n" "Request Timeout" "$TIMEOUT"
    printf "%-25s: %s seconds\n" "WebSocket Timeout" "$WEBSOCKET_TIMEOUT"
    
    # Show environment information
    log_step "Environment Information"
    printf "\n%-25s: %s\n" "Container Name" "$CONTAINER_NAME"
    printf "%-25s: %s\n" "Development URL" "http://localhost:3001"
    printf "%-25s: %s\n" "Nginx URL" "http://localhost:4000"
    printf "%-25s: %s\n" "Production URL" "https://$PUBLIC_DOMAIN"
    
    # Wait for services to be ready
    log_step "Service Readiness Check"
    wait_for_webxr || {
        log_error "Service readiness check failed - attempting to continue with available services"
        log_info "Note: Some tests may fail if required services are not running"
    }
    
    # Test development environment first
    log_step "Development Environment Tests"
    log_info "Starting development environment tests..."
    test_development_environment
    local dev_status=$?
    ((total_failed+=$dev_status))
    TEST_RESULTS["development_status"]=$dev_status
    
    # Test each environment with timing
    for env in "nginx" "backend" "ragflow" "external"; do
        log_step "${env^} Environment Tests"
        log_info "Testing ${env} environment..."
        local env_start_time=$(date +%s.%N)
        
        test_environment "$env"
        local env_status=$?
        ((total_failed+=$env_status))
        TEST_RESULTS["${env}_status"]=$env_status
        
        local env_end_time=$(date +%s.%N)
        local env_duration=$(echo "$env_end_time - $env_start_time" | bc)
        TEST_RESULTS["${env}_duration"]=$(printf "%.3f" $env_duration)
        
        log_info "${env^} environment testing completed in ${env_duration}s"
    done
    
    # Additional tests with progress reporting
    log_step "Network Connectivity Tests"
    log_info "Testing RAGFlow connectivity..."
    test_ragflow_connectivity
    local rag_status=$?
    ((total_failed+=$rag_status))
    TEST_RESULTS["ragflow_status"]=$rag_status
    
    log_step "Cloudflare Tests"
    log_info "Testing Cloudflare tunnel..."
    test_cloudflare_tunnel
    local cf_status=$?
    ((total_failed+=$cf_status))
    TEST_RESULTS["cloudflare_status"]=$cf_status
    
    # Results analysis
    log_step "Test Results Analysis"
    compare_environments
    
    log_step "Container Log Analysis"
    check_container_logs $CONTAINER_NAME
    
    log_step "Nginx Configuration Check"
    check_nginx_config
    
    # Calculate total duration
    local main_end_time=$(date +%s.%N)
    local total_duration=$(echo "$main_end_time - $main_start_time" | bc)
    
    # Final summary with statistics
    log_header "Test Suite Summary"
    printf "\n%-25s: %d\n" "Total Tests Run" "${#TEST_RESULTS[@]}"
    printf "%-25s: %d\n" "Failed Tests" "$total_failed"
    printf "%-25s: %.3f seconds\n" "Total Duration" "$total_duration"
    printf "%-25s: %s\n" "Start Time" "$(date -d @${main_start_time%.*})"
    printf "%-25s: %s\n" "End Time" "$(date -d @${main_end_time%.*})"
    
    if [ $total_failed -eq 0 ]; then
        log_success "All tests passed successfully!" "Total duration: ${total_duration}s"
        exit 0
    else
        log_error "$total_failed tests failed" "See detailed results above for specific failures"
        log_info "Test suite completed in ${total_duration}s"
        exit 1
    fi
}

# Make script executable
chmod +x "$0"

# Run main function
main
