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
    ["internal"]="http://localhost:$BACKEND_PORT"
    ["docker"]="http://localhost:$NGINX_PORT"
    ["production"]="https://$PUBLIC_DOMAIN"
)

# REST endpoints to test
declare -a REST_ENDPOINTS=(
    # Graph endpoints
    "/api/graph/data"
    "/api/graph/data/paginated?page=0&page_size=10"
    
    # Visualization Settings Categories
    "/api/visualization/settings/animations"
    "/api/visualization/settings/ar"
    "/api/visualization/settings/audio"
    "/api/visualization/settings/bloom"
    "/api/visualization/settings/edges"
    "/api/visualization/settings/hologram"
    "/api/visualization/settings/labels"
    "/api/visualization/settings/nodes"
    "/api/visualization/settings/physics"
    "/api/visualization/settings/rendering"
    
    # System Settings
    "/api/visualization/settings/network"
    "/api/visualization/settings/websocket"
    "/api/visualization/settings/security"
    "/api/visualization/settings/client-debug"
    "/api/visualization/settings/server-debug"
    
    # Individual Settings Tests (examples for each category)
    "/api/visualization/settings/animations/enabled"
    "/api/visualization/settings/ar/enabled"
    "/api/visualization/settings/audio/enabled"
    "/api/visualization/settings/bloom/enabled"
    "/api/visualization/settings/edges/enabled"
    "/api/visualization/settings/hologram/enabled"
    "/api/visualization/settings/labels/enabled"
    "/api/visualization/settings/nodes/enabled"
    "/api/visualization/settings/physics/enabled"
    "/api/visualization/settings/rendering/enabled"
    "/api/visualization/settings/network/enabled"
    "/api/visualization/settings/websocket/update-rate"
    "/api/visualization/settings/security/enabled"
    "/api/visualization/settings/client-debug/enabled"
    "/api/visualization/settings/server-debug/enabled"
    
    # Other API endpoints
    "/api/files/fetch"
    "/api/chat/stream"
    "/api/perplexity"
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

# Function to test WebSocket connection
test_websocket() {
    local url="$1"
    local description="$2"
    
    log_info "Testing WebSocket: $description"
    log_info "URL: $url"
    
    # Use websocat to test WebSocket connection if available
    if command -v websocat >/dev/null 2>&1; then
        timeout $WEBSOCKET_TIMEOUT websocat --no-close "$url" 2>&1 || {
            log_error "WebSocket connection failed"
            return 1
        }
        log_success "WebSocket connection successful"
        return 0
    else
        # Fallback to curl for basic connection test
        if curl --include \
            --no-buffer \
            --header "Connection: Upgrade" \
            --header "Upgrade: websocket" \
            --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
            --header "Sec-WebSocket-Version: 13" \
            -s "$url" 2>&1 | grep -q "101 Switching Protocols"; then
            log_success "WebSocket handshake successful"
            return 0
        else
            log_error "WebSocket handshake failed"
            return 1
        fi
    fi
}

# Function to test REST endpoint
test_rest_endpoint() {
    local url="$1"
    local description="$2"
    local method="${3:-GET}"
    local data="${4:-}"
    local extra_opts="${5:-}"
    
    log_info "Testing $method $description..."
    log_info "URL: $url"
    
    local curl_opts="-X $method -m $TIMEOUT -s -w '%{http_code}'"
    [[ "$url" == https://* ]] && curl_opts="$curl_opts -k"
    [[ -n "$data" ]] && curl_opts="$curl_opts -H 'Content-Type: application/json' -d '$data'"
    [[ -n "$extra_opts" ]] && curl_opts="$curl_opts $extra_opts"
    
    local response
    local http_code
    
    # Execute curl command
    eval "response=\$(curl $curl_opts '$url' 2>&1)"
    http_code=${response: -3}
    response=${response:0:${#response}-3}
    
    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        log_success "$description successful (HTTP $http_code)"
        log_info "Response: $response"
        return 0
    else
        log_error "$description failed (HTTP $http_code)"
        log_info "Response: $response"
        return 1
    fi
}

# Function to test all endpoints for a given environment
test_environment() {
    local env="$1"
    local base_url="${ENDPOINTS[$env]}"
    local failed=0
    
    log_header "Testing $env environment ($base_url)"
    
    # Test REST endpoints
    for endpoint in "${REST_ENDPOINTS[@]}"; do
        test_rest_endpoint "$base_url$endpoint" "$env $endpoint" || ((failed++))
        
        # Test PUT for settings endpoints
        if [[ "$endpoint" == "/api/visualization/settings/"* ]]; then
            test_rest_endpoint "$base_url$endpoint" "$env $endpoint (PUT)" "PUT" '{"value": 30}' || ((failed++))
        fi
    done
    
    # Test WebSocket endpoints
    local ws_protocol="ws"
    [[ "$base_url" == https://* ]] && ws_protocol="wss"
    local ws_base_url="${base_url/http:/$ws_protocol:}"
    local ws_base_url="${ws_base_url/https:/$ws_protocol:}"
    
    test_websocket "$ws_base_url/wss" "$env WebSocket Binary Protocol" || ((failed++))
    
    return $failed
}

# Main execution
main() {
    log_header "Starting comprehensive endpoint tests across all environments..."
    local total_failed=0
    
    # Test each environment
    for env in "${!ENDPOINTS[@]}"; do
        test_environment "$env"
        local env_failed=$?
        ((total_failed += env_failed))
        
        # Print environment summary
        echo
        log_info "Environment $env: $([ $env_failed -eq 0 ] && echo "PASS" || echo "FAIL ($env_failed failed)")"
    done
    
    # Print final summary
    echo
    log_header "Test Summary:"
    if [ $total_failed -eq 0 ]; then
        log_success "All tests passed successfully!"
        exit 0
    else
        log_error "${total_failed} tests failed"
        exit 1
    fi
}

# Make script executable
chmod +x "$0"

# Run main function
main
