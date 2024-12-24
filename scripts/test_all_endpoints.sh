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
    local show_raw="${6:-false}"
    
    log_info "Testing $method $description..."
    log_info "URL: $url"
    
    local curl_opts="-X $method -m $TIMEOUT -s -w '\n%{http_code}'"
    [[ "$url" == https://* ]] && curl_opts="$curl_opts -k"
    [[ -n "$data" ]] && curl_opts="$curl_opts -H 'Content-Type: application/json' -d '$data'"
    [[ -n "$extra_opts" ]] && curl_opts="$curl_opts $extra_opts"
    
    # Execute curl command
    local response
    local http_code
    eval "response=\$(curl $curl_opts '$url' 2>&1)"
    http_code=$(echo "$response" | tail -n1)
    response=$(echo "$response" | sed \$d)  # Remove the last line (status code)
    
    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        log_success "$description successful (HTTP $http_code)"
        if [ "$show_raw" = true ]; then
            log_info "Raw Response:"
            printf "%s\n" "$response"
            log_info "Pretty Response:"
            pretty_json "$response"
        else
            log_info "Response Summary:"
            pretty_json "$response" | head -n 20
            local lines=$(echo "$response" | wc -l)
            if [ "$lines" -gt 20 ]; then
                log_info "... (${lines} lines total, showing first 20)"
            fi
        fi
        return 0
    else
        log_error "$description failed (HTTP $http_code)"
        log_info "Response: $response"
        return 1
    fi
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

# Function to test environment
test_environment() {
    local env="$1"
    local base_url="${ENDPOINTS[$env]}"
    local failed=0
    
    log_header "Testing $env environment ($base_url)"
    
    # Test paginated endpoint with raw output
    test_endpoint "$base_url/api/graph/data/paginated?page=0&page_size=10" "$env paginated graph data" "GET" "" "" true || ((failed++))
    
    # Test other endpoints
    test_endpoint "$base_url/api/graph/data" "$env graph data" || ((failed++))
    
    # Test settings endpoints
    test_endpoint "$base_url/api/settings/visualization" "$env visualization settings" || ((failed++))
    test_endpoint "$base_url/api/settings/visualization/animations/enable_motion_blur" "$env enable motion blur setting" || ((failed++))
    test_endpoint "$base_url/api/settings/visualization/bloom/enabled" "$env bloom enabled setting" || ((failed++))
    
    # Test updating a setting
    test_endpoint "$base_url/api/settings/visualization/animations/enable_motion_blur" "$env update enable motion blur setting" "PUT" '{"value": true}' || ((failed++))
    
    # Test saving settings
    test_endpoint "$base_url/api/settings/save" "$env save settings" "PUT" || ((failed++))
    
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
    log_header "Starting endpoint tests..."
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
