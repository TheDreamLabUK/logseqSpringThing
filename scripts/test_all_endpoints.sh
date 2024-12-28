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
    
    # Only capture status code, discard response body
    local http_code
    if [[ -n "$data" ]]; then
        http_code=$(curl -X $method -m $TIMEOUT -s -o /dev/null -w "%{http_code}" \
            -H "Accept: application/json" -H "Content-Type: application/json" \
            -d "$data" $extra_opts "$url")
    else
        http_code=$(curl -X $method -m $TIMEOUT -s -o /dev/null -w "%{http_code}" \
            -H "Accept: application/json" $extra_opts "$url")
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
    
    # Test graph endpoints
    test_endpoint "$base_url/api/graph/data" "$env full graph data" || ((failed++))
    test_endpoint "$base_url/api/graph/paginated?page=0&pageSize=10" "$env paginated graph data" || ((failed++))
    test_endpoint "$base_url/api/graph/update" "$env graph update" "POST" '{"nodes": [], "edges": []}' || ((failed++))
    
    # Test settings endpoints
    test_endpoint "$base_url/api/settings" "$env all settings" || ((failed++))
    test_endpoint "$base_url/api/settings/visualization" "$env visualization settings" || ((failed++))
    test_endpoint "$base_url/api/settings/websocket" "$env WebSocket control API" || ((failed++))
    
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
    
    # Test graph data endpoint
    log_info "Testing graph data endpoint..."
    local status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
        -H "Accept: application/json" "http://localhost:4000/api/graph/data")
    if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
        log_success "Container internal graph data endpoint successful (HTTP $status)"
    else
        log_error "Container internal graph data endpoint failed (HTTP $status)"
        ((failed++))
    fi
    
    # Test settings endpoint
    log_info "Testing settings endpoint..."
    status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
        -H "Accept: application/json" "http://localhost:4000/api/settings")
    if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
        log_success "Container internal settings endpoint successful (HTTP $status)"
    else
        log_error "Container internal settings endpoint failed (HTTP $status)"
        ((failed++))
    fi
    
    # Test WebSocket inside container
    log_info "Testing WebSocket endpoint..."
    status=$(docker exec $CONTAINER_NAME curl -v -i \
        --no-buffer \
        -H "Accept: application/json" \
        -H "Connection: Upgrade" \
        -H "Upgrade: websocket" \
        -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
        -H "Sec-WebSocket-Version: 13" \
        "http://localhost:4000/wss" 2>&1 | grep -oP '(?<=HTTP/1.1 )\d{3}')
    if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
        log_success "Container internal WebSocket endpoint successful (HTTP $status)"
    else
        log_error "Container internal WebSocket endpoint failed (HTTP $status)"
        ((failed++))
    fi
    
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
    
    # Test graph data endpoint
    local status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
        -H "Accept: application/json" "http://localhost:3001/api/graph/data")
    if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
        log_success "Backend graph data endpoint successful (HTTP $status)"
    else
        log_error "Backend graph data endpoint failed (HTTP $status)"
        ((failed++))
    fi
    
    # Test settings endpoints with and without trailing slash
    local settings_endpoints=(
        "/api/settings"
        "/api/settings/"
        "/api/settings/visualization"
        "/api/settings/websocket"
        "/api/settings/system"
        "/api/settings/all"
    )
    
    for endpoint in "${settings_endpoints[@]}"; do
        log_info "Testing $endpoint..."
        status=$(docker exec $CONTAINER_NAME curl -s -o /dev/null -w "%{http_code}" \
            -H "Accept: application/json" "http://localhost:3001$endpoint")
        if [[ "$status" =~ ^2[0-9][0-9]$ ]]; then
            log_success "Backend $endpoint successful (HTTP $status)"
        else
            log_error "Backend $endpoint failed (HTTP $status)"
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
    
    # Check if cloudflared container is running
    if ! docker ps -q -f name="^/cloudflared-tunnel$" > /dev/null 2>&1; then
        log_error "Cloudflared tunnel container is not running"
        return 1
    }
    log_success "Cloudflared tunnel container is running"

    # Get tunnel URL from cloudflared logs
    local tunnel_url=$(docker logs cloudflared-tunnel 2>&1 | grep -o 'https://[^ ]*\.trycloudflare\.com' | tail -n1)
    if [ -z "$tunnel_url" ]; then
        log_error "Could not find tunnel URL in cloudflared logs"
        return 1
    }
    log_success "Found tunnel URL: $tunnel_url"

    # Test tunnel endpoint
    local response=$(curl -s -o /dev/null -w "%{http_code}" "$tunnel_url" || echo "000")
    if [ "$response" = "200" ]; then
        log_success "Tunnel endpoint is accessible"
    else
        log_error "Tunnel endpoint returned HTTP $response"
        return 1
    }
}

# Main execution
main() {
    local start_time=$(date +%s)
    log_header "Starting endpoint tests"

    # Wait for WebXR to be ready
    wait_for_webxr || {
        log_error "WebXR container is not ready"
        exit 1
    }

    # Test Cloudflare tunnel first
    test_cloudflare_tunnel || warn "Cloudflare tunnel test failed"

    # Test each environment
    for env in "${!ENDPOINTS[@]}"; do
        test_environment "$env" || warn "Tests for $env environment failed"
    done

    # Test backend directly
    test_backend_directly || warn "Backend tests failed"

    # Test container internal endpoints
    test_container_endpoints || warn "Container internal endpoint tests failed"

    # Test RAGFlow network connectivity
    test_ragflow_connectivity || warn "RAGFlow network connectivity tests failed"

    # If any tests failed, show logs again
    if [ $? -ne 0 ]; then
        log_header "Test Failed - Showing Recent Logs"
        check_container_logs "$CONTAINER_NAME" 100
    fi

    # Final summary
    log_header "Test Summary"
    if [ $? -eq 0 ]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Make script executable
chmod +x "$0"

# Run main function
main
