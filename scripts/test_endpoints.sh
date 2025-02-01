#!/bin/bash

# Enable error reporting
set -e

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Symbols
CHECK_MARK="✓"
CROSS_MARK="✗"
ARROW="→"
BULLET="•"

# Configuration
BACKEND_PORT=3001
NGINX_PORT=4000
CONTAINER_NAME="logseq-xr-webxr"
PUBLIC_DOMAIN="www.visionflow.info"
RAGFLOW_NETWORK="docker_ragflow"
VERBOSE=true
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT=5

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true ;;
        --settings-only) TEST_SETTINGS_ONLY=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Logging functions
log() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${CYAN}${BULLET}${NC} $1" | tee -a "$LOG_FILE"
    fi
}

log_error() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}${CROSS_MARK} ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}${CHECK_MARK}${NC} $1" | tee -a "$LOG_FILE"
}

log_message() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n" | tee -a "$LOG_FILE"
}

# Function to check graph endpoints
check_graph_endpoints() {
    log_section "Checking Graph Endpoints"
    
    # Test graph endpoints with minimal output
    log_message "Testing graph endpoints..."
    # Test basic graph data
    response=$(docker exec ${CONTAINER_NAME} curl -s \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/graph/data")
    nodes_count=$(echo "$response" | jq -r '.nodes | length')
    edges_count=$(echo "$response" | jq -r '.edges | length')
    log_message "Graph data: ${nodes_count} nodes, ${edges_count} edges"

    # Test pagination
    response=$(docker exec ${CONTAINER_NAME} curl -s \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/graph/data/paginated?page=1&pageSize=100")
    page_count=$(echo "$response" | jq -r '.nodes | length')
    log_message "Paginated data: ${page_count} nodes in first page"
}

# Function to check settings endpoint
check_settings_endpoint() {
    log_section "Checking Settings Endpoint"
    
    # Check critical service status
    log_message "Checking service status..."
    docker exec ${CONTAINER_NAME} bash -c '
        echo -e "=== Service Status ==="
        # Show only running processes and ports
        ps aux | grep "[w]ebxr" | head -n 2
        netstat -tulpn | grep -E ":(3001|4000)" | head -n 2
        
        # Show only recent errors if any
        echo -e "\n=== Recent Errors ==="
        grep "error" /var/log/nginx/error.log 2>/dev/null | tail -n 2 || echo "No recent errors"
        
        # Show settings file status
        echo -e "\n=== Settings ==="
        ls -l /app/settings.yaml
    ' | tee -a "$LOG_FILE"
    
    # Test settings endpoint through nginx (concise)
    log_message "Testing settings endpoint through nginx..."
    response=$(docker exec ${CONTAINER_NAME} curl -s \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/settings")
    
    # Extract and show key settings
    tunnel_id=$(echo "$response" | jq -r '.system.network.tunnelId')
    domain=$(echo "$response" | jq -r '.system.network.domain')
    port=$(echo "$response" | jq -r '.system.network.port')
    log_message "Settings: domain=${domain}, port=${port}, tunnel_id=${tunnel_id}"
}

# Function to check WebSocket Endpoint with verbose output for detailed diagnostics
check_websocket_endpoint() {
    log_section "Checking WebSocket Endpoint"
    log_message "Attempting WebSocket upgrade test with verbose output..."
    ws_response=$(docker exec ${CONTAINER_NAME} curl -v -i -N -s \
        -H "Connection: Upgrade" \
        -H "Upgrade: websocket" \
        -H "Sec-WebSocket-Version: 13" \
        -H "Sec-WebSocket-Key: testkey" \
        "http://localhost:${NGINX_PORT}/wss" || true)
    log_message "WebSocket test response (verbose):"
    if [ -z "$ws_response" ]; then
        log_error "No response received from WebSocket upgrade test."
    else
        echo "$ws_response" | tee -a "$LOG_FILE"
        log_message "WebSocket response length: $(echo "$ws_response" | wc -c) characters"
    fi
}

# Function to test GitHub API endpoints
check_github_endpoints() {
    log_section "Testing GitHub API Endpoints"
    
    # Load GitHub credentials from .env
    if [ -f ../.env ]; then
        source ../.env
    else
        log_error ".env file not found"
        return 1
    fi
    
    # Test a single representative file
    local test_files=(
        "p(doom).md"  # Representative markdown file
    )
    
    log_message "Testing GitHub API access..."
    
    for file in "${test_files[@]}"; do
        
        # Test with base path only (skip raw path test)
        path="${GITHUB_BASE_PATH}/${file}"
        encoded_path=$(echo -n "${path}" | jq -sRr @uri)
        
        # Check commits and contents in one request
        response=$(curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" \
                       -H "Accept: application/vnd.github+json" \
                       "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/commits?path=${encoded_path}")
        count=$(echo "$response" | jq -r '. | length')
        
        content_response=$(curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" \
                               -H "Accept: application/vnd.github+json" \
                               "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encoded_path}")
        exists=$(echo "$content_response" | jq -e 'has("message")' > /dev/null && echo "no" || echo "yes")
        
        # Single line status for file
        log_message "File: ${file} (${count} commits, exists: ${exists})"
        
        # Rate limit delay
        sleep 1
    done
}

# Main execution
main() {
    log "${YELLOW}Starting endpoint diagnostics...${NC}"
    
    # Check container status
    if ! docker ps -q -f name=${CONTAINER_NAME} > /dev/null 2>&1; then
        log_error "Container ${YELLOW}${CONTAINER_NAME}${NC} is not running"
        exit 1
    fi
    
    # Check network connectivity (concise)
    log_message "Checking network connectivity..."
    docker exec ${CONTAINER_NAME} bash -c '
        echo -e "=== Network Status ==="
        # Show only main interface
        ip addr show | grep -A2 "eth0" | head -n3
        # Show default route
        ip route | grep default
        # Show DNS servers
        grep "nameserver" /etc/resolv.conf | head -n2
    ' | tee -a "$LOG_FILE"
    
    # Check settings endpoint
    check_settings_endpoint
    
    # Check graph endpoints
    check_graph_endpoints
    
    # Check GitHub API endpoints
    check_github_endpoints
    
    log "${YELLOW}Diagnostics completed${NC}"
}

# Run main function
main