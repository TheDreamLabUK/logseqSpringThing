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
    
    # Test graph data endpoint
    log_message "Testing graph data endpoint..."
    docker exec ${CONTAINER_NAME} curl -v \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/graph/data" 2>&1 | tee -a "$LOG_FILE"
    
    # Test paginated graph data endpoint
    log_message "Testing paginated graph data endpoint..."
    docker exec ${CONTAINER_NAME} curl -v \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/graph/data/paginated?page=1&pageSize=100" 2>&1 | tee -a "$LOG_FILE"
}

# Function to check settings endpoint
check_settings_endpoint() {
    log_section "Checking Settings Endpoint"
    
    # Check nginx configuration
    log_message "Checking nginx configuration..."
    docker exec ${CONTAINER_NAME} nginx -T | grep -A 10 "location /api" | tee -a "$LOG_FILE"
    
    # Check recent nginx access logs for settings requests
    log_message "Checking nginx access logs for settings requests..."
    docker exec ${CONTAINER_NAME} bash -c '
        echo -e "\n=== Nginx Access Logs (Settings Requests) ==="
        grep "/api/settings" /var/log/nginx/access.log | tail -n 20
        
        echo -e "\n=== Nginx Error Logs ==="
        tail -n 50 /var/log/nginx/error.log
        
        echo -e "\n=== Backend Process Status ==="
        ps aux | grep "[w]ebxr"
        
        echo -e "\n=== Network Status ==="
        netstat -tulpn | grep -E ":(3001|4000)"
        
        echo -e "\n=== Settings File Status ==="
        ls -l /app/settings.yaml
        
        echo -e "\n=== Backend Logs ==="
        if [ -f /tmp/webxr.log ]; then
            grep -i "settings\|error\|GET /api\|POST /api" /tmp/webxr.log | tail -n 50
        fi
        
        echo -e "\n=== Route Registration ==="
        if [ -f /tmp/webxr.log ]; then
            grep -i "route.*settings" /tmp/webxr.log
        fi
    ' | tee -a "$LOG_FILE"
    
    # Test settings endpoint through nginx
    log_message "Testing settings endpoint through nginx..."
    docker exec ${CONTAINER_NAME} curl -v \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/settings" 2>&1 | tee -a "$LOG_FILE"
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
    
    # Test files with different path patterns
    local test_files=(
        "Two Heads Are Better Than One.md"
        "p(doom).md"
        "chatgpt__2024-08-20 09:48:00.md"
    )
    
    for file in "${test_files[@]}"; do
        log_message "Testing commits for file: $file"
        
        # Test with and without base path
        paths=(
            "${file}"
            "${GITHUB_BASE_PATH}/${file}"
        )
        
        for path in "${paths[@]}"; do
            # URL encode the path
            encoded_path=$(echo -n "${path}" | jq -sRr @uri)
            
            # Test the commits endpoint
            log_message "Testing commits API with path: ${path}"
            log_message "GET /repos/${GITHUB_OWNER}/${GITHUB_REPO}/commits?path=${encoded_path}"
            
            response=$(curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" \
                           -H "Accept: application/vnd.github+json" \
                           "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/commits?path=${encoded_path}")
            
            # Log full response in verbose mode
            if [ "$VERBOSE" = true ]; then
                log_verbose "Full response:"
                echo "$response" | jq '.' | tee -a "$LOG_FILE"
            fi
            
            # Check commit count
            count=$(echo "$response" | jq -r '. | length')
            if [ "$count" = "0" ]; then
                log_error "No commits found for path: ${path}"
            else
                log_success "Found ${count} commits for path: ${path}"
            fi
            
            # Test contents endpoint as well
            log_message "Testing contents API with path: ${path}"
            log_message "GET /repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encoded_path}"
            
            content_response=$(curl -s -H "Authorization: Bearer ${GITHUB_TOKEN}" \
                                  -H "Accept: application/vnd.github+json" \
                                  "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encoded_path}")
            
            # Log full response in verbose mode
            if [ "$VERBOSE" = true ]; then
                log_verbose "Full contents response:"
                echo "$content_response" | jq '.' | tee -a "$LOG_FILE"
            fi
            
            # Check if file exists
            if echo "$content_response" | jq -e 'has("message")' > /dev/null; then
                log_error "File not found at path: ${path}"
            else
                log_success "File exists at path: ${path}"
            fi
            
            # Add a small delay to respect rate limits
            sleep 1
        done
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
    
    # Check network connectivity
    log_message "Checking network connectivity..."
    docker exec ${CONTAINER_NAME} bash -c '
        echo -e "\n=== Network Configuration ==="
        ip addr show
        
        echo -e "\n=== Network Routes ==="
        ip route
        
        echo -e "\n=== DNS Resolution ==="
        cat /etc/resolv.conf
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
