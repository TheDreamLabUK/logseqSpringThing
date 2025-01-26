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
        ls -l /app/settings.toml
        
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

# Main execution
main() {
    log "${YELLOW}Starting settings endpoint diagnostics...${NC}"
    
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
    
    log "${YELLOW}Diagnostics completed${NC}"
}

# Run main function
main
