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
VERBOSE=false
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT=5

# Add these arrays at the top with other configuration
declare -a WORKING_BACKEND_ENDPOINTS=()
declare -a WORKING_NGINX_ENDPOINTS=()
declare -a WORKING_PUBLIC_ENDPOINTS=()

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true ;;
        --health-only) TEST_HEALTH_ONLY=true ;;
        --backend-only) TEST_BACKEND_ONLY=true ;;
        --nginx-only) TEST_NGINX_ONLY=true ;;
        --network-only) TEST_NETWORK_ONLY=true ;;
        --public-only) TEST_PUBLIC_ONLY=true ;;
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

log_section() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n" | tee -a "$LOG_FILE"
}

# Function to safely execute docker commands with timeout
docker_exec() {
    timeout $TIMEOUT docker exec "$CONTAINER_NAME" $@ 2>&1 || echo "Command timed out after ${TIMEOUT}s"
}

# Function to get container IP
get_container_ip() {
    local container=$1
    docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container"
}

# Function to check if port is open
check_port() {
    local host="$1"
    local port="$2"
    timeout $TIMEOUT bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null
    return $?
}

# Function to diagnose REST endpoint failures
diagnose_endpoint() {
    local endpoint=$1
    local port=$2
    local container_ip=$3
    
    log_section "Detailed Diagnostics for ${YELLOW}${endpoint}${NC}"
    
    # Get process info and logs
    echo -e "${MAGENTA}${BOLD}Process and Log Analysis:${NC}" | tee -a "$LOG_FILE"
    docker exec ${CONTAINER_NAME} bash -c '
        # Get webxr process info
        pid=$(pgrep webxr)
        if [ ! -z "$pid" ]; then
            echo -e "\n'"${CYAN}${BOLD}"'=== Process Info ==='"${NC}"'"
            ps -fp $pid
            echo -e "\n'"${CYAN}${BOLD}"'=== Process Environment ==='"${NC}"'"
            cat /proc/$pid/environ | tr "\0" "\n"
            echo -e "\n'"${CYAN}${BOLD}"'=== Process Open Files ==='"${NC}"'"
            ls -l /proc/$pid/fd
        fi
        
        # Check webxr logs
        if [ -f /tmp/webxr.log ]; then
            echo -e "\n'"${CYAN}${BOLD}"'=== WebXR Log (/tmp/webxr.log) ==='"${NC}"'"
            tail -n 200 /tmp/webxr.log
            echo -e "\n'"${RED}${BOLD}"'=== Recent Errors in webxr.log ==='"${NC}"'"
            grep -i "error\|panic\|fatal" /tmp/webxr.log | tail -n 20
        fi
        
        # Check nginx logs
        echo -e "\n'"${CYAN}${BOLD}"'=== Nginx Error Log ==='"${NC}"'"
        if [ -f /var/log/nginx/error.log ]; then
            tail -n 100 /var/log/nginx/error.log
            echo -e "\n'"${RED}${BOLD}"'=== Recent Nginx Errors ==='"${NC}"'"
            grep -i "error\|warn\|notice" /var/log/nginx/error.log | tail -n 20
        fi' | tee -a "$LOG_FILE"
    
    # For graph endpoints, add specific diagnostics
    if [[ "${endpoint}" == *"graph"* ]]; then
        echo -e "\n${MAGENTA}${BOLD}Graph-Specific Diagnostics:${NC}" | tee -a "$LOG_FILE"
        docker exec ${CONTAINER_NAME} bash -c '
            echo -e "\n'"${CYAN}${BOLD}"'=== Graph Data Directory ==='"${NC}"'"
            ls -la /app/data/graph/
            echo -e "\n'"${CYAN}${BOLD}"'=== Graph Cache ==='"${NC}"'"
            ls -la /app/data/cache/
            echo -e "\n'"${CYAN}${BOLD}"'=== Memory Usage ==='"${NC}"'"
            free -h
            echo -e "\n'"${CYAN}${BOLD}"'=== Graph Settings ==='"${NC}"'"
            cat /app/settings.toml | grep -i "graph" || echo "No graph settings found"' | tee -a "$LOG_FILE"
    fi
    
    # Test endpoint directly with verbose output
    echo -e "\n${MAGENTA}${BOLD}Direct Endpoint Test:${NC}" | tee -a "$LOG_FILE"
    curl -v -H "Accept: application/json" "http://${container_ip}:${port}${endpoint}" 2>&1 | tee -a "$LOG_FILE"
}

# Function to check container health
check_container_health() {
    local container_ip=$(get_container_ip ${CONTAINER_NAME})
    
    log_section "Container Health Check"
    
    local health_failed=0
    
    # Check if container exists and is running
    if ! docker ps -q -f name=${CONTAINER_NAME} > /dev/null 2>&1; then
        log_error "Container ${YELLOW}${CONTAINER_NAME}${NC} is not running"
        docker ps -a -f name=${CONTAINER_NAME} --format "{{.Status}}" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # Check required processes
    log "Checking required processes..."
    if docker_exec pgrep webxr > /dev/null; then
        log_success "WebXR process is running"
    else
        log_error "WebXR process is not running"
        ((health_failed++))
    fi
    
    if docker_exec pgrep nginx > /dev/null; then
        log_success "Nginx process is running"
    else
        log_error "Nginx process is not running"
        ((health_failed++))
    fi
    
    # Check port accessibility
    log "Checking port accessibility..."
    if check_port "$container_ip" "$BACKEND_PORT"; then
        log_success "Backend port $BACKEND_PORT is accessible"
    else
        log_error "Backend port $BACKEND_PORT is not accessible"
        ((health_failed++))
    fi
    
    if check_port "$container_ip" "$NGINX_PORT"; then
        log_success "Nginx port $NGINX_PORT is accessible"
    else
        log_error "Nginx port $NGINX_PORT is not accessible"
        ((health_failed++))
    fi
    
    # Show process status
    echo -e "\n${CYAN}${BOLD}Running processes:${NC}" | tee -a "$LOG_FILE"
    docker_exec ps aux | grep -E "nginx|webxr|rust" | tee -a "$LOG_FILE" || true
    
    if [ $health_failed -eq 0 ]; then
        log_success "Container appears healthy"
        return 0
    else
        log_error "Container health check found $health_failed issues"
        return $health_failed
    fi
}

# Function to check static files
check_static_files() {
    log "${BLUE}Checking static files in container...${NC}"
    docker_exec ls -la /app/client || true
    docker_exec cat /app/client/index.html || true
}

# Function to test backend endpoints
test_backend() {
    log_section "Testing Backend Endpoints"
    local failed=0
    
    # Test backend endpoints
    local endpoints=(
        # Graph Data Endpoints
        "/api/graph/data:Retrieves the complete graph data including nodes, edges, and metadata"
        "/api/graph/data/paginated:Retrieves paginated graph data with configurable page size and filters"
        
        # File Management Endpoints
        "/api/files/fetch:Fetches and processes files from the repository"
        "/api/files/content:Retrieves content of specific files"
        "/api/files/refresh:Refreshes the graph data from current files"
        
        # Visualization Settings Endpoints
        "/api/visualization/settings/nodes:Gets/updates node appearance settings (color, size, etc.)"
        "/api/visualization/settings/edges:Gets/updates edge appearance settings (width, arrows, etc.)"
        "/api/visualization/settings/rendering:Gets/updates rendering settings (lighting, shadows, etc.)"
        "/api/visualization/settings/physics:Gets/updates physics simulation settings"
        "/api/visualization/settings/labels:Gets/updates label display settings"
        "/api/visualization/settings/bloom:Gets/updates bloom effect settings"
        "/api/visualization/settings/animations:Gets/updates animation settings"
        "/api/visualization/settings/audio:Gets/updates audio feedback settings"
        "/api/visualization/settings/clientDebug:Gets/updates client-side debug settings"
        "/api/visualization/settings/serverDebug:Gets/updates server-side debug settings"
        "/api/visualization/settings/security:Gets/updates security settings"
        "/api/visualization/settings/websocket:Gets/updates WebSocket connection settings"
        "/api/visualization/settings/network:Gets/updates network configuration settings"
        "/api/visualization/settings/default:Gets/updates default application settings"
        
        # RAGFlow Integration Endpoints
        "/api/ragflow/status:Checks RAGFlow service status"
        "/api/ragflow/query:Sends queries to the RAGFlow service"
        
        # Perplexity Integration Endpoints
        "/api/perplexity:Processes queries through Perplexity API for enhanced content understanding"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        log_verbose "Testing backend endpoint: $endpoint ($description)"
        local response=$(curl -s "http://localhost:$BACKEND_PORT$endpoint")
        if [ $? -eq 0 ] && [ -n "$response" ]; then
            log_success "Backend endpoint $endpoint accessible"
            log_verbose "Response: $response"
            WORKING_BACKEND_ENDPOINTS+=("$endpoint:$description")
        else
            log_error "Backend endpoint $endpoint failed"
            ((failed++))
        fi
    done
    
    return $failed
}

# Function to test nginx endpoints
test_nginx() {
    log_section "Testing Nginx Endpoints"
    local failed=0
    
    # Test nginx endpoints
    local endpoints=(
        # Static File Serving
        "/:Serves the main application interface"
        "/index.html:Serves the main HTML entry point"
        "/assets:Serves static assets (images, styles, scripts)"
        
        # API Proxying
        "/api/graph/data:Proxies graph data requests to backend"
        "/api/visualization/settings:Proxies visualization settings requests"
        "/api/files:Proxies file management requests"
        
        # WebSocket Endpoints
        "/wss:WebSocket endpoint for real-time updates"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        log_verbose "Testing nginx endpoint: $endpoint ($description)"
        local response=$(curl -s "http://localhost:$NGINX_PORT$endpoint")
        if [ $? -eq 0 ]; then
            log_success "Nginx endpoint $endpoint accessible"
            log_verbose "Response: $response"
            WORKING_NGINX_ENDPOINTS+=("$endpoint:$description")
        else
            log_error "Nginx endpoint $endpoint failed"
            ((failed++))
        fi
    done
    
    return $failed
}

# Function to test network
test_network() {
    log_section "Testing RAGFlow Network"
    local failed=0
    local container_ip=$(get_container_ip ${CONTAINER_NAME})
    
    # Test network connectivity
    if ! check_port "$container_ip" "$NGINX_PORT"; then
        log_error "Cannot connect to container on port $NGINX_PORT"
        ((failed++))
    else
        log_success "Container port $NGINX_PORT is accessible"
    fi
    
    # Test DNS resolution
    local dns_response=$(docker run --rm --network "$RAGFLOW_NETWORK" alpine nslookup webxr-client)
    if [ $? -eq 0 ]; then
        log_success "DNS resolution working"
        log_verbose "DNS Response: $dns_response"
    else
        log_error "DNS resolution failed"
        log_verbose "DNS Response: $dns_response"
        ((failed++))
    fi
    
    return $failed
}

# Function to test public URL
test_public() {
    log_section "Testing Public URL"
    local failed=0
    
    # Test HTTPS endpoints
    local endpoints=(
        # Main Application Access
        "/:Public access to main application interface"
        "/index.html:Public access to main HTML entry point"
        
        # Public API Access
        "/api/graph/data:Public access to graph data"
        "/api/settings:Public access to application settings"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        local response=$(curl -sk "https://$PUBLIC_DOMAIN$endpoint")
        if [ $? -eq 0 ] && [ -n "$response" ]; then
            log_success "Public endpoint $endpoint accessible"
            log_verbose "Response: $response"
            WORKING_PUBLIC_ENDPOINTS+=("$endpoint:$description")
        else
            log_error "Public endpoint $endpoint failed"
            ((failed++))
        fi
    done
    
    return $failed
}

# Add a small delay function
wait_between_tests() {
    sleep 2
    echo -e "\n"
}

# Add this function before main()
print_endpoint_summary() {
    log_section "Endpoint Summary"
    
    echo -e "\n${BLUE}${BOLD}Working Backend Endpoints:${NC}"
    for endpoint_info in "${WORKING_BACKEND_ENDPOINTS[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        echo -e "${GREEN}${CHECK_MARK}${NC} ${BOLD}$endpoint${NC}"
        echo -e "${GRAY}   $description${NC}"
    done
    
    echo -e "\n${BLUE}${BOLD}Working Nginx Endpoints:${NC}"
    for endpoint_info in "${WORKING_NGINX_ENDPOINTS[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        echo -e "${GREEN}${CHECK_MARK}${NC} ${BOLD}$endpoint${NC}"
        echo -e "${GRAY}   $description${NC}"
    done
    
    echo -e "\n${BLUE}${BOLD}Working Public Endpoints:${NC}"
    for endpoint_info in "${WORKING_PUBLIC_ENDPOINTS[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        echo -e "${GREEN}${CHECK_MARK}${NC} ${BOLD}$endpoint${NC}"
        echo -e "${GRAY}   $description${NC}"
    done
}

# Main execution
main() {
    log "${YELLOW}Starting comprehensive endpoint tests...${NC}"
    local total_failed=0
    
    # Run tests based on flags or run all if no specific test is requested
    if [ "$TEST_HEALTH_ONLY" = true ]; then
        check_container_health
        exit $?
    fi
    
    if [ "$TEST_BACKEND_ONLY" = true ]; then
        test_backend
        exit $?
    fi
    
    if [ "$TEST_NGINX_ONLY" = true ]; then
        test_nginx
        exit $?
    fi
    
    if [ "$TEST_NETWORK_ONLY" = true ]; then
        test_network
        exit $?
    fi
    
    if [ "$TEST_PUBLIC_ONLY" = true ]; then
        test_public
        exit $?
    fi
    
    # Run all tests with delays between them
    check_container_health
    local health_failed=$?
    ((total_failed += health_failed))
    wait_between_tests
    
    test_backend
    local backend_failed=$?
    ((total_failed += backend_failed))
    wait_between_tests
    
    test_nginx
    local nginx_failed=$?
    ((total_failed += nginx_failed))
    wait_between_tests
    
    test_network
    local network_failed=$?
    ((total_failed += network_failed))
    wait_between_tests
    
    test_public
    local public_failed=$?
    ((total_failed += public_failed))
    
    # Print endpoint summary at the end
    print_endpoint_summary
    
    log "${YELLOW}Tests completed with $total_failed failures${NC}"
    return $total_failed
}

# Run main function
main
