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

# Endpoint descriptions
declare -A ENDPOINT_DESCRIPTIONS=(
    ["/api/settings"]="Get application settings and configuration"
    ["/api/graph/data"]="Retrieve graph data and node connections"
    ["/api/settings/graph"]="Get graph-specific settings"
    ["/api/pages"]="List all available pages"
    ["/api/files/fetch"]="Fetch and process files from the repository"
)

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

log_message() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
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
        fi

        # Check application logs
        echo -e "\n'"${CYAN}${BOLD}"'=== Application Log ==='"${NC}"'"
        if [ -f /app/webxr.log ]; then
            tail -n 200 /app/webxr.log
            echo -e "\n'"${RED}${BOLD}"'=== Recent Application Errors ==='"${NC}"'"
            grep -i "error\|warn\|panic\|fatal" /app/webxr.log | tail -n 20
        fi

        # Check metadata
        echo -e "\n'"${CYAN}${BOLD}"'=== Metadata Status ==='"${NC}"'"
        if [ -f /app/data/metadata/metadata.json ]; then
            echo "Metadata file exists and contains:"
            cat /app/data/metadata/metadata.json | jq length
            echo -e "\n'"${CYAN}${BOLD}"'=== First few entries ==='"${NC}"'"
            cat /app/data/metadata/metadata.json | jq "to_entries | .[0:3]"
        else
            echo "Metadata file does not exist!"
        fi

        # Check markdown directory
        echo -e "\n'"${CYAN}${BOLD}"'=== Markdown Files ==='"${NC}"'"
        ls -la /app/data/markdown/
        echo "Total markdown files: $(ls -1 /app/data/markdown/*.md 2>/dev/null | wc -l)"
' | tee -a "$LOG_FILE"

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
            cat /app/settings.toml | grep -i "graph" || echo "No graph settings found"
        ' | tee -a "$LOG_FILE"
    fi

    # For file endpoints, add specific diagnostics
    if [[ "${endpoint}" == *"file"* ]]; then
        echo -e "\n${MAGENTA}${BOLD}File-Specific Diagnostics:${NC}" | tee -a "$LOG_FILE"
        docker exec ${CONTAINER_NAME} bash -c '
            echo -e "\n'"${CYAN}${BOLD}"'=== GitHub Environment ==='"${NC}"'"
            env | grep -i "github"
            echo -e "\n'"${CYAN}${BOLD}"'=== Recent File Operations ==='"${NC}"'"
            grep -i "file\|github" /tmp/webxr.log | tail -n 50
        ' | tee -a "$LOG_FILE"
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
    
    # First check if webxr process is running
    local webxr_pid=$(docker exec ${CONTAINER_NAME} pgrep webxr || echo "")
    if [ -z "$webxr_pid" ]; then
        log_error "WebXR process is not running!"
        # Check both possible log locations and show permissions
        docker exec ${CONTAINER_NAME} bash -c '
            echo "=== Checking log locations ==="
            echo "Permissions for /tmp:"
            ls -la /tmp/
            echo -e "\nPermissions for /app:"
            ls -la /app/
            echo -e "\nAttempting to read logs:"
            echo "/tmp/webxr.log:"
            tail -n 50 /tmp/webxr.log 2>&1
            echo -e "\n/app/webxr.log:"
            tail -n 50 /app/webxr.log 2>&1
        '
        return 1
    fi
    log_success "WebXR process running with PID: $webxr_pid"
    
    # Check if routes are registered
    log_message "Checking registered routes..."
    docker exec ${CONTAINER_NAME} bash -c '
        echo "=== WebXR Routes ==="
        if [ -f /tmp/webxr.log ]; then
            grep -A 10 "Registered routes:" /tmp/webxr.log | tail -n 11
        fi
    '
    
    local endpoints=(
        "/api/settings"
        "/api/graph/data"
        "/api/settings/graph"
        "/api/pages"
        "/api/files/fetch"
    )
    
    local failed=0
    
    for endpoint in "${endpoints[@]}"; do
        log_message "• Testing backend endpoint: $endpoint (${ENDPOINT_DESCRIPTIONS[$endpoint]})"
        
        # Test with more verbose output and headers
        local response=$(docker exec ${CONTAINER_NAME} curl -s -v \
            -H "Content-Type: application/json" \
            -H "Accept: application/json" \
            -H "X-Request-ID: test-$(date +%s)" \
            "http://localhost:3001$endpoint" 2>&1)
        
        if [[ $response == *"200 OK"* ]] || [[ $response == *"201 Created"* ]]; then
            log_success "✓ Endpoint $endpoint is accessible"
            log_message "Response body: $(echo "$response" | grep -v '^*' | grep -v '^>' | grep -v '^<')"
            WORKING_BACKEND_ENDPOINTS+=("$endpoint")
        else
            log_error "✗ Failed to access $endpoint"
            log_message "Response: $response"
            # Check logs for errors around this request
            log_message "Checking recent logs for errors..."
            docker exec ${CONTAINER_NAME} bash -c "grep -B 5 -A 5 \"${endpoint}\" /tmp/webxr.log | tail -n 20"
            ((failed++))
        fi
        
        # Add a small delay between tests
        sleep 1
    done
    
    return $failed
}

# Function to test nginx endpoints
test_nginx() {
    log_section "Testing Nginx Endpoints"
    local failed=0
    local container_ip=$(get_container_ip ${CONTAINER_NAME})
    
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
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint description <<< "$endpoint_info"
        log_verbose "Testing nginx endpoint: $endpoint ($description)"
        
        # Use docker exec to run curl inside the container
        local response=$(docker exec ${CONTAINER_NAME} curl -s -v \
            -H "Accept: */*" \
            -H "Connection: keep-alive" \
            "http://localhost:$NGINX_PORT$endpoint" 2>&1)
        
        if [ $? -eq 0 ] && [[ "$response" != *"Failed to connect"* ]] && [[ "$response" != *"Connection refused"* ]]; then
            log_success "Nginx endpoint $endpoint accessible"
            log_verbose "Response: $response"
            WORKING_NGINX_ENDPOINTS+=("$endpoint:$description")
        else
            log_error "Nginx endpoint $endpoint failed"
            log_verbose "Error response: $response"
            ((failed++))
            # Run diagnostics for failed endpoint
            diagnose_endpoint "$endpoint" "$NGINX_PORT" "$container_ip"
        fi
        
        # Add a small delay between requests
        sleep 1
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
        log_verbose "Testing public endpoint: $endpoint ($description)"
        
        # Use curl with verbose output and proper headers
        local response=$(curl -sk -v \
            -H "Accept: */*" \
            -H "Connection: keep-alive" \
            "https://$PUBLIC_DOMAIN$endpoint" 2>&1)
        
        if [ $? -eq 0 ] && [[ "$response" != *"Failed to connect"* ]] && [[ "$response" != *"Connection refused"* ]]; then
            log_success "Public endpoint $endpoint accessible"
            log_verbose "Response: $response"
            WORKING_PUBLIC_ENDPOINTS+=("$endpoint:$description")
        else
            log_error "Public endpoint $endpoint failed"
            log_verbose "Error response: $response"
            ((failed++))
        fi
        
        # Add a small delay between requests
        sleep 1
    done
    
    return $failed
}

# Add a small delay function
wait_between_tests() {
    sleep 2
    echo -e "\n"
}

# Add this function before main()
print_summary() {
    log_section "API Endpoint Test Summary"
    
    if [ ${#WORKING_BACKEND_ENDPOINTS[@]} -gt 0 ]; then
        log_success "Working Backend Endpoints:"
        for endpoint in "${WORKING_BACKEND_ENDPOINTS[@]}"; do
            log_message "  ✓ $endpoint"
            log_message "    Description: ${ENDPOINT_DESCRIPTIONS[$endpoint]}"
        done
    else
        log_error "No working backend endpoints found"
    fi
    
    if [ ${#WORKING_NGINX_ENDPOINTS[@]} -gt 0 ]; then
        log_success "Working Nginx Endpoints:"
        for endpoint in "${WORKING_NGINX_ENDPOINTS[@]}"; do
            log_message "  ✓ $endpoint"
        done
    else
        log_error "No working Nginx endpoints found"
    fi
    
    if [ ${#WORKING_PUBLIC_ENDPOINTS[@]} -gt 0 ]; then
        log_success "Working Public Endpoints:"
        for endpoint in "${WORKING_PUBLIC_ENDPOINTS[@]}"; do
            log_message "  ✓ $endpoint"
        done
    else
        log_error "No working public endpoints found"
    fi
}

# Function to check application state
check_app_state() {
    log_section "Checking Application State"
    
    docker exec ${CONTAINER_NAME} bash -c '
        echo -e "\n=== Process Status ==="
        ps aux | grep -E "webxr|nginx"
        
        echo -e "\n=== Directory Structure ==="
        echo "Markdown directory:"
        ls -la /app/data/markdown/
        echo -e "\nMetadata directory:"
        ls -la /app/data/metadata/
        
        echo -e "\n=== Metadata Content ==="
        if [ -f /app/data/metadata/metadata.json ]; then
            echo "Metadata file size: $(stat -f %z /app/data/metadata/metadata.json) bytes"
            echo "Number of entries: $(cat /app/data/metadata/metadata.json | jq length)"
        else
            echo "Metadata file does not exist!"
        fi
        
        echo -e "\n=== Recent Logs ==="
        echo "Last 10 lines of application log:"
        tail -n 10 /tmp/webxr.log 2>/dev/null || echo "No application log found"
        
        echo -e "\nLast 10 lines of nginx error log:"
        tail -n 10 /var/log/nginx/error.log 2>/dev/null || echo "No nginx error log found"
    '
}

# Main execution
main() {
    log "${YELLOW}Starting comprehensive endpoint tests...${NC}"
    local total_failed=0
    
    # Check application state first
    check_app_state
    
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
    # check_container_health
    # local health_failed=$?
    # ((total_failed += health_failed))
    # wait_between_tests
    
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
    print_summary
    
    log "${YELLOW}Tests completed with $total_failed failures${NC}"
    return $total_failed
}

# Run main function
main
