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
        --skip-github) SKIP_GITHUB=true ;;
        --skip-websocket) SKIP_WEBSOCKET=true ;;
        --settings-only) TEST_SETTINGS_ONLY=true ;;
        --help) echo "Usage: $0 [-v|--verbose] [--skip-github] [--skip-websocket] [--settings-only] [--help]"; exit 0 ;;
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
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}✗ ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
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
    local failed=0
    
    log_message "Testing graph endpoints..."
    
    # Test basic graph data
    log_message "Testing full graph data endpoint..."
    response=$(docker exec ${CONTAINER_NAME} curl -s --max-time ${TIMEOUT} \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/graph/data")
    status=$?
    
    if [ $status -eq 0 ] && [ ! -z "$response" ]; then
        nodes_count=$(echo "$response" | jq -r '.nodes | length')
        edges_count=$(echo "$response" | jq -r '.edges | length')
        if [ ! -z "$nodes_count" ] && [ ! -z "$edges_count" ]; then
            log_success "Full graph data endpoint responded"
            log_message "Graph data: ${nodes_count} nodes, ${edges_count} edges"
        else
            log_error "Invalid graph data response"
            failed=1
        fi
    else
        log_error "Failed to fetch full graph data (status: ${status})"
        failed=1
    fi

    # Test pagination
    log_message "Testing paginated graph data endpoint..."
    response=$(docker exec ${CONTAINER_NAME} curl -s --max-time ${TIMEOUT} \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        "http://localhost:4000/api/graph/data/paginated?page=1&pageSize=100")
    status=$?
    
    if [ $status -eq 0 ] && [ ! -z "$response" ]; then
        page_count=$(echo "$response" | jq -r '.nodes | length')
        if [ ! -z "$page_count" ]; then
            log_success "Paginated graph data endpoint responded"
            log_message "Paginated data: ${page_count} nodes in first page"
        else
            log_error "Invalid paginated data response"
            failed=1
        fi
    else
        log_error "Failed to fetch paginated graph data (status: ${status})"
        failed=1
    fi
    
    return $failed
}

# Function to check settings endpoint using dedicated script
check_settings_endpoint() {
    log_section "Checking Settings Endpoint"
    log_message "Invoking test_settings.sh..."
    ./scripts/test_settings.sh ${VERBOSE}
    ret=$?
    log_message "test_settings.sh completed with exit code $ret"
    return $ret
}

# Function to test GitHub API endpoints
check_github_endpoints() {
    log_section "Testing GitHub API Endpoints"
    local failed=0
    
    # Load GitHub credentials from .env
    if [ -f ../.env ]; then
        source ../.env
    else
        log_error ".env file not found"
        return 1
    fi
    
    # Test multiple representative files
    local test_files=(
        "p(doom).md"      # Representative markdown file
        "README.md"       # Common documentation file
        "settings.yaml"   # Configuration file
        "package.json"    # Project metadata
    )
    
    log_message "Testing GitHub API access..."
    
    for file in "${test_files[@]}"; do
        # Test with base path only
        path="${GITHUB_BASE_PATH}/${file}"
        encoded_path=$(echo -n "${path}" | jq -sRr @uri)
        
        # Check rate limit before making requests
        rate_limit=$(curl -s -I --max-time ${TIMEOUT} \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/rate_limit")
        status=$?
        
        if [ $status -ne 0 ]; then
            log_error "Failed to check rate limit (status: ${status})"
            failed=1
            break
        fi
        
        remaining=$(echo "$rate_limit" | grep "x-ratelimit-remaining" | cut -d: -f2 | tr -d ' \r')
        
        if [ -z "$remaining" ] || [ "$remaining" -lt 2 ]; then
            log_error "GitHub API rate limit exceeded. Remaining: ${remaining}"
            failed=1
            break
        fi
        
        # Check commits
        commit_response=$(curl -s --max-time ${TIMEOUT} \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/commits?path=${encoded_path}")
        status=$?
        
        if [ $status -ne 0 ]; then
            log_error "Failed to fetch commits for ${file} (status: ${status})"
            failed=1
            continue
        fi
        
        if echo "$commit_response" | jq -e 'has("message")' > /dev/null; then
            error_msg=$(echo "$commit_response" | jq -r '.message')
            log_error "Failed to fetch commits for ${file}: ${error_msg}"
            failed=1
        else
            count=$(echo "$commit_response" | jq -r '. | length')
            
            # Check contents
            content_response=$(curl -s --max-time ${TIMEOUT} \
                -H "Authorization: Bearer ${GITHUB_TOKEN}" \
                -H "Accept: application/vnd.github+json" \
                "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encoded_path}")
            status=$?
            
            if [ $status -ne 0 ]; then
                log_error "Failed to fetch content for ${file} (status: ${status})"
                failed=1
                continue
            fi
            
            if echo "$content_response" | jq -e 'has("message")' > /dev/null; then
                error_msg=$(echo "$content_response" | jq -r '.message')
                log_error "Failed to fetch content for ${file}: ${error_msg}"
                failed=1
            else
                log_success "File: ${file}"
                log_message "  - Commits: ${count}"
                log_message "  - Size: $(echo "$content_response" | jq -r '.size') bytes"
                log_message "  - SHA: $(echo "$content_response" | jq -r '.sha')"
            fi
        fi
        
        # Rate limit delay
        sleep 1
    done
    
    return $failed
}

# Main execution
main() {
    log "${YELLOW}Starting endpoint diagnostics...${NC}"
    local failed=0
    local test_count=0
    local failed_count=0
    
    # Check container status
    if ! docker ps -q -f name=${CONTAINER_NAME} > /dev/null 2>&1; then
        log_error "Container ${YELLOW}${CONTAINER_NAME}${NC} is not running"
        exit 1
    else
        # Check container health status
        health_status=$(docker inspect --format='{{.State.Health.Status}}' ${CONTAINER_NAME})
        log_message "Container health status: ${health_status}"
        if [ "$health_status" = "unhealthy" ]; then
            log_error "Container is marked as unhealthy, but will continue testing..."
            # Show last health check error
            last_health_log=$(docker inspect --format='{{json .State.Health.Log}}' ${CONTAINER_NAME} | jq -r '.[-1].Output')
            log_message "Last health check error: ${last_health_log}"
        fi
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
    
    # Check basic HTTP endpoint
    log_message "Testing basic HTTP endpoint..."
    response=$(docker exec ${CONTAINER_NAME} curl -s --max-time ${TIMEOUT} "http://localhost:4000/")
    status=$?
    
    if [ $status -eq 0 ]; then
        log_success "Basic HTTP endpoint is responding"
    elif [ $status -eq 28 ]; then
        log_error "Basic HTTP endpoint test timed out"
        failed=1
    else
        log_error "Basic HTTP endpoint test failed with status ${status}"
        failed=1
    fi
    
    if [ "$TEST_SETTINGS_ONLY" = true ]; then
        log_message "Running settings tests only..."
        check_settings_endpoint
    else
        # Check settings endpoint
        ((test_count++))
        if ! check_settings_endpoint; then
            failed=1
            ((failed_count++))
        fi
        
        # Check graph endpoints
        ((test_count++))
        if ! check_graph_endpoints; then
            failed=1
            ((failed_count++))
        fi
        
        # Check GitHub API endpoints if not skipped
        if [ "$SKIP_GITHUB" != true ]; then
            ((test_count++))
            if ! check_github_endpoints; then
                failed=1
                ((failed_count++))
            fi
        fi
    fi
    
    # Print summary
    log_section "Test Summary"
    log_message "Total tests: ${test_count}"
    log_message "Failed tests: ${failed_count}"
    [ $failed -eq 0 ] && \
        log_success "All tests passed successfully" || log_error "Some tests failed"
    
    log "${YELLOW}Diagnostics completed${NC}"
    exit $failed
}

# Run main function
main