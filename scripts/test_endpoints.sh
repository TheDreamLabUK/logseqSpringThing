#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test configuration
declare -A HOST_CONFIGS=(
    ["localhost:4000"]="http"
    ["172.19.0.8:4000"]="http"
    ["www.visionflow.info"]="https"
)

# Function to log messages with timestamps
log() {
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to check required tools
check_requirements() {
    local missing_tools=()

    # Check for curl
    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi

    # Check for websocat
    if ! command -v websocat &> /dev/null; then
        missing_tools+=("websocat")
    fi

    # Check for docker
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    # If any tools are missing, print instructions and exit
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log "${RED}Missing required tools:${NC}"
        for tool in "${missing_tools[@]}"; do
            case $tool in
                "curl")
                    echo "- curl: Install using your package manager (apt install curl)"
                    ;;
                "websocat")
                    echo "- websocat: Install using cargo (cargo install websocat)"
                    ;;
                "docker")
                    echo "- docker: Follow installation guide at https://docs.docker.com/engine/install/"
                    ;;
            esac
        done
        exit 1
    fi
}

# Function to check Docker network and container status
check_docker_status() {
    log "${YELLOW}Checking Docker status...${NC}"

    # Check if docker_ragflow network exists
    if ! docker network ls | grep -q "docker_ragflow"; then
        log "${RED}Error: docker_ragflow network not found${NC}"
        log "Create it with: docker network create docker_ragflow"
        exit 1
    fi

    # Check if required containers are running
    local required_containers=("logseq-xr-webxr" "ragflow-server")
    local missing_containers=()

    for container in "${required_containers[@]}"; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            missing_containers+=("$container")
        fi
    done

    if [ ${#missing_containers[@]} -ne 0 ]; then
        log "${RED}Error: Required containers not running:${NC}"
        printf '%s\n' "${missing_containers[@]}"
        exit 1
    fi

    log "${GREEN}Docker network and containers are ready${NC}"
}

# Function to get base URL for a host
get_base_url() {
    local host=$1
    local protocol=${HOST_CONFIGS[$host]}
    echo "${protocol}://${host}"
}

# Function to get WebSocket URL for a host
get_ws_url() {
    local host=$1
    local protocol=${HOST_CONFIGS[$host]}
    if [[ $protocol == "https" ]]; then
        echo "wss://${host}/wss"
    else
        echo "ws://${host}/wss"
    fi
}

# Function to test REST endpoints for a given host
test_rest_endpoints() {
    local host=$1
    local base_url=$(get_base_url "$host")
    
    log "${YELLOW}Testing REST endpoints on ${host}...${NC}"

    # Array of endpoints to test with their expected response patterns
    declare -A endpoints=(
        ["/api/files/fetch"]="nodes"
        ["/api/graph/data"]="nodes"
        ["/api/graph/data/paginated?page=0&page-size=100"]="nodes"
        ["/api/visualization/settings/client-debug"]="enable"
        ["/api/visualization/settings/server-debug"]="enable"
        ["/api/visualization/settings/nodes/base-size"]="value"
        ["/api/ragflow/init"]="conversation"
        ["/api/perplexity/models"]="models"
    )

    local failed=0
    for endpoint in "${!endpoints[@]}"; do
        local expected="${endpoints[$endpoint]}"
        log "Testing ${endpoint}..."
        
        response=$(curl -sk "${base_url}${endpoint}")
        if [[ $response == *"${expected}"* ]]; then
            log "${GREEN}✓ ${endpoint} working${NC}"
        else
            log "${RED}✗ ${endpoint} failed${NC}"
            log "Response: ${response}"
            ((failed++))
        fi
    done

    # Test POST endpoints
    log "Testing POST endpoints..."

    # Test graph update endpoint
    update_response=$(curl -sk -X POST "${base_url}/api/graph/update" \
        -H "Content-Type: application/json" \
        -d '{"nodes":[],"edges":[]}')
    if [[ $update_response != *"error"* ]]; then
        log "${GREEN}✓ POST /api/graph/update working${NC}"
    else
        log "${RED}✗ POST /api/graph/update failed${NC}"
        ((failed++))
    fi

    # Test RAGFlow message endpoint
    ragflow_response=$(curl -sk -X POST "${base_url}/api/ragflow/send" \
        -H "Content-Type: application/json" \
        -d '{"conversation_id":"test","messages":[{"role":"user","content":"test"}]}')
    if [[ $ragflow_response != *"error"* ]]; then
        log "${GREEN}✓ POST /api/ragflow/send working${NC}"
    else
        log "${RED}✗ POST /api/ragflow/send failed${NC}"
        ((failed++))
    fi

    return $failed
}

# Function to test WebSocket connection for a given host
test_websocket() {
    local host=$1
    local ws_url=$(get_ws_url "$host")
    
    log "${YELLOW}Testing WebSocket connection on ${host}...${NC}"

    # Additional options for secure WebSocket
    local ws_opts=""
    if [[ ${HOST_CONFIGS[$host]} == "https" ]]; then
        ws_opts="--no-verify"
    fi

    # Test initial connection
    log "Testing WebSocket connection establishment..."
    if ! timeout 5 websocat $ws_opts "$ws_url" > /dev/null 2>&1 <<< '{"type":"ping"}'; then
        log "${RED}✗ WebSocket connection failed${NC}"
        return 1
    fi
    log "${GREEN}✓ WebSocket connection successful${NC}"

    # Test message types
    declare -A messages=(
        ["ping"]='{"type":"ping"}'
        ["enableBinaryUpdates"]='{"type":"enableBinaryUpdates","data":{"enabled":true}}'
        ["requestInitialData"]='{"type":"requestInitialData"}'
    )

    local failed=0
    for msg_type in "${!messages[@]}"; do
        log "Testing ${msg_type} message..."
        if echo "${messages[$msg_type]}" | websocat $ws_opts "$ws_url" --binary -n1 > /dev/null; then
            log "${GREEN}✓ ${msg_type} message handled successfully${NC}"
        else
            log "${RED}✗ ${msg_type} message failed${NC}"
            ((failed++))
        fi
    done

    # Test error handling
    log "Testing invalid message handling..."
    if echo "invalid json" | websocat $ws_opts "$ws_url" -n1 2>&1 | grep -q "error"; then
        log "${GREEN}✓ Invalid message handled correctly${NC}"
    else
        log "${RED}✗ Invalid message handling failed${NC}"
        ((failed++))
    fi

    return $failed
}

# Function to test all endpoints on all hosts
test_all_endpoints() {
    local total_failed=0

    for host in "${!HOST_CONFIGS[@]}"; do
        echo
        log "${YELLOW}Testing host: ${host} (${HOST_CONFIGS[$host]})${NC}"
        echo "================================"

        # Test REST endpoints
        test_rest_endpoints "$host"
        local rest_failed=$?
        ((total_failed += rest_failed))

        # Test WebSocket
        test_websocket "$host"
        local ws_failed=$?
        ((total_failed += ws_failed))

        echo
        log "${YELLOW}Results for ${host}:${NC}"
        echo "REST Tests: $([ $rest_failed -eq 0 ] && echo "${GREEN}PASS${NC}" || echo "${RED}FAIL ($rest_failed failed)${NC}")"
        echo "WebSocket Tests: $([ $ws_failed -eq 0 ] && echo "${GREEN}PASS${NC}" || echo "${RED}FAIL ($ws_failed failed)${NC}")"
        echo "--------------------------------"
    done

    return $total_failed
}

# Main execution
main() {
    log "${YELLOW}Starting comprehensive endpoint tests...${NC}"

    # Check for required tools
    check_requirements

    # Check Docker status
    check_docker_status

    echo "Testing endpoints on the following hosts:"
    for host in "${!HOST_CONFIGS[@]}"; do
        echo "- ${host} (${HOST_CONFIGS[$host]})"
    done
    echo

    test_all_endpoints
    local total_failed=$?

    echo
    log "${YELLOW}Final Summary:${NC}"
    if [ $total_failed -eq 0 ]; then
        log "${GREEN}All tests passed successfully!${NC}"
        exit 0
    else
        log "${RED}${total_failed} tests failed across all hosts${NC}"
        exit 1
    fi
}

# Run main function
main
