#!/bin/bash
# Modernized diagnostics tool that combines logging and endpoint checking

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER_NAME="logseq_spring_thing_webxr"

show_diagnostics() {
    echo -e "${YELLOW}=== Container Status ===${NC}"
    docker ps -a | grep $CONTAINER_NAME

    echo -e "\n${YELLOW}=== Container Logs ===${NC}"
    docker logs $CONTAINER_NAME

    echo -e "\n${YELLOW}=== Application Logs ===${NC}"
    docker exec $CONTAINER_NAME cat /app/webxr.log

    echo -e "\n${YELLOW}=== Environment Variables ===${NC}"
    docker exec $CONTAINER_NAME env

    echo -e "\n${YELLOW}=== Network Status ===${NC}"
    docker exec $CONTAINER_NAME ip addr show
    docker exec $CONTAINER_NAME netstat -tulpn

    echo -e "\n${YELLOW}=== Resource Usage ===${NC}"
    docker stats $CONTAINER_NAME --no-stream

    echo -e "\n${YELLOW}=== GPU Status ===${NC}"
    docker exec $CONTAINER_NAME nvidia-smi
}

check_endpoints() {
    echo -e "${YELLOW}=== Testing Endpoints ===${NC}"
    endpoints=(
        "http://localhost:4000/api/health"
        "http://localhost:4000/api/graph/data"
        "http://localhost:4000/api/files/fetch"
        "http://192.168.0.51/v1/ping" # Check RAGFlow API
    )

    for endpoint in "${endpoints[@]}"; do
        echo -e "\nTesting $endpoint"
        # For RAGFlow, we might need an API key, but a basic connectivity check should work
        if [[ "$endpoint" == *"192.168.0.51"* ]]; then
            # Attempt to curl with API key if available in env, otherwise just curl
            # This assumes diagnostics.sh might be run in an env where RAGFLOW_API_KEY is set
            if [ -n "$RAGFLOW_API_KEY" ]; then
                curl -v -H "Authorization: Bearer $RAGFLOW_API_KEY" "$endpoint"
            else
                curl -v "$endpoint"
            fi
        else
            curl -v "$endpoint"
        fi
    done
}

# Main execution
echo -e "${GREEN}Starting diagnostics...${NC}"
show_diagnostics
check_endpoints