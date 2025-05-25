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
        "http://ragflow-server:9380/api/v1/system/healthz" # Check RAGFlow API using service name and standard health endpoint
        "http://whisper-webui:8000/health" # Check Whisper WebUI health
        "http://reverent_murdock:8880/health" # Check Kokoro TTS health
    )

    for endpoint in "${endpoints[@]}"; do
        echo -e "\nTesting $endpoint"
        if [[ "$endpoint" == *"ragflow-server"* ]]; then
            # RAGFlow health endpoint usually doesn't require auth, but good to have if testing other RAGFlow endpoints
            if [ -n "$RAGFLOW_API_KEY" ]; then
                echo "Attempting RAGFlow check with API Key (though /healthz might not need it)"
                curl -v -H "Authorization: Bearer $RAGFLOW_API_KEY" "$endpoint"
            else
                echo "Attempting RAGFlow check without API Key"
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