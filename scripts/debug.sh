#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to show logs
show_logs() {
    echo -e "${YELLOW}=== Container Logs ===${NC}"
    docker logs logseq-xr-webxr

    echo -e "\n${YELLOW}=== Application Logs ===${NC}"
    docker exec logseq-xr-webxr cat /app/webxr.log

    echo -e "\n${YELLOW}=== Environment Variables ===${NC}"
    docker exec logseq-xr-webxr env

    echo -e "\n${YELLOW}=== Directory Structure ===${NC}"
    docker exec logseq-xr-webxr ls -la /app/data/markdown
    docker exec logseq-xr-webxr ls -la /app/data/metadata

    echo -e "\n${YELLOW}=== Metadata Content ===${NC}"
    docker exec logseq-xr-webxr cat /app/data/metadata/metadata.json

    echo -e "\n${YELLOW}=== GitHub API Test ===${NC}"
    docker exec logseq-xr-webxr curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
        "https://api.github.com/repos/$GITHUB_OWNER/$GITHUB_REPO/contents/$GITHUB_BASE_PATH"
}

# Function to check endpoints
check_endpoints() {
    echo -e "${YELLOW}=== Testing Endpoints ===${NC}"
    curl -v http://localhost:4000/api/graph/data
    echo -e "\n"
    curl -v http://localhost:4000/api/files/fetch
}

# Main debug flow
echo -e "${GREEN}Starting debug process...${NC}"
show_logs
check_endpoints 