#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER_NAME="logseqspringthing-webxr-1"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Test categories
test_connectivity() {
    echo -e "${YELLOW}Testing basic connectivity...${NC}"
    curl -s http://localhost:4000/api/health
}

test_settings() {
    echo -e "${YELLOW}Testing settings endpoints...${NC}"
    curl -s http://localhost:4000/api/user-settings
}

test_websocket() {
    echo -e "${YELLOW}Testing WebSocket connection...${NC}"
    websocat ws://localhost:4000/wss
}

# Main execution
echo -e "${GREEN}Starting tests...${NC}"
test_connectivity
test_settings
test_websocket