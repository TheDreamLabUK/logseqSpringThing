#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Security checks
check_security() {
    log "${YELLOW}Running security checks...${NC}"
    pnpm audit
    cargo audit
}

# Environment setup
setup_environment() {
    # Load environment variables
    set -a
    source .env
    set +a

    # Set GPU configuration
    export NVIDIA_GPU_UUID="GPU-553dc306-dab3-32e2-c69b-28175a6f4da6"
    export NVIDIA_VISIBLE_DEVICES="$NVIDIA_GPU_UUID"
}

# Main execution
setup_environment
check_security

log "${YELLOW}Building and starting services...${NC}"
docker compose build --pull
docker compose up -d

log "${GREEN}Services started. Use these commands:${NC}"
echo "logs:    docker compose logs -f"
echo "stop:    docker compose down"
echo "restart: docker compose restart"