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

# Setup development environment
setup_dev() {
    log "${YELLOW}Setting up development environment...${NC}"
    mkdir -p /tmp/docker-cache
    
    # Install client dependencies
    cd client && pnpm install
    cd ..
}

# Start development servers with PTX compilation in container
start_dev() {
    log "${YELLOW}Starting development servers...${NC}"
    
    # Build and start containers
    DOCKER_BUILDKIT=1 docker compose -f docker-compose.dev.yml up --build
}

# Main execution
setup_dev
start_dev
