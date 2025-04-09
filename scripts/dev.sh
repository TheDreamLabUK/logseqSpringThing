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
    
    # Install client dependencies
    cd client && npm install
    cd ..
}

# Start development servers
start_dev() {
    log "${YELLOW}Starting development servers...${NC}"
    
    # Build and start containers with updated configuration
    DOCKER_BUILDKIT=1 docker compose -f docker-compose.dev.yml up --build
}

# Main execution
setup_dev
start_dev
