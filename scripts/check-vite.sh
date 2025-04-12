#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Container name
CONTAINER_NAME="logseq_spring_thing_webxr"

# Check if container is running
if ! docker ps -q -f name=$CONTAINER_NAME > /dev/null; then
    echo -e "${RED}Container $CONTAINER_NAME is not running.${NC}"
    echo -e "Start the development environment with ${YELLOW}./scripts/dev.sh${NC} first."
    exit 1
fi

# Function to display usage
usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  ./scripts/check-vite.sh [options]"
    echo -e "\n${YELLOW}Options:${NC}"
    echo -e "  -f, --follow    Follow the Vite logs (like tail -f)"
    echo -e "  -n, --lines N   Show last N lines (default: 50)"
    echo -e "  -h, --help      Show this help message"
    echo -e "\n${YELLOW}Examples:${NC}"
    echo -e "  ./scripts/check-vite.sh            # Show last 50 lines"
    echo -e "  ./scripts/check-vite.sh -f         # Follow the logs"
    echo -e "  ./scripts/check-vite.sh -n 100     # Show last 100 lines"
}

# Default values
FOLLOW=false
LINES=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check Vite process status
echo -e "${GREEN}Checking Vite process status:${NC}"
docker exec $CONTAINER_NAME ps aux | grep -v grep | grep -E "npm run dev|node.*vite"

# Display Vite logs
echo -e "\n${GREEN}Vite Server Logs:${NC}"
if [ "$FOLLOW" = true ]; then
    echo -e "${YELLOW}Following logs. Press Ctrl+C to exit.${NC}"
    docker logs -f $CONTAINER_NAME | grep -v "Rust server" | grep -v "Rotating logs"
else
    docker logs $CONTAINER_NAME --tail $LINES | grep -v "Rust server" | grep -v "Rotating logs"
fi
