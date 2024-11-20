#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check pnpm security
check_pnpm_security() {
    echo -e "${YELLOW}Running pnpm security audit...${NC}"
    
    # Run pnpm audit and capture the output
    local audit_output=$(pnpm audit 2>&1)
    local audit_exit=$?
    
    # Count critical vulnerabilities
    local critical_count=$(echo "$audit_output" | grep -i "critical" | grep -o '[0-9]\+ vulnerabilities' | awk '{print $1}' || echo "0")
    
    echo "$audit_output"
    
    if [ "$critical_count" -gt 0 ]; then
        echo -e "${RED}Found $critical_count critical vulnerabilities!${NC}"
        return 1
    elif [ "$audit_exit" -ne 0 ]; then
        echo -e "${YELLOW}Found non-critical vulnerabilities${NC}"
    else
        echo -e "${GREEN}No critical vulnerabilities found${NC}"
    fi
    return 0
}

# Function to check Rust security
check_rust_security() {
    echo -e "${YELLOW}Running cargo audit...${NC}"
    
    # Run cargo audit and capture the output
    local audit_output=$(cargo audit 2>&1)
    local audit_exit=$?
    
    # Count critical vulnerabilities
    local critical_count=$(echo "$audit_output" | grep -i "critical" | wc -l || echo "0")
    
    echo "$audit_output"
    
    if [ "$critical_count" -gt 0 ]; then
        echo -e "${RED}Found $critical_count critical vulnerabilities!${NC}"
        return 1
    elif [ "$audit_exit" -ne 0 ]; then
        echo -e "${YELLOW}Found non-critical vulnerabilities${NC}"
    else
        echo -e "${GREEN}No critical vulnerabilities found${NC}"
    fi
    return 0
}

# Function to read settings from TOML file
read_settings() {
    # Extract domain and port from settings.toml
    export DOMAIN=$(grep "domain = " settings.toml | cut -d'"' -f2)
    export PORT=$(grep "port = " settings.toml | awk '{print $3}')
    
    if [ -z "$DOMAIN" ] || [ -z "$PORT" ]; then
        echo -e "${RED}Error: DOMAIN or PORT not set in settings.toml. Please check your configuration.${NC}"
        exit 1
    fi
}

# Function to check system resources
check_system_resources() {
    echo -e "${YELLOW}Checking GPU availability...${NC}"
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found${NC}"
        exit 1
    fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

# Function to check Docker setup
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi

    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif docker-compose version &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        echo -e "${RED}Error: Docker Compose not found${NC}"
        exit 1
    fi
}

# Function to clean up existing processes
cleanup_existing_processes() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    # Stop all services except cloudflared
    $DOCKER_COMPOSE stop $(docker compose ps --services | grep -v cloudflared) >/dev/null 2>&1
    $DOCKER_COMPOSE rm -f $(docker compose ps --services | grep -v cloudflared) >/dev/null 2>&1

    if netstat -tuln | grep -q ":$PORT "; then
        local pid=$(lsof -t -i:"$PORT")
        if [ ! -z "$pid" ]; then
            kill -9 $pid >/dev/null 2>&1
        fi
    fi
    sleep 2
}

# Function to check container health
check_container_health() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Checking container health...${NC}"
    while [ $attempt -le $max_attempts ]; do
        # Check if container is running
        if ! docker ps | grep -q "logseq-xr-${service}"; then
            echo -e "${RED}Container is not running${NC}"
            return 1
        fi

        # Check container health status directly
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "logseq-xr-${service}")
        
        if [ "$health_status" = "healthy" ]; then
            echo -e "${GREEN}Container is healthy${NC}"
            return 0
        fi

        if (( attempt % 10 == 0 )); then
            echo -e "${YELLOW}Recent logs:${NC}"
            $DOCKER_COMPOSE logs --tail=10 $service
        fi

        echo "Health check attempt $attempt/$max_attempts... (status: $health_status)"
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Container failed to become healthy${NC}"
    $DOCKER_COMPOSE logs --tail=20 $service
    return 1
}

# Function to check application readiness
check_application_readiness() {
    local max_attempts=60
    local attempt=1

    echo -e "${YELLOW}Checking application readiness...${NC}"
    while [ $attempt -le $max_attempts ]; do
        if timeout 5 curl -s http://localhost:4000/ >/dev/null; then
            echo -e "${GREEN}Application is ready${NC}"
            return 0
        fi
        echo "Readiness check attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Application failed to become ready${NC}"
    return 1
}

# Function to test endpoints
test_endpoints() {
    echo -e "\n${YELLOW}Testing endpoints...${NC}"
    
    # Test local index endpoint
    echo -e "\nTesting local index endpoint..."
    local_index_response=$(curl -s -w "\nHTTP Status: %{http_code}\n" http://localhost:4000/)
    if [ $? -eq 0 ] && [ ! -z "$local_index_response" ]; then
        http_status_local=$(echo "$local_index_response" | grep "HTTP Status" | awk '{print $3}')
        echo "$local_index_response" | sed '/HTTP Status/d'
        echo -e "${GREEN}Local index endpoint: OK${NC} (HTTP Status: $http_status_local)"
    else
        echo -e "${RED}Local index endpoint: Failed${NC}"
        return 1
    fi
}

# Function to ensure cloudflared is running and healthy
ensure_cloudflared() {
    local max_attempts=3
    local attempt=1
    local success=false

    while [ $attempt -le $max_attempts ] && [ "$success" = false ]; do
        echo -e "\n${YELLOW}Checking cloudflared status (Attempt $attempt/$max_attempts)...${NC}"
        
        if ! docker ps | grep -q cloudflared-tunnel; then
            echo -e "${YELLOW}Cloudflared tunnel not running, starting it...${NC}"
            $DOCKER_COMPOSE up -d cloudflared
            sleep 10
        fi

        # Check container health status
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' cloudflared-tunnel 2>/dev/null || echo "unknown")
        
        if [ "$health_status" = "healthy" ]; then
            echo -e "${GREEN}Cloudflared tunnel is healthy${NC}"
            success=true
            break
        fi

        # Validate ingress configuration
        echo -e "${YELLOW}Validating cloudflared ingress configuration...${NC}"
        if ! docker exec cloudflared-tunnel cloudflared tunnel ingress validate; then
            echo -e "${RED}Ingress validation failed${NC}"
            if [ $attempt -lt $max_attempts ]; then
                echo -e "${YELLOW}Restarting cloudflared...${NC}"
                $DOCKER_COMPOSE restart cloudflared
                sleep 10
                attempt=$((attempt + 1))
                continue
            else
                echo -e "${RED}Failed to validate cloudflared configuration after $max_attempts attempts${NC}"
                return 1
            fi
        fi

        # Check connectivity
        echo -e "${YELLOW}Checking cloudflared tunnel connectivity...${NC}"
        local conn_attempts=30
        local conn_attempt=1
        
        while [ $conn_attempt -le $conn_attempts ]; do
            # Check for active tunnel connections and errors
            local connection_count=$(docker logs cloudflared-tunnel 2>&1 | grep -c "Registered tunnel connection")
            local error_count=$(docker logs cloudflared-tunnel 2>&1 | grep -c "no more connections active and exiting")
            local conn_error_count=$(docker logs cloudflared-tunnel 2>&1 | grep -c "Unable to reach the origin service")
            
            if [ $connection_count -gt 0 ] && [ $error_count -eq 0 ] && [ $conn_error_count -eq 0 ]; then
                echo -e "${GREEN}Cloudflared tunnel is connected${NC}"
                success=true
                break
            fi
            
            if [ $conn_error_count -gt 0 ]; then
                echo -e "${RED}Connection errors detected${NC}"
                if [ $attempt -lt $max_attempts ]; then
                    echo -e "${YELLOW}Restarting cloudflared...${NC}"
                    $DOCKER_COMPOSE restart cloudflared
                    sleep 10
                    break
                fi
            fi
            
            echo "Connection attempt $conn_attempt/$conn_attempts..."
            sleep 2
            conn_attempt=$((conn_attempt + 1))
        done

        if [ "$success" = false ]; then
            if [ $attempt -lt $max_attempts ]; then
                echo -e "${YELLOW}Retrying cloudflared setup...${NC}"
                $DOCKER_COMPOSE restart cloudflared
                sleep 10
            else
                echo -e "${RED}Failed to establish cloudflared tunnel after $max_attempts attempts${NC}"
                echo -e "${YELLOW}Recent cloudflared logs:${NC}"
                docker logs --tail 50 cloudflared-tunnel
                return 1
            fi
        fi

        attempt=$((attempt + 1))
    done

    if [ "$success" = true ]; then
        return 0
    else
        return 1
    fi
}

# Check environment
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# Source .env file
set -a
source .env
set +a

# Read settings from TOML
read_settings

# Initial setup
check_docker
check_system_resources

# Run security checks
echo -e "\n${YELLOW}Running security checks...${NC}"
if ! check_pnpm_security; then
    echo -e "${RED}Critical vulnerabilities found in pnpm dependencies. Aborting startup.${NC}"
    exit 1
fi

if ! check_rust_security; then
    echo -e "${RED}Critical vulnerabilities found in Rust dependencies. Aborting startup.${NC}"
    exit 1
fi

cleanup_existing_processes

# Clean up old resources
echo -e "${YELLOW}Cleaning up old resources...${NC}"
docker volume ls -q | grep "logseqXR" | xargs -r docker volume rm >/dev/null 2>&1
docker image prune -f >/dev/null 2>&1

# Ensure data directory exists
mkdir -p data/markdown

# Build and start services
echo -e "${YELLOW}Building and starting services...${NC}"
$DOCKER_COMPOSE build --pull --no-cache
$DOCKER_COMPOSE up -d $(docker compose ps --services | grep -v cloudflared)

# Check health and readiness
if ! check_container_health "webxr"; then
    echo -e "${RED}Startup failed${NC}"
    exit 1
fi

if ! check_application_readiness; then
    echo -e "${RED}Startup failed${NC}"
    $DOCKER_COMPOSE logs --tail=50 webxr-graph
    exit 1
fi

# Test endpoints
test_endpoints

# Ensure cloudflared is running and healthy
if ! ensure_cloudflared; then
    echo -e "${RED}Failed to ensure cloudflared is running and healthy${NC}"
    exit 1
fi

# Print final status
echo -e "\n${GREEN}🚀 Services are running!${NC}"

echo -e "\nResource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo -e "\nCommands:"
echo "logs:    $DOCKER_COMPOSE logs -f"
echo "stop:    $DOCKER_COMPOSE down"
echo "restart: $DOCKER_COMPOSE restart"

# Keep script running to show logs
echo -e "\n${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
$DOCKER_COMPOSE logs -f
