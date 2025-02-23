#!/bin/bash

# Enable error reporting
set -e

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="logseq-xr-webxr"
VERBOSE=false
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/settings_test_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Logging functions
log() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}✗ ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n" | tee -a "$LOG_FILE"
}

# Function to check if a setting exists and is not null
check_setting() {
    local obj=$1
    local path=$2
    local name=$3
    local value=$(echo "$obj" | jq -r "$path")
    
    if [ -z "$value" ] || [ "$value" = "null" ]; then
        log_error "$name is missing or invalid"
        return 1
    elif [ "$VERBOSE" = true ]; then
        log "Found $name: $value"
    fi
    return 0
}

# Function to check visualization settings
check_visualization_settings() {
    local settings=$1
    local failed=0
    
    log_section "Checking Visualization Settings"
    
    # Check nodes section
    local nodes=$(echo "$settings" | jq -r '.visualization.nodes')
    if [ -z "$nodes" ] || [ "$nodes" = "null" ]; then
        log_error "Node settings missing"
        failed=1
    else
        check_setting "$nodes" '.base_color' "Node base color" || failed=1
        check_setting "$nodes" '.size_range' "Node size range" || failed=1
        check_setting "$nodes" '.quality' "Node quality" || failed=1
        check_setting "$nodes" '.enable_instancing' "Node instancing" || failed=1
        check_setting "$nodes" '.enable_hologram' "Node hologram" || failed=1
        check_setting "$nodes" '.enable_metadata_visualization' "Node metadata visualization" || failed=1
    fi
    
    # Check edges section
    local edges=$(echo "$settings" | jq -r '.visualization.edges')
    if [ -z "$edges" ] || [ "$edges" = "null" ]; then
        log_error "Edge settings missing"
        failed=1
    else
        check_setting "$edges" '.color' "Edge color" || failed=1
        check_setting "$edges" '.width_range' "Edge width range" || failed=1
        check_setting "$edges" '.quality' "Edge quality" || failed=1
        check_setting "$edges" '.arrow_size' "Edge arrow size" || failed=1
    fi
    
    # Check physics section
    local physics=$(echo "$settings" | jq -r '.visualization.physics')
    if [ -z "$physics" ] || [ "$physics" = "null" ]; then
        log_error "Physics settings missing"
        failed=1
    else
        check_setting "$physics" '.enabled' "Physics enabled" || failed=1
        check_setting "$physics" '.iterations' "Physics iterations" || failed=1
        check_setting "$physics" '.attraction_strength' "Attraction strength" || failed=1
        check_setting "$physics" '.repulsion_strength' "Repulsion strength" || failed=1
    fi
    
    # Check rendering section
    local rendering=$(echo "$settings" | jq -r '.visualization.rendering')
    if [ -z "$rendering" ] || [ "$rendering" = "null" ]; then
        log_error "Rendering settings missing"
        failed=1
    else
        check_setting "$rendering" '.background_color' "Background color" || failed=1
        check_setting "$rendering" '.ambient_light_intensity' "Ambient light intensity" || failed=1
        check_setting "$rendering" '.enable_antialiasing' "Antialiasing" || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        log_success "Visualization settings validation passed"
        return 0
    fi
    return 1
}

# Function to check system settings
check_system_settings() {
    local settings=$1
    local failed=0
    
    log_section "Checking System Settings"
    
    # Check websocket section
    local websocket=$(echo "$settings" | jq -r '.system.websocket')
    if [ -z "$websocket" ] || [ "$websocket" = "null" ]; then
        log_error "WebSocket settings missing"
        failed=1
    else
        check_setting "$websocket" '.update_rate' "WebSocket update rate" || failed=1
        check_setting "$websocket" '.reconnect_attempts' "Reconnect attempts" || failed=1
        check_setting "$websocket" '.binary_chunk_size' "Binary chunk size" || failed=1
    fi
    
    # Check debug section
    local debug=$(echo "$settings" | jq -r '.system.debug')
    if [ -z "$debug" ] || [ "$debug" = "null" ]; then
        log_error "Debug settings missing"
        failed=1
    else
        check_setting "$debug" '.enabled' "Debug enabled" || failed=1
        check_setting "$debug" '.log_level' "Log level" || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        log_success "System settings validation passed"
        return 0
    fi
    return 1
}

# Function to check XR settings
check_xr_settings() {
    local settings=$1
    local failed=0
    
    log_section "Checking XR Settings"
    
    local xr=$(echo "$settings" | jq -r '.xr')
    if [ -z "$xr" ] || [ "$xr" = "null" ]; then
        log_error "XR settings missing"
        failed=1
    else
        check_setting "$xr" '.mode' "XR mode" || failed=1
        check_setting "$xr" '.room_scale' "Room scale" || failed=1
        check_setting "$xr" '.enable_hand_tracking' "Hand tracking" || failed=1
        check_setting "$xr" '.hand_mesh_enabled' "Hand mesh" || failed=1
        check_setting "$xr" '.movement_speed' "Movement speed" || failed=1
        check_setting "$xr" '.interaction_radius' "Interaction radius" || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        log_success "XR settings validation passed"
        return 0
    fi
    return 1
}

# Main function to test settings
main() {
    log "${YELLOW}Starting settings validation...${NC}"
    local failed=0
    
    # Fetch settings from API
    log "Fetching settings from API..."
    local response=$(docker exec ${CONTAINER_NAME} curl -s \
        "http://localhost:4000/api/user-settings")
    
    if [ -z "$response" ] || [ "$response" = "null" ]; then
        log_error "Failed to fetch settings from API"
        exit 1
    fi
    
    # First verify we got valid JSON
    if ! echo "$response" | jq -e . >/dev/null 2>&1; then
        log_error "Invalid JSON response from API"
        exit 1
    fi
    
    # Run all checks
    check_visualization_settings "$response" || failed=1
    check_system_settings "$response" || failed=1
    check_xr_settings "$response" || failed=1
    
    # Final status
    log_section "Settings Validation Summary"
    if [ $failed -eq 0 ]; then
        log_success "All settings validated successfully"
        exit 0
    else
        log_error "Some settings validation failed"
        exit 1
    fi
}

# Run main function
main