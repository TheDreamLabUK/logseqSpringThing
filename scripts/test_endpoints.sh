#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# Best Practice Refactored Test Script with Dotenv Loading
# ------------------------------------------------------------------------------
# - Each test function returns 0 for success, 1 for failure.
# - The main function accumulates test results, prints a summary,
#   and exits with the final code.
# - Loads GitHub credentials from a .env file located in the parent or current directory.
# ------------------------------------------------------------------------------

set -Eeuo pipefail  # Safer defaults; won't auto-exit on single command failure

# For debugging/troubleshooting (optional):
trap 'echo -e "\n[ERROR] Script encountered an error at line $LINENO." >&2' ERR

# ------------------------------------------------------------------------------
# Constants & Variables
# ------------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Other configuration
BACKEND_PORT=3001
NGINX_PORT=4000
CONTAINER_NAME="logseq-xr-webxr"
PUBLIC_DOMAIN="www.visionflow.info"
RAGFLOW_NETWORK="docker_ragflow"
VERBOSE=true
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT=5

# ------------------------------------------------------------------------------
# Load Dotenv Credentials for GitHub
# ------------------------------------------------------------------------------
if [ -f "../.env" ]; then
    set -a
    source "../.env"
    set +a
elif [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
else
    echo -e "${RED}Error: .env file not found in parent or current directory${NC}"
    exit 1
fi

# Verify required GitHub variables are loaded
if [ -z "$GITHUB_TOKEN" ] || [ -z "$GITHUB_OWNER" ] || [ -z "$GITHUB_REPO" ]; then
    echo -e "${YELLOW}Warning: Some GitHub credentials are missing in .env file${NC}"
    echo "Required variables:"
    echo "GITHUB_TOKEN: ${GITHUB_TOKEN:-not set}"
    echo "GITHUB_OWNER: ${GITHUB_OWNER:-not set}"
    echo "GITHUB_REPO: ${GITHUB_REPO:-not set}"
fi

# ------------------------------------------------------------------------------
# Logging Helpers
# ------------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

log() {
  echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_verbose() {
  if [ "$VERBOSE" = true ]; then
    echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${CYAN}•${NC} $1" | tee -a "$LOG_FILE"
  fi
}

log_error() {
  echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}✗ ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
  echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

log_message() {
  echo -e "${GRAY}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
  echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n" | tee -a "$LOG_FILE"
}

# ------------------------------------------------------------------------------
# Command-Line Args
# ------------------------------------------------------------------------------
SKIP_GITHUB=false
SKIP_WEBSOCKET=false
TEST_SETTINGS_ONLY=false

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -v|--verbose) VERBOSE=true ;;
    --skip-github) SKIP_GITHUB=true ;;
    --skip-websocket) SKIP_WEBSOCKET=true ;;
    --settings-only) TEST_SETTINGS_ONLY=true ;;
    --help)
      echo "Usage: $0 [-v|--verbose] [--skip-github] [--skip-websocket] [--settings-only] [--help]"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

# ------------------------------------------------------------------------------
# GitHub API Helper Function
# ------------------------------------------------------------------------------
check_github_file() {
  local file="$1"
  local encoded_path
  encoded_path=$(echo -n "$file" | jq -sRr @uri)

  # Fetch file metadata from GitHub
  local response
  response=$(curl -s --max-time "${TIMEOUT}" \
      -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${encoded_path}" || true)

  if [ -z "$response" ]; then
      log_error "GitHub API: No response for file ${file}"
      return 1
  fi

  if echo "$response" | jq -e 'has("message")' >/dev/null 2>&1; then
      local error_msg
      error_msg=$(echo "$response" | jq -r '.message')
      log_error "GitHub API: Error for file ${file}: ${error_msg}"
      return 1
  fi

  # Optionally, fetch commit history for the file
  local commits_response
  commits_response=$(curl -s --max-time "${TIMEOUT}" \
      -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/commits?path=${encoded_path}" || true)

  if [ -z "$commits_response" ]; then
      log_error "GitHub API: No commit history for file ${file}"
      return 1
  fi

  if echo "$commits_response" | jq -e 'has("message")' >/dev/null 2>&1; then
      local error_msg
      error_msg=$(echo "$commits_response" | jq -r '.message')
      log_error "GitHub API: Error fetching commits for file ${file}: ${error_msg}"
      return 1
  fi

  local commit_count
  commit_count=$(echo "$commits_response" | jq -r 'length')

  log_success "GitHub file check passed for ${file} (Commits: ${commit_count})"
  return 0
}

# ------------------------------------------------------------------------------
# Test Functions
# ------------------------------------------------------------------------------

check_container_health() {
  # Returns 0 if container is healthy, 1 otherwise

  local container_running
  container_running=$(docker ps -q -f name="${CONTAINER_NAME}" || true)
  if [ -z "$container_running" ]; then
    log_error "Container ${CONTAINER_NAME} is not running"
    return 1
  fi

  if ! docker exec "${CONTAINER_NAME}" curl -s --max-time 5 "http://localhost:4000/" >/dev/null; then
    log_error "Container HTTP endpoint is not responding"
    return 1
  fi

  local process_check
  process_check=$(docker exec "${CONTAINER_NAME}" ps -e | grep -v grep | grep "webxr" || true)
  if [ -z "$process_check" ]; then
    log_error "Main WebXR process is not running in container"
    return 1
  fi

  log_success "Container health check passed"
  return 0
}

check_settings_endpoint() {
  log_section "Checking Settings Endpoint"
  log_message "Testing settings endpoint..."

  local response status
  response=$(docker exec "${CONTAINER_NAME}" curl -s --max-time "${TIMEOUT}" \
    "http://localhost:4000/api/user-settings" || true)
  status=$?

  if [ $status -ne 0 ] || [ -z "$response" ]; then
    log_error "Failed to fetch or empty response from settings endpoint"
    return 1
  fi

  if ! echo "$response" | jq -e . >/dev/null 2>&1; then
    log_error "Invalid JSON response from settings endpoint"
    return 1
  fi

  log_success "Settings endpoint check passed"
  return 0
}

check_graph_endpoints() {
  log_section "Checking Graph Endpoints"
  log_message "Testing full graph data endpoint..."

  local response
  response=$(docker exec "${CONTAINER_NAME}" curl -s --max-time "${TIMEOUT}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    "http://localhost:4000/api/graph/data" || true)

  if [ -z "$response" ]; then
    log_error "Empty or no response from graph data endpoint"
    return 1
  fi

  log_success "Graph endpoints check passed"
  return 0
}

check_github_endpoints() {
    log_section "Testing GitHub API Access"
    local failed=0

    # Check if required variables are set (including GITHUB_BASE_PATH)
    if [ -z "${GITHUB_TOKEN:-}" ] || [ -z "${GITHUB_OWNER:-}" ] || [ -z "${GITHUB_REPO:-}" ] || [ -z "${GITHUB_BASE_PATH:-}" ]; then
        log_error "Missing required GitHub configuration"
        return 1
    fi

    # Test a few markdown files in the specified base path
    local files=(
        "${GITHUB_BASE_PATH}/Debug Test Page.md"
    )

    for file in "${files[@]}"; do
        check_github_file "$file"
        if [ $? -ne 0 ]; then
            failed=$((failed + 1))
        fi
    done

    return $failed
}

check_settings_sync() {
  log_section "Testing Settings Synchronization"
  local failed=0

  local test_value="#FF0000"
  local test_setting="visualization.nodes.base_color"

  log_message "Updating setting to ${test_value}..."
  local update_response
  update_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"$test_setting\":\"$test_value\"}" \
    "http://localhost:${NGINX_PORT}/api/user-settings" || true)

  if [ -z "$update_response" ]; then
    log_error "Failed to update setting"
    return 1
  fi

  sleep 1
  local verify_response
  verify_response=$(curl -s "http://localhost:${NGINX_PORT}/api/user-settings" || true)

  local actual_value
  actual_value=$(echo "$verify_response" | jq -r ".$test_setting" || echo "")
  if [ "$actual_value" != "$test_value" ]; then
    log_error "Setting verification failed. Expected: $test_value, Got: $actual_value"
    failed=1
  fi

  return $failed
}

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
main() {
  log "${YELLOW}Starting endpoint diagnostics...${NC}"
  local failed=0
  local test_count=0
  local failed_count=0

  if ! check_container_health; then
    log_error "Container health check failed. Continuing with tests..."
  fi

  log_message "Checking network connectivity..."
  if ! docker exec "${CONTAINER_NAME}" bash -c '
        echo -e "=== Network Status ==="
        ip addr show | grep -A2 "eth0" | head -n3
        ip route | grep default
        grep "nameserver" /etc/resolv.conf | head -n2
  ' | tee -a "$LOG_FILE"; then
    log_error "Network connectivity check failed. Continuing..."
  fi

  log_message "Testing basic HTTP endpoint..."
  if docker exec "${CONTAINER_NAME}" curl -s --max-time "${TIMEOUT}" "http://localhost:4000/" > /dev/null; then
    log_success "Basic HTTP endpoint is responding"
  else
    log_error "Basic HTTP endpoint test failed - cannot continue"
    exit 1
  fi

  if [ "$TEST_SETTINGS_ONLY" = true ]; then
    log_section "Settings-Only Mode"
    check_settings_endpoint
    exit $?
  fi

  log_section "Running All Tests"

  test_count=$((test_count+1))
  if ! check_settings_endpoint; then
    failed_count=$((failed_count+1))
    failed=1
  fi

  test_count=$((test_count+1))
  if ! check_graph_endpoints; then
    failed_count=$((failed_count+1))
    failed=1
  fi

  if [ "$SKIP_GITHUB" != true ]; then
    test_count=$((test_count+1))
    if ! check_github_endpoints; then
      failed_count=$((failed_count+1))
      failed=1
    fi
  fi

  log_section "Test Summary"
  log_message "Total tests:  ${test_count}"
  log_message "Failed tests: ${failed_count}"

  if [ $failed -eq 0 ]; then
    log_success "All tests passed successfully"
  else
    log_error "Some tests failed"
  fi

  log "${YELLOW}Diagnostics completed${NC}"
  exit $failed
}

main
