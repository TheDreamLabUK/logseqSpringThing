#!/bin/bash
set -euo pipefail

# Function to log messages with timestamps
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Function to test an API endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local expected_status=$3
    local description=$4

    log "Testing $description ($method $endpoint)"
    
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" -X "$method" "http://localhost:4000$endpoint")
    
    if [ "$status_code" = "$expected_status" ]; then
        log "✅ $description: Success (Status: $status_code)"
        return 0
    else
        log "❌ $description: Failed (Expected: $expected_status, Got: $status_code)"
        return 1
    fi
}

# Function to test WebSocket connection
test_websocket() {
    log "Testing WebSocket connection (/wss)"
    
    # Use websocat to test WebSocket connection (needs to be installed)
    if ! command -v websocat &> /dev/null; then
        log "⚠️  websocat not found. Install with: cargo install websocat"
        return 1
    fi
    
    # Try to connect with a 5-second timeout
    if timeout 5 websocat "ws://localhost:4000/wss" --no-close &> /dev/null; then
        log "✅ WebSocket connection: Success"
        return 0
    else
        log "❌ WebSocket connection: Failed"
        return 1
    fi
}

# Function to test backend health
test_backend_health() {
    log "Testing backend health"
    
    # Check if container is running
    if ! docker ps --filter "name=logseq-xr-webxr" --format "{{.Status}}" | grep -q "Up"; then
        log "❌ Container is not running"
        return 1
    fi
    
    # Check container health
    local health
    health=$(docker inspect --format "{{.State.Health.Status}}" logseq-xr-webxr)
    
    if [ "$health" = "healthy" ]; then
        log "✅ Container health: Healthy"
        return 0
    else
        log "❌ Container health: $health"
        # Show recent health check logs
        docker inspect logseq-xr-webxr | jq -r ".[0].State.Health.Log[-5:][].Output"
        return 1
    fi
}

# Function to test backend logs
test_backend_logs() {
    log "Checking backend logs for errors"
    
    # Get backend logs
    local logs
    logs=$(docker exec logseq-xr-webxr cat /tmp/webxr.log)
    
    # Check for error messages
    if echo "$logs" | grep -i "error" > /dev/null; then
        log "❌ Found errors in backend logs:"
        echo "$logs" | grep -i "error"
        return 1
    else
        log "✅ No errors found in backend logs"
        return 0
    fi
}

# Main test execution
main() {
    local failed=0
    
    # Test backend health
    if ! test_backend_health; then
        failed=$((failed + 1))
    fi
    
    # Test backend logs
    if ! test_backend_logs; then
        failed=$((failed + 1))
    fi
    
    # Test Graph API endpoints
    if ! test_endpoint "GET" "/api/graph/data" "200" "Graph Data API"; then
        failed=$((failed + 1))
    fi
    
    if ! test_endpoint "GET" "/api/graph/paginated?page=1&pageSize=10" "200" "Paginated Graph API"; then
        failed=$((failed + 1))
    fi
    
    # Test Settings API endpoints
    if ! test_endpoint "GET" "/api/settings" "200" "Settings API"; then
        failed=$((failed + 1))
    fi
    
    if ! test_endpoint "GET" "/api/settings/visualization" "200" "Visualization Settings API"; then
        failed=$((failed + 1))
    fi
    
    if ! test_endpoint "GET" "/api/settings/websocket" "200" "WebSocket Settings API"; then
        failed=$((failed + 1))
    fi
    
    # Test WebSocket connection
    if ! test_websocket; then
        failed=$((failed + 1))
    fi
    
    # Summary
    if [ "$failed" -eq 0 ]; then
        log "🎉 All tests passed!"
        return 0
    else
        log "❌ $failed test(s) failed"
        return 1
    fi
}

# Run tests
main
