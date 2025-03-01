# Graph Node Stacking Fix

## Issue Description

This document describes the issue where nodes would appear stacked on top of each other for the first client connecting to the visualization, while subsequent clients would see the graph correctly distributed.

### Root Cause Analysis

The issue was identified as a timing problem related to when clients connect and request data, combined with how the server handles the initial graph setup and GPU computation:

1. **Server Start Sequence**: When the server starts, it loads the initial graph data and begins GPU computation to optimize node positions.

2. **First Client Connection Issue**: If a client connects before the GPU has had enough time to run even a single iteration, the server would send unoptimized node positions (possibly all at the origin or random positions).

3. **Subsequent Clients**: Later clients would connect after the GPU had been running for some time, receiving well-distributed node positions.

4. **Update Limitation**: Due to the `initial_data_sent` flag in the WebSocket handler, the first client would only receive the initial (poor) position data, but not subsequent updates from the GPU, causing nodes to remain stacked.

## Solution Implementation

The fix involves several coordinated changes across the codebase:

### Server-Side Changes

1. **Remove `initial_data_sent` Flag** (in `src/handlers/socket_flow_handler.rs`):
   - Removed the flag that prevented sending continuous updates after initial data
   - This ensures all clients receive ongoing position updates from the GPU computation

2. **Add Loading Message**:
   - Added a "loading" message when clients first connect
   - This signals clients to display a loading indicator while waiting for optimized positions

3. **Add Initial Delay** (in `src/main.rs`):
   - Added a 500ms delay between graph initialization and HTTP server startup
   - This gives the GPU computation time to run and optimize node positions before any clients can connect

```rust
// Add a delay to allow GPU computation to run before accepting client connections
info!("Waiting for initial GPU layout calculation...");
tokio::time::sleep(Duration::from_millis(500)).await;
info!("Initial delay complete. Starting HTTP server...");
```

### Client-Side Changes

1. **Loading State Handling** (in `client/websocket/websocketService.ts`):
   - Added handling for "loading" and "updatesStarted" messages
   - Added `onLoadingStatusChange` callback registration for components to respond to loading state

2. **Loading Indicator** (in `client/rendering/VisualizationController.ts`):
   - Implemented a loading indicator that displays during the initial graph calculation
   - Added methods to show/hide the loading indicator based on WebSocket messages

## Results

With these changes implemented:

1. The GPU computation has time to run before any clients connect
2. All clients receive the same well-distributed initial layout
3. Users see a loading indicator during the initialization phase
4. All clients receive continuous position updates from ongoing GPU computation

This ensures a consistent and visually pleasing experience for all users, regardless of when they connect to the visualization.

## Related Components

- `src/handlers/socket_flow_handler.rs`: WebSocket server implementation
- `src/main.rs`: Application entry point and initialization
- `client/websocket/websocketService.ts`: Client WebSocket handler
- `client/rendering/VisualizationController.ts`: Main visualization controller