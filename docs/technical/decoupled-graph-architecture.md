# Decoupled Graph Architecture

## Overview

The LogseqXR graph architecture has been modernized to decouple graph initialization and physics processing from client connections. This document outlines the new architecture and explains the key components and their interactions.

## Architecture Components

### Server-Side Components

-   **GraphService ([`src/services/graph_service.rs`](../../src/services/graph_service.rs))**: Continuously maintains the force-directed graph (nodes, edges, positions) independently of client connections. It runs the physics simulation.
-   **ClientManager** (typically part of or used by [`src/handlers/socket_flow_handler.rs`](../../src/handlers/socket_flow_handler.rs)): A static instance that tracks all connected WebSocket clients and handles broadcasting updates received from `GraphService`.
-   **Force-Directed Physics**: Pre-computes node positioning with server-side physics processing within `GraphService`.
-   **WebSocket Handler ([`src/handlers/socket_flow_handler.rs`](../../src/handlers/socket_flow_handler.rs))**: Manages WebSocket connections, client registration with `ClientManager`, and message relay.

### Client-Side Components

-   **WebSocketService ([`client/src/services/WebSocketService.ts`](../../client/src/services/WebSocketService.ts))**: Handles WebSocket communication with the server.
-   **GraphDataManager ([`client/src/features/graph/managers/graphDataManager.ts`](../../client/src/features/graph/managers/graphDataManager.ts))**: Manages client-side graph data, processes incoming node position updates from `WebSocketService`, and can send user interactions (like node drags) back to the server. (This component effectively acts as the "NodeManager" in this context).
-   **GraphRenderer (Components like [`GraphManager.tsx`](../../client/src/features/graph/components/GraphManager.tsx) and [`GraphCanvas.tsx`](../../client/src/features/graph/components/GraphCanvas.tsx))**: Visualizes the graph using data from `GraphDataManager`, updating node positions as they are received.

## Key Architectural Improvements

### 1. Independent Graph Initialization

The graph is now initialized once at server startup, regardless of client connections. Key benefits:

- Reduced resource utilization by avoiding redundant graph creation
- Consistent graph state across all clients
- Immediate graph availability for new client connections

```mermaid
sequenceDiagram
    participant Server
    participant GraphService
    participant Client
    
    Server->>GraphService: Initialize on startup
    GraphService->>GraphService: Pre-compute node positions
    GraphService->>GraphService: Continuously update physics
    
    Client->>Server: Connect via WebSocket
    Server->>Client: Send pre-computed graph state
    GraphService-->>Client: Stream position updates
```

### 2. Continuous Force-Directed Layout

The server now maintains a continuous physics simulation:

- Graph nodes find optimal positions before any client connects
- Reduced initial loading time for clients as layout is pre-calculated
- Physics simulation stabilizes over time, creating a more balanced visualisation

### 3. Bidirectional Synchronization

The new architecture supports true bidirectional updates:

- Server broadcasts position updates to all connected clients
- Any client can update node positions (e.g., during user interaction)
- All changes are synchronized across all clients in real-time
- Server maintains position authority for consistency

```mermaid
sequenceDiagram
    participant ClientA
    participant Server
    participant ClientB
    
    ClientA->>Server: Move node position
    Server->>Server: Apply to graph model
    Server->>ClientA: Confirm position update
    Server->>ClientB: Broadcast position update
```

### 4. Optimized Data Transfer

The system includes several optimizations:

- Selective updates: Only nodes that change significantly trigger updates
- Position deadbanding: Filters out minor position changes
- Automatic compression for larger messages
- Dynamic update rate based on graph activity level

## Implementation Details

### Server-Side Physics Processing

The server uses a hybrid approach to physics processing:

1.  GPU-accelerated computing when available, primarily via CUDA, managed by [`src/utils/gpu_compute.rs`](../../src/utils/gpu_compute.rs). WebGPU is not the primary target for server-side GPU compute in this context.
2.  CPU fallback for physics calculations within `GraphService::calculate_layout_cpu` if GPU is not available or disabled.
3.  Physics parameters (from `AppFullSettings.visualisation.physics` and `SimulationParams`) are tuned for stability and performance.

### Client Connection Lifecycle

When a client connects:
1. The server sends the complete, settled graph state (metadata, node positions, edge data)
2. The client renders the initial state
3. The server begins streaming position updates
4. The client can send position updates to the server
5. The server broadcasts these changes to all other clients

## Performance Benefits

- **Reduced CPU/GPU usage**: Physics calculations shared across all clients
- **Lower bandwidth usage**: Only changed positions are transmitted
- **Faster initialization**: Clients receive pre-computed positions
- **Better scalability**: Multiple clients supported with minimal additional resource usage

## Future Improvements

- Real-time collaborative editing of graph content
- Conflict resolution for simultaneous node edits
- Region-based updates for very large graphs
- Client-specific view customizations