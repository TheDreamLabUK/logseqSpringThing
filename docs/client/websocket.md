# WebSocket Communication

This document describes the WebSocket communication system used in the client.

## Overview

The client uses WebSocket connections for real-time communication with the server, particularly for:
- Binary position updates for graph nodes
- Graph data synchronization
- Event notifications
- Connection status management

## Architecture

```mermaid
flowchart TB
    subgraph Client
        WebSocketService[WebSocket Service]
        GraphDataManager[Graph Data Manager]
        VisualisationManager[Visualisation Manager]
        BinaryProtocol[Binary Protocol Handler]
    end
    
    subgraph Server
        WSServer[WebSocket Server]
        Physics[Physics Engine]
        DataSync[Data Sync]
    end
    
    WebSocketService <--> WSServer
    WebSocketService --> BinaryProtocol
    BinaryProtocol --> GraphDataManager
    GraphDataManager --> VisualisationManager
    Physics --> WSServer
    DataSync --> WSServer
```

## WebSocket Service

The WebSocket service (`client/websocket/websocketService.ts`) is implemented as a singleton that manages:
- Connection establishment and maintenance
- Message handling
- Binary protocol processing
- Error handling and recovery

### Key Features

- Automatic reconnection with exponential backoff
- Binary message support
- Connection status monitoring
- Event-based message handling

## Binary Protocol

The binary protocol is used for efficient transmission of node position updates.

### Message Format

Position updates use a binary format where each node's data is packed as follows:

```
| Field    | Type        | Size (bytes) | Description           |
|----------|-------------|--------------|------------------------|
| Node ID  | uint16      | 2           | Unique node identifier |
| Position | float32[3]  | 12          | X, Y, Z coordinates    |
| Velocity | float32[3]  | 12          | VX, VY, VZ components |
```

Total bytes per node: 26 bytes

### Processing Flow

```mermaid
sequenceDiagram
    participant Server
    participant WebSocket
    participant BinaryHandler
    participant GraphManager
    participant Visualisation
    
    Server->>WebSocket: Binary Message
    WebSocket->>BinaryHandler: Process ArrayBuffer
    BinaryHandler->>GraphManager: Update Node Positions
    GraphManager->>Visualisation: Trigger Update
```

## Message Types

The WebSocket service handles several types of messages:

1. **Binary Position Updates**
   - Format: ArrayBuffer
   - Handler: `onBinaryMessage`
   - Used for real-time node position updates

2. **Connection Status**
   - Format: JSON
   - Handler: `onConnectionStatusChange`
   - Used for connection state management

## Error Handling

The WebSocket service implements robust error handling, primarily by logging errors and attempting reconnection.

### Recovery Strategy

```mermaid
stateDiagram-v2
    [*] --> Connected
    Connected --> Disconnected: Connection Lost
    Disconnected --> Retrying: Auto Reconnect
    Retrying --> Connected: Success
    Retrying --> Failed: Max Retries
    Failed --> [*]: Fatal Error
    Retrying --> Disconnected: Retry Failed
```

## Configuration

WebSocket behavior can be configured through settings:

```typescript
interface WebSocketSettings {
    reconnectAttempts: number;    // Maximum reconnection attempts (e.g., from system.websocket.reconnectAttempts)
    reconnectDelay: number;       // Base delay between retries in ms (e.g., from system.websocket.reconnectDelay)
    // binaryChunkSize, compressionEnabled, compressionThreshold, updateRate are also relevant
    // but their direct mapping to the store might be via other system.websocket.* paths or defaults.
}
```

## Performance Considerations

1. **Binary Protocol**
   - Reduces message size by ~60% compared to JSON
   - Minimizes parsing overhead
   - Enables efficient batch updates

2. **Message Batching**
   - Position updates are batched for efficiency
   - Configurable batch size and update rate
   - Automatic throttling under high load

3. **Connection Management**
   - Heartbeat mechanism for connection health
   - Automatic reconnection with backoff
   - Connection status monitoring

## Usage Example

```typescript
// Initialize WebSocket service
const ws = WebSocketService.getInstance();

// Subscribe to binary updates
ws.onBinaryMessage((data) => {
    if (data instanceof ArrayBuffer) {
        graphDataManager.updateNodePositions(new Float32Array(data));
    }
});

// Handle connection status
ws.onConnectionStatusChange((connected) => {
    if (connected) {
        graphDataManager.setBinaryUpdatesEnabled(true);
    }
});

// Connect to server
await ws.connect();
```

## Related Documentation

- [State Management](state.md) - State management integration
- [Graph Data](graph.md) - Graph data structure and updates
- [Performance](performance.md) - Performance optimization details