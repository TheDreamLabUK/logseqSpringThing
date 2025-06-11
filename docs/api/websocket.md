# WebSocket API Reference

## Overview
The WebSocket implementation in LogseqXR provides real-time graph updates using an optimized binary protocol.

## Connection

Connect to: `wss://your-domain/wss` (Note: The actual path is `/wss` as handled by `src/handlers/socket_flow_handler.rs`)

### Connection Flow
1. Client connects to WebSocket endpoint (`/wss`)
2. Server sends: `{"type": "connection_established", "timestamp": <timestamp>}`
3. Client sends authentication (if required, typically handled via HTTP session before WebSocket upgrade)
4. Client sends: `{"type": "requestInitialData"}`
5. Server begins binary updates (configured by `binary_update_rate`)
6. Server sends: `{"type": "updatesStarted", "timestamp": <timestamp>}`
7. Server sends: `{"type": "loading", "message": "Calculating initial layout..."}` (if applicable)

## Authentication

Authentication for WebSocket connections in LogseqXR is primarily handled during the initial HTTP handshake that upgrades to a WebSocket connection. This means that user authentication (e.g., via Nostr) should occur before or during the establishment of the WebSocket connection, typically through standard HTTP mechanisms (like cookies or authorization headers). The server's `socket_flow_handler.rs` does not process an explicit `{"type": "auth", "token": "..."}` message over the WebSocket itself.

## Message Types

### Control Messages

#### 1. Connection Established
```json
{
  "type": "connection_established",
  "timestamp": 1679417762000
}
```

#### 2. Request Initial Data
```json
{
  "type": "requestInitialData"
}
```

#### 3. Updates Started
```json
{
  "type": "updatesStarted",
  "timestamp": 1679417763000
}
```

#### 4. Loading State
```json
{
  "type": "loading",
  "message": "Calculating initial layout..."
}
```

### Binary Messages - Position Updates

Position updates are transmitted as binary messages in both directions:

- **Each node update is 28 bytes**.
- Format: **Node ID (u32, 4 bytes)**, Position (3x f32, 12 bytes), Velocity (3x f32, 12 bytes). Total: 28 bytes per node.
- Position and Velocity are three consecutive `f32` values (x, y, z).
- Server-side `BinaryNodeData` (defined in `src/utils/socket_flow_messages.rs`) includes additional fields like `mass`, `flags`, and `padding` for physics simulation, but these are **not** part of the **28-byte** wire format sent to the client.
- The client-side `BinaryNodeData` (defined in `client/src/types/binaryProtocol.ts`) and the server-side `WireNodeDataItem` in `binary_protocol.rs` correctly reflect the **28-byte** wire format: `nodeId`, `position`, `velocity`.
- Server-side compression (zlib) is applied... Client-side decompression is handled by the `graph.worker.ts` off the main UI thread.

#### Server → Client Updates

The server continuously sends position updates to all connected clients:

1. Updates are pre-computed by the server's continuous physics engine
2. Only nodes that changed significantly are included
3. Update frequency varies based on graph activity (5-60 updates/sec)
4. Each update can contain multiple node positions in a single binary message
5. When the physics simulation stabilizes, update frequency is reduced

#### Client → Server Updates

Clients can send position updates back to the server:

1. Position updates use the same binary format as server messages
2. Updates are processed by the server's physics system
3. Changes are validated and broadcast to all other connected clients
4. Modifications that violate physics constraints may be adjusted by the server

### Position Synchronization Protocol

The bidirectional synchronization protocol ensures consistent graph state:

1. Server maintains the authoritative graph state
2. Any client can send position updates during user interaction
3. Server processes updates and applies physics constraints
4. All clients receive the same set of position updates
5. Late-joining clients receive the complete current graph state


## Control Messages (JSON) - Revisited

The `socket_flow_handler.rs` primarily handles the following JSON messages:

**Server -> Client:**
- `{"type": "connection_established", "timestamp": <timestamp>}`
- `{"type": "updatesStarted", "timestamp": <timestamp>}`
- `{"type": "loading", "message": "Calculating initial layout..."}`
- `{"type": "pong"}` (in response to client's ping)

**Client -> Server:**
- `{"type": "ping"}`
- `{"type": "requestInitialData"}`: This message implicitly starts the binary update stream if the server is ready.
- `{"type": "subscribe_position_updates", "binary": true, "interval": <number>}`: While not a distinct message type in the server's `Message` enum (`src/utils/socket_flow_messages.rs`), the `requestInitialData` handler in `socket_flow_handler.rs` implicitly starts the binary update stream. The client can send a `subscribe_position_updates` message to configure the stream, but the server's primary trigger is `requestInitialData`.
- `{"type": "enableRandomization", "enabled": <boolean>}`: This message is acknowledged by the server, but server-side randomization has been removed. The client is responsible for any randomization effects.

## Optimization Features

- Zlib compression for binary messages larger than `compression_threshold` (default 512 bytes, configurable).
- Fixed-size binary format (28 bytes per node update) for efficient parsing.
- Minimal overhead for binary messages (no explicit headers per node update within a batch).
- Consistent use of `THREE.Vector3` for positions and velocities on the client-side.

## Error Handling

The `socket_flow_handler.rs` does not explicitly send these structured JSON error messages.
- Errors encountered during WebSocket communication (e.g., deserialization issues, unexpected message types) are typically logged on the server-side.
- The WebSocket connection might be closed by the server if unrecoverable errors occur.
- Clients should implement their own timeout and error detection logic for the WebSocket connection itself (e.g., detecting a closed connection).

## Rate Limiting

This section refers to the server's dynamic management of binary position update frequency and client-side handling, rather than strict message rate limiting (e.g., X messages per second).

- **Server-Side Update Rate:** The server dynamically adjusts the rate of binary position updates based on graph activity and physics simulation stability. This is controlled by settings in `settings.yaml` under `system.websocket`:
    - `min_update_rate`: Minimum updates per second when the graph is stable.
    - `max_update_rate`: Maximum updates per second during high activity.
    - `motion_threshold`: Sensitivity to node movement for determining activity.
- **Client-Side Throttling:** The client (`client/src/features/graph/managers/graphDataManager.ts`) implements a `lastBinaryUpdateTime` check to avoid processing updates too rapidly if they arrive faster than the client can render, effectively throttling the application of received binary messages.
- **Debug Logging:** `socket_flow_handler.rs` includes a `DEBUG_LOG_SAMPLE_RATE` to control how frequently detailed debug logs about message handling are produced, which is a diagnostic aid rather than a rate limit.

## Diagnostics

### Common Issues

1. Connection Issues
   - Mixed Content: Ensure WebSocket uses WSS with HTTPS
   - CORS: Check server configuration for cross-origin
   - Proxy/Firewall: Verify WebSocket ports are open

2. Binary Protocol Issues
   - Message Size: Verify 28 bytes per node
   - Data Integrity: Validate Vector3 data
