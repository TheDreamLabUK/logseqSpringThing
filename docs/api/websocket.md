# WebSocket API Reference

## Overview
The WebSocket implementation in LogseqXR provides real-time graph updates using an optimized binary protocol.

## Connection

Connect to: `wss://your-domain/wss`

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

- Each node update is 26 bytes
- Format: [Node ID (2 bytes)][Position (12 bytes)][Velocity (12 bytes)]
- Position and Velocity are three consecutive float32 values (x,y,z)
- Messages are compressed with zlib if size > 1KB

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


## Optimization Features

- Zlib compression for messages >1KB
- Fixed-size format for efficient parsing
- No message headers to minimize overhead (for binary messages)
- Consistent use of THREE.Vector3 throughout (client-side)

## Error Handling

### Error Message Format

#### 1. Connection Error
```json
{
  "type": "error",
  "code": "connection_error",
  "message": "Connection failed"
}
```

#### 2. Authentication Error
```json
{
  "type": "error",
  "code": "auth_error",
  "message": "Invalid token"
}
```

#### 3. Position Update Error
```json
{
  "type": "error",
  "code": "position_update_error",
  "message": "Invalid node position data"
}
```

### Error Handling Features
- Connection failures trigger automatic reconnection
- Invalid messages are logged and skipped
- Server-side validation prevents corrupt data transmission

## Rate Limiting

- Server-side throttling applies for high-frequency position updates.

## Diagnostics

### Common Issues

1. Connection Issues
   - Mixed Content: Ensure WebSocket uses WSS with HTTPS
   - CORS: Check server configuration for cross-origin
   - Proxy/Firewall: Verify WebSocket ports are open

2. Binary Protocol Issues
   - Message Size: Verify 26 bytes per node
   - Data Integrity: Validate Vector3 data
