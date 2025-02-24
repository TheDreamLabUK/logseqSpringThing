# WebSocket API Documentation

## Connection

Connect to the WebSocket endpoint at: `ws://localhost:4000/ws` (or your configured domain)

### Connection Flow

1. Client initiates WebSocket connection
2. Server sends connection confirmation:
```json
{
  "type": "connection_established",
  "timestamp": 1708598400000
}
```

## Message Types

### Text Messages

#### 1. Ping/Pong
Used for connection health monitoring.

**Ping Request:**
```json
{
  "type": "ping",
  "timestamp": 1708598400000
}
```

**Pong Response:**
```json
{
  "type": "pong",
  "timestamp": 1708598400000
}
```

#### 2. Initial Data Request
Request to start receiving node position updates.

**Request:**
```json
{
  "type": "requestInitialData"
}
```

**Response:**
```json
{
  "type": "updatesStarted",
  "timestamp": 1708598400000
}
```

### Binary Messages

Binary messages use an optimized protocol for efficient transmission of node position and velocity updates. The protocol uses Vec3Data for consistent vector representation across the stack.

#### Message Format

```
[4 bytes] message_type (u32)
[4 bytes] node_count (u32)
For each node:
  [4 bytes] node_id (u32)
  [12 bytes] Position (Vec3Data)
    - [4 bytes] x (f32)
    - [4 bytes] y (f32)
    - [4 bytes] z (f32)
  [12 bytes] Velocity (Vec3Data)
    - [4 bytes] vx (f32)
    - [4 bytes] vy (f32)
    - [4 bytes] vz (f32)
```

#### Message Types
- 0x01: PositionVelocityUpdate - Real-time updates of node positions and velocities

#### Size Calculations
- Header: 8 bytes
- Per node: 28 bytes (4 + 12 + 12)
- Total message size: 8 + (28 × number_of_nodes) bytes

### Compression

Binary messages may be compressed using zlib compression when:
1. Compression is enabled in settings
2. Message size exceeds the compression threshold
3. Compressed size is smaller than uncompressed size

## Update Flow

1. Client connects and sends `requestInitialData`
2. Server begins sending binary position updates at configured rate (default: 30 Hz)
3. Client can send position updates for up to 2 nodes during interaction
4. Server broadcasts updated positions to all connected clients

## Error Handling

### Connection Errors

```json
{
  "error": "WebSocket upgrade required",
  "message": "This endpoint requires a WebSocket connection"
}
```

### Message Parse Errors

```json
{
  "type": "error",
  "error": "parse_error",
  "message": "Failed to parse message"
}
```

### Protocol Errors

```json
{
  "type": "error",
  "error": "protocol_error",
  "message": "Invalid message format"
}
```

## Performance Considerations

### Network Optimization
- Fixed-size binary format reduces overhead
- Optional compression for large messages
- Efficient buffer allocation
- Direct Vec3 memory layout
- Zero-copy potential for Vec3 data

### Client-side Processing
- Fixed message format enables efficient parsing
- Direct TypedArray access for binary data
- No JSON parsing overhead for position updates
- Consistent Vec3 representation with Three.js

### Server Processing
- Optimized for high-frequency updates
- Efficient binary message generation
- Configurable update rate
- Debug mode for detailed logging
- CUDA-compatible Vec3 layout

## Configuration Options

The following settings can be configured:

```json
{
  "system": {
    "websocket": {
      "binary_update_rate": 30,
      "compression_enabled": true,
      "compression_threshold": 1024
    },
    "debug": {
      "enabled": false,
      "enable_websocket_debug": false
    }
  }
}
```

## Related Documentation
- [Vec3 Alignment](../technical/vec3-alignment.md)
- [Binary Protocol Details](../technical/binary-protocol.md)
- [Performance Optimizations](../technical/performance.md)
- [REST API](./rest.md)