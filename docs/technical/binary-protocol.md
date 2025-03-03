# WebSocket Binary Protocol

This document describes the binary protocol used for efficient real-time updates of node positions and velocities over WebSockets.

## Overview

The binary protocol is designed to minimize bandwidth usage while providing fast updates for node positions and velocities in the 3D visualization. The protocol uses a fixed-size format for each node to simplify parsing and ensure consistency.

## Protocol Format

Each binary message consists of a series of node updates, where each node update is exactly 26 bytes:

| Field    | Type      | Size (bytes) | Description                       |
|----------|-----------|--------------|-----------------------------------|
| Node ID  | uint16    | 2            | Unique identifier for the node    |
| Position | float32[3]| 12           | X, Y, Z coordinates               |
| Velocity | float32[3]| 12           | X, Y, Z velocity components       |

Total: 26 bytes per node

## Compression

For large updates (more than 1KB), the binary data is compressed using zlib compression. The client automatically detects and decompresses these messages using the pako library.

## Server-Side Only Fields

The server maintains additional data for each node that is not transmitted over the wire:

- `mass` (u8): Node mass used for physics calculations 
- `flags` (u8): Bit flags for node properties
- `padding` (u8[2]): Reserved for future use

These fields are used for server-side physics calculations and GPU processing but are not transmitted to clients to optimize bandwidth.

## Flow Sequence

1. Client connects to WebSocket endpoint (`/wss`)
2. Server sends a text message: `{"type": "connection_established"}`
3. Client sends a text message: `{"type": "requestInitialData"}`
4. Server starts sending binary updates at regular intervals (configured by `binary_update_rate` setting)
5. Server sends a text message: `{"type": "updatesStarted"}`
6. Client processes binary updates and updates the visualization

## Error Handling

If a binary message has an invalid size (not a multiple of 26 bytes), the client will log an error and discard the message. The server includes additional logging to help diagnose issues with binary message transmission.

## Implementation Notes

- All numeric values use little-endian byte order
- Position and velocity are represented as:
  - Server-side: `Vec3Data` objects with x, y, z properties
  - Client-side: THREE.Vector3 objects
  - Wire format: float32[3] arrays for efficient binary transmission
- Binary conversion takes place at the protocol boundary only

## Data Type Handling

### Server-side (Rust)

```rust
// Vec3Data object representation
struct Vec3Data {
    x: f32,
    y: f32,
    z: f32,
}
```

### Client-side (TypeScript)

```typescript
// Conversion from binary to Vector3
function decodeNodeData(view: DataView, offset: number): NodeData {
    const position = new THREE.Vector3(
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true)
    );
    // ...
}
```

## Debugging

To enable WebSocket debugging:

1. Set `system.debug.enabled = true` in settings.yaml
2. Set `system.debug.enable_websocket_debug = true` in settings.yaml

This will enable detailed logging of WebSocket messages on both client and server.

## Recent Optimizations

The binary protocol was recently optimized to:

1. **Reduce Message Size**: Changed Node ID from uint32 (4 bytes) to uint16 (2 bytes), reducing each node's size from 28 to 26 bytes (7% reduction)

2. **Simplify Processing**: Removed headers with version numbers, sequence numbers and timestamps to reduce overhead

3. **Improve Type Consistency**: Ensured consistent use of structured vector objects through the entire pipeline:
   - Server: Vec3Data objects with x, y, z properties
   - Wire format: Compact binary arrays for transmission
   - Client: Direct conversion to THREE.Vector3 objects
   - GPU: Helper functions for array conversion when needed for CUDA