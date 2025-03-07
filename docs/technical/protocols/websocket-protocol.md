# WebSocket Protocol Specification

## Overview
The WebSocket implementation in LogseqXR provides real-time graph updates using an optimized binary protocol.

## Binary Protocol Format

Each binary message consists of node updates, where each node update is exactly 26 bytes:

| Field    | Type      | Size (bytes) | Description                       |
|----------|-----------|--------------|-----------------------------------|
| Node ID  | uint16    | 2            | Unique identifier for the node    |
| Position | float32[3]| 12           | X, Y, Z coordinates               |
| Velocity | float32[3]| 12           | X, Y, Z velocity components       |

## Connection Flow

1. Client connects to WebSocket endpoint (`/wss`)
2. Server sends: `{"type": "connection_established"}`
3. Client sends: `{"type": "requestInitialData"}`
4. Server begins binary updates (configured by `binary_update_rate`)
5. Server sends: `{"type": "updatesStarted"}`

## Optimization Features

- Zlib compression for messages >1KB
- Fixed-size format for efficient parsing
- No message headers to minimize overhead
- Consistent use of THREE.Vector3 throughout

## Diagnostics

### Common Issues

1. Connection Issues
   - Mixed Content: Ensure WebSocket uses WSS with HTTPS
   - CORS: Check server configuration for cross-origin
   - Proxy/Firewall: Verify WebSocket ports are open

2. Binary Protocol Issues
   - Message Size: Verify 26 bytes per node
   - Data Integrity: Validate Vector3 data

### Diagnostic Tools

```typescript
// Run comprehensive diagnostics
WebSocketDiagnostics.runDiagnostics();

// Test API connectivity
WebSocketDiagnostics.testApiConnectivity();

// Validate vector data
WebSocketDiagnostics.validateVectorData();
```

## Error Handling

- Connection failures trigger automatic reconnection
- Invalid messages are logged and skipped
- Server-side validation prevents corrupt data transmission