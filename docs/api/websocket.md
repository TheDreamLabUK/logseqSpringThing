# WebSocket API Reference

## Overview
The WebSocket implementation in LogseqXR provides real-time graph updates using an optimized binary protocol.

## Connection

Connect to: `wss://your-domain/wss`

### Connection Flow
1. Client connects to WebSocket endpoint (`/wss`)
2. Server sends: `{"type": "connection_established"}`
3. Client sends authentication (if required)
4. Client sends: `{"type": "requestInitialData"}`
5. Server begins binary updates (configured by `binary_update_rate`)
6. Server sends: `{"type": "updatesStarted"}`

## Authentication

Send authentication message immediately after connection:

```json
{
  "type": "auth",
  "token": "your_nostr_token"
}
```

## Message Types

### Control Messages

1. Connection Established
```json
{
  "type": "connection_established"
}
```

2. Request Initial Data
```json
{
  "type": "requestInitialData"
}
```

3. Updates Started
```json
{
  "type": "updatesStarted"
}
```

### Binary Protocol Format

Node position updates are sent as binary messages. Each binary message consists of node updates, where each node update is exactly 26 bytes:

| Field    | Type      | Size (bytes) | Description                       |
|----------|-----------|--------------|-----------------------------------|
| Node ID  | uint16    | 2            | Unique identifier for the node    |
| Position | float32[3]| 12           | X, Y, Z coordinates               |
| Velocity | float32[3]| 12           | X, Y, Z velocity components       |

### Settings Synchronization

```json
{
  "type": "settings_update",
  "category": "visualisation",
  "settings": {
    "edges": {
      "scaleFactor": 2.0
    }
  }
}
```

## Optimization Features

- Zlib compression for messages >1KB
- Fixed-size format for efficient parsing
- No message headers to minimize overhead
- Consistent use of THREE.Vector3 throughout

## Error Handling

### Error Message Format

1. Connection Error
```json
{
  "type": "error",
  "code": "connection_error",
  "message": "Connection failed"
}
```

2. Authentication Error
```json
{
  "type": "error",
  "code": "auth_error",
  "message": "Invalid token"
}
```

### Error Handling Features
- Connection failures trigger automatic reconnection
- Invalid messages are logged and skipped
- Server-side validation prevents corrupt data transmission

## Rate Limiting

- 60 messages per minute per connection
- Binary updates don't count towards rate limit

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