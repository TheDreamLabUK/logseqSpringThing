# LogseqXR API Documentation

This section provides comprehensive documentation for all APIs available in LogseqXR.

## Available APIs

### REST API

The [REST API](rest.md) provides HTTP endpoints for managing files, graph data, settings, and more. It's primarily used for:

- File operations (fetching, processing, updating)
- Graph management
- Settings configuration
- Health monitoring
- Authentication

### WebSocket API

The [WebSocket API](websocket.md) enables real-time communication between the client and server. It's used for:

- Real-time node position updates
- Binary data streaming for efficient 3D visualization
- Connection management
- Settings synchronization

## Authentication

Both APIs use the same authentication mechanism based on Nostr. Include the authentication token in the Authorization header:

```
Authorization: Bearer <token>
```

## API Versioning

The current API version is v1. All endpoints are prefixed with `/api/v1/` except for WebSocket connections which use `/wss`.

## Error Handling

All APIs follow a consistent error format:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "specific error information"
  }
}
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse. The current limits are:

- REST API: 100 requests per minute per IP
- WebSocket: 60 messages per minute per connection

## Further Reading

- [Binary Protocol](../technical/binary-protocol.md) - Details on the binary format used for WebSocket communication
- [WebSocket Implementation](../technical/websockets.md) - In-depth documentation on the WebSocket implementation
