# WebSocket API Reference

## Connection

Connect to: `wss://your-domain/wss`

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

### Binary Messages

Node position updates are sent as binary messages:

- Each node update is 26 bytes
- Format: [Node ID (2 bytes)][Position (12 bytes)][Velocity (12 bytes)]
- Position and Velocity are float32[3] arrays

### Settings Synchronization

```json
{
  "type": "settings_update",
  "category": "visualization",
  "settings": {
    "edges": {
      "scaleFactor": 2.0
    }
  }
}
```

## Error Handling

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

## Rate Limiting

- 60 messages per minute per connection
- Binary updates don't count towards rate limit
