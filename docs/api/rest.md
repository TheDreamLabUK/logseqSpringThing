# REST API Reference

## Overview
The REST API provides endpoints for graph data management, content operations, and system status.

## Base URL
```
http://localhost:4000/api
```
(or deployment-dependent, e.g., `https://api.webxr.dev/v1`)

## Authentication

All API requests primarily use Nostr authentication.

#### Login
```http
POST /api/auth/nostr
```

**Request Body:**
```json
{
  "event": {
    "id": "event_id",
    "pubkey": "your_public_key",
    "created_at": 1678886400,
    "kind": 22242,
    "tags": [
      ["challenge", "random_challenge_string"],
      ["relay", "wss://relay.damus.io"]
    ],
    "content": "Login to LogseqXR",
    "sig": "event_signature"
  }
}
```

**Response:**
```json
{
  "user": {
    "pubkey": "user_public_key",
    "npub": "user_npub",
    "is_power_user": boolean,
    "features": ["feature1", "feature2"]
  },
  "token": "session_token",
  "expires_at": 1234567890
}
```

#### Verify Token
```http
POST /api/auth/nostr/verify
```

**Request Body:**
```json
{
  "pubkey": "your_public_key",
  "token": "your_token"
}
```

#### Logout
```http
DELETE /api/auth/nostr
```

## Graph API

### Get Graph Data
```http
GET /api/graph/data
```

Returns complete graph structure:
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {...}
}
```

### Get Paginated Graph Data
```http
GET /api/graph/data/paginated
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 100)
- `sort`: Sort field
- `filter`: Filter expression

**Response:**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {},
  "totalPages": 0,
  "currentPage": 1,
  "totalItems": 0,
  "pageSize": 100
}

### Update Graph
```http
POST /api/graph/update
```

This endpoint triggers a rebuild of the graph from the current metadata. It does not accept a request body for direct node/edge updates. Client-side node position updates are handled via the WebSocket API.

### Refresh Graph
```http
POST /api/graph/refresh
```

## Files API

### Process Files
```http
POST /api/files/process
```

Triggers fetching and processing of Markdown files.

**Response:**
```json
{
  "status": "success",
  "processed_files": ["file1.md", "file2.md"]
}
```

### Get File Content
```http
GET /api/files/get_content/{filename}
```

### Upload Content
```http
POST /api/files/upload
```

This endpoint is currently not implemented in the server. File content is primarily managed via GitHub integration or direct file system access on the server.

## Settings API

### Get Public Settings
```http
GET /api/user-settings
```

Returns global/default UI settings. This endpoint does not require authentication.

### Get User-Specific Settings
```http
GET /api/user-settings/sync
```

Requires authentication. Returns user-specific UI settings. For power users, this endpoint returns and allows modification of the global UI settings.

### Get Visualisation Settings by Category
```http
GET /api/visualisation/settings/{category}
```

Returns specific visualisation settings by category. The `{category}` can be a dot-separated path for nested visualisation settings (e.g., `nodes`, `edges.color`, `physics.gravity`).

### Update API Keys
```http
POST /api/auth/nostr/api-keys
```

**Request Body:**
```json
{
  "perplexity": "api_key",
  "openai": "api_key",
  "ragflow": "api_key"
}
```

## AI Services

### RAGFlow Chat
```http
POST /api/ragflow/chat
```

**Request Body:**
```json
{
  "question": "Your question here",
  "sessionId": "optional-previous-conversation-id",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "The response from RAGFlow AI",
  "conversation_id": "conversation-id-for-follow-up-queries"
}
```


## System Status

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "metadata_count": 123,
  "nodes_count": 456,
  "edges_count": 789
}
```

## Error Responses

All endpoints may return the following error responses:

### Standard Error Format
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

### Common Error Codes
- `400`: Bad Request - Invalid parameters or request
- `401`: Unauthorized - Invalid or missing authentication token
- `403`: Forbidden - Valid token but insufficient permissions
- `404`: Not Found - Resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server-side error

## Related Documentation
- [WebSocket API](./websocket.md)
- [Development Setup](../development/setup.md)