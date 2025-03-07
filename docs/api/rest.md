# REST API Reference

## Authentication

### Login
```http
POST /api/auth/nostr
```

**Request Body:**
```json
{
  "pubkey": "your_public_key",
  "signature": "signed_challenge"
}
```

**Response:**
```json
{
  "user": {
    "pubkey": "user_public_key",
    "npub": "user_npub",
    "is_power_user": boolean,
    "last_seen": 1234567890
  },
  "token": "session_token",
  "expires_at": 1234567890,
  "features": ["feature1", "feature2"]
}
```

### Verify Token
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

### Logout
```http
DELETE /api/auth/nostr
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

## Graph API

### Get Graph Data
```http
GET /api/graph/data
```

Returns complete graph structure.

### Get Paginated Graph Data
```http
GET /api/graph/data/paginated
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 100)
- `sort`: Sort field
- `filter`: Filter expression

### Update Graph
```http
POST /api/graph/update
```

**Request Body:**
```json
{
  "nodes": [...],
  "edges": [...]
}
```

### Refresh Graph
```http
POST /api/graph/refresh
```

## Settings API

### Get Visualization Settings
```http
GET /api/user-settings/visualization
```

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

### Perplexity Query
```http
POST /api/perplexity
```

**Request Body:**
```json
{
  "query": "Your question here",
  "conversation_id": "optional-previous-conversation-id"
}
```

**Response:**
```json
{
  "answer": "The response from Perplexity AI",
  "conversation_id": "conversation-id-for-follow-up-queries"
}
```

## Error Responses

All endpoints may return the following error responses:

#### 400 Bad Request
```json
{
  "error": "bad_request",
  "message": "Invalid parameters",
  "details": {
    "field": "reason for error"
  }
}
```

#### 401 Unauthorized
```json
{
  "error": "unauthorized",
  "message": "Invalid or missing authentication token"
}
```

#### 404 Not Found
```json
{
  "error": "not_found",
  "message": "Resource not found"
}
```

#### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An internal error occurred",
  "request_id": "abc-123"
}
```

## Related Documentation
- [WebSocket API](./websocket.md)
- [Development Setup](../development/setup.md)
- [Technical Architecture](../overview/architecture.md)