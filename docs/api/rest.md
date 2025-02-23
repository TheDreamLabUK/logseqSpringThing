# REST API Documentation

This document details the REST API endpoints provided by LogseqXR.

## Base URL
All endpoints are relative to: `http://localhost:4000/api` (or your configured domain)

## Authentication
Authentication is handled through Nostr. Include the authentication token in the Authorization header:
```
Authorization: Bearer <token>
```

## API Endpoints

### Files API

#### Process Files
```http
POST /files/process
```
Triggers fetching and processing of Markdown files from GitHub.

**Response**
```json
{
  "status": "success",
  "processed_files": ["file1.md", "file2.md"]
}
```

#### Get File Content
```http
GET /files/get_content/{filename}
```
Retrieves the raw content of a specified Markdown file.

**Parameters**
- `filename`: The name of the file to retrieve

**Response**
```json
{
  "status": "error",
  "message": "File not found or unreadable: filename.md"
}
```

#### Refresh Graph
```http
POST /files/refresh_graph
```
Rebuilds the graph data structure from current metadata.

**Response**
```json
{
  "status": "success",
  "message": "Graph refreshed successfully"
}
```

#### Update Graph
```http
POST /files/update_graph
```
Forces an update of graph nodes and edges based on newly processed files.

**Response**
```json
{
  "status": "success",
  "message": "Graph updated successfully"
}
```

### Graph API

#### Get Graph Data
```http
GET /graph/data
```
Returns the complete graph structure.

**Response**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {
    "file1.md": {
      // Metadata properties
    }
  }
}
```

#### Get Paginated Graph Data
```http
GET /graph/data/paginated
```
Returns paginated graph data for large datasets.

**Query Parameters**
- `page`: Page number (default: 1)
- `pageSize`: Items per page (default: 100)
- `query`: Optional search query
- `sort`: Optional sort field
- `filter`: Optional filter criteria

**Response**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {},
  "totalPages": 10,
  "currentPage": 1,
  "totalItems": 1000,
  "pageSize": 100
}
```

### Settings API

The settings API provides access to all system configuration options through a hierarchical structure of categories and settings.

#### Available Categories
- visualization.nodes
- visualization.edges
- visualization.rendering
- visualization.labels
- visualization.bloom
- visualization.animations
- visualization.physics
- visualization.hologram
- system.network
- system.websocket
- system.security
- system.debug
- xr
- github
- ragflow
- perplexity
- openai

#### Get Setting Value
```http
GET /settings/{category}/{setting}
```
Retrieves the current value of a particular setting.

**Parameters**
- `category`: Setting category (e.g., "visualization.nodes", "system.network")
- `setting`: Specific setting name

**Response**
```json
{
  "category": "visualization.nodes",
  "setting": "size",
  "value": 1.5,
  "success": true,
  "error": null
}
```

#### Update Setting
```http
PUT /settings/{category}/{setting}
```
Updates a setting value.

**Request Body**
```json
{
  "value": 1.5
}
```

**Response**
```json
{
  "category": "visualization.nodes",
  "setting": "size",
  "value": 1.5,
  "success": true,
  "error": null
}
```

#### Get Category Settings
```http
GET /settings/{category}
```
Returns all settings within a given category.

**Parameters**
- `category`: Category name (e.g., "visualization.nodes", "system.network")

**Response**
```json
{
  "category": "visualization.nodes",
  "settings": {
    "size": 1.0,
    "color": "#007bff",
    "opacity": 0.8,
    "visible": true
  },
  "success": true,
  "error": null
}
```

### Perplexity AI API

The Perplexity AI API provides natural language query capabilities with conversation support.

#### Query Perplexity
```http
POST /perplexity
```
Send queries to Perplexity AI and maintain conversation context.

**Request Body**
```json
{
  "query": "Your question here",
  "conversation_id": "optional-previous-conversation-id"
}
```

**Response**
```json
{
  "answer": "The response from Perplexity AI",
  "conversation_id": "conversation-id-for-follow-up-queries"
}
```

**Error Response**
```json
{
  "error": "Perplexity service is not available"
}
```

### Authentication API

#### Login
```http
POST /auth/nostr
```
Authenticate using Nostr credentials.

**Request Body**
```json
{
  "pubkey": "your_public_key",
  "signature": "signed_challenge"
}
```

#### Logout
```http
DELETE /auth/nostr
```
End the current session.

#### Verify Session
```http
POST /auth/nostr/verify
```
Verify the current session is valid.

#### Refresh Token
```http
POST /auth/nostr/refresh
```
Refresh the authentication token.

#### Get API Keys
```http
GET /auth/nostr/api-keys
```
Retrieve the user's API keys.

#### Update API Keys
```http
POST /auth/nostr/api-keys
```
Update the user's API keys.

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