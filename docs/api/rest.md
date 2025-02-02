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
  "success": true,
  "processed_files": 42,
  "metadata_updated": true
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
  "content": "# File Content\n...",
  "metadata": {
    "last_modified": "2024-02-02T10:00:00Z",
    "size": 1024,
    "sha": "abc123..."
  }
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
  "success": true,
  "nodes": 100,
  "edges": 250
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
  "success": true,
  "updated_nodes": 5,
  "updated_edges": 12
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
  "nodes": [
    {
      "id": "file1.md",
      "size": 1.5,
      "position": {"x": 0, "y": 0, "z": 0},
      "metadata": {
        "title": "File 1",
        "topics": ["topic1", "topic2"]
      }
    }
  ],
  "edges": [
    {
      "source": "file1.md",
      "target": "file2.md",
      "weight": 1.0
    }
  ]
}
```

#### Get Paginated Graph Data
```http
GET /graph/data/paginated?page=1&limit=50
```
Returns paginated graph data for large datasets.

**Parameters**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 50)

**Response**
```json
{
  "data": {
    "nodes": [...],
    "edges": [...]
  },
  "pagination": {
    "current_page": 1,
    "total_pages": 10,
    "total_items": 500
  }
}
```

#### Update Graph Layout
```http
POST /graph/update
```
Updates graph data using either GPU-accelerated or CPU-based computations.

**Request Body**
```json
{
  "use_gpu": true,
  "physics_params": {
    "spring_strength": 0.1,
    "repulsion": 1.0,
    "damping": 0.8
  }
}
```

### Visualization API

#### Get Setting Value
```http
GET /visualization/settings/{category}/{setting}
```
Retrieves the current value of a particular setting.

**Parameters**
- `category`: Setting category (e.g., "visualization", "system")
- `setting`: Specific setting name

**Response**
```json
{
  "value": 1.5,
  "default": 1.0,
  "type": "number",
  "description": "Node size multiplier"
}
```

#### Update Setting
```http
PUT /visualization/settings/{category}/{setting}
```
Updates a setting value.

**Request Body**
```json
{
  "value": 2.0
}
```

#### Get Category Settings
```http
GET /visualization/settings/{category}
```
Returns all settings within a given category.

**Parameters**
- `category`: Category name (e.g., "nodes", "edges", "physics")

**Response**
```json
{
  "nodes": {
    "size": 1.0,
    "color": "#007bff",
    "opacity": 0.8
  }
}
```

### Perplexity AI API (In Development)

> **Note:** The following endpoints are currently under development and will be available in upcoming releases.

#### Analyze Content
```http
POST /perplexity/analyze
```
Analyzes Markdown files for potential updates and improvements.

**Request Body**
```json
{
  "files": ["file1.md", "file2.md"],
  "analysis_type": "content_update"
}
```

**Response**
```json
{
  "analysis_results": [
    {
      "file": "file1.md",
      "suggestions": [
        {
          "type": "content_update",
          "section": "Technical Overview",
          "current_content": "...",
          "suggested_content": "...",
          "reason": "Information is outdated",
          "confidence": 0.85
        }
      ]
    }
  ]
}
```

#### Create Pull Request
```http
POST /perplexity/create-pr
```
Creates a GitHub pull request with suggested updates.

**Request Body**
```json
{
  "analysis_id": "abc123",
  "approved_suggestions": ["suggestion1", "suggestion2"]
}
```

**Response**
```json
{
  "success": true,
  "pull_request_url": "https://github.com/user/repo/pull/123",
  "updated_files": ["file1.md", "file2.md"]
}
```

#### Get Analysis Status
```http
GET /perplexity/analysis/{analysis_id}
```
Retrieves the status of a content analysis request.

**Response**
```json
{
  "status": "completed",
  "progress": 100,
  "suggestions_count": 5,
  "completion_time": "2024-02-02T10:00:00Z"
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