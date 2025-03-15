# REST API Endpoints

This document provides comprehensive documentation of the REST API endpoints used by the client to communicate with the server. Each endpoint is fully documented with its URL, method, parameters, request format, response format, error handling, and example usage.

## API Base URL

The API base URL is constructed dynamically based on the environment:

```typescript
// From client/core/api.ts
export function buildApiUrl(path: string): string {
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    // Check if we're in production (any visionflow.info domain)
    const isProduction = host.endsWith('visionflow.info');
    const base = isProduction 
        ? `${protocol}//${host}`
        : `${protocol}//${host}:4000`;
    return `${base}${path}`;
}
```

- Production: `https://[hostname]/api/...`
- Development: `http://[hostname]:4000/api/...`

## Authentication

Many API endpoints require authentication. The client handles this through the `getAuthHeaders` function:

```typescript
// From client/core/api.ts
export function getAuthHeaders(): HeadersInit {
    const headers: HeadersInit = {
        'Content-Type': 'application/json'
    };
    
    const pubkey = localStorage.getItem('nostr_pubkey');
    const token = localStorage.getItem('nostr_token');
    if (pubkey && token) {
        headers['X-Nostr-Pubkey'] = pubkey;
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}
```

Authentication is handled through Nostr authentication tokens, which are included in the request headers.

## API Endpoints

### Graph Data Endpoints

#### Get Graph Data

Retrieves the complete graph data structure.

**Endpoint**: `/api/graph/data`
**Method**: `GET`
**Authentication**: Required

**Response Format**:
```json
{
  "nodes": [
    {
      "id": "1",
      "data": {
        "position": {"x": 0, "y": 0, "z": 0},
        "velocity": {"x": 0, "y": 0, "z": 0},
        "metadata": {
          "name": "Node Name",
          "lastModified": 1615478982,
          "links": ["2", "3"],
          "references": ["ref1", "ref2"],
          "fileSize": 1024,
          "hyperlinkCount": 5
        }
      }
    }
  ],
  "edges": [
    {
      "source": "1",
      "target": "2",
      "data": {
        "weight": 1,
        "type": "reference"
      }
    }
  ],
  "metadata": {
    "timestamp": 1615478982,
    "version": "1.0"
  }
}
```

**Error Responses**:
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### Get Paginated Graph Data

Retrieves graph data with pagination support for handling large datasets.

**Endpoint**: `/api/graph/data/paginated`
**Method**: `GET`
**Authentication**: Required

**Query Parameters**:
- `page` (number, required): Page number, starting from 1
- `pageSize` (number, optional): Number of items per page, default 100

**Response Format**:
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {...},
  "totalPages": 10,
  "currentPage": 1,
  "totalItems": 987,
  "pageSize": 100
}
```

**Error Responses**:
- `400 Bad Request`: Invalid pagination parameters
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### Update Graph

Updates graph data on the server.

**Endpoint**: `/api/graph/update`
**Method**: `POST`
**Authentication**: Required

**Request Body**:
```json
{
  "nodes": [
    {
      "id": "1",
      "data": {
        "position": {"x": 10, "y": 5, "z": 3},
        "velocity": {"x": 0, "y": 0, "z": 0}
      }
    }
  ]
}
```

**Response Format**:
```json
{
  "success": true,
  "updatedCount": 1,
  "timestamp": 1615479082
}
```

**Error Responses**:
- `400 Bad Request`: Invalid update data
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

### Settings Endpoints

#### Get User Settings

Retrieves the user's settings.

**Endpoint**: `/api/user-settings`
**Method**: `GET`
**Authentication**: Required

**Response Format**:
```json
{
  "visualization": {
    "nodes": {
      "size": 1,
      "color": "#4CAF50"
    },
    "edges": {
      "thickness": 0.25,
      "color": "#E0E0E0"
    },
    "labels": {
      "visible": true,
      "size": 1,
      "color": "#FFFFFF",
      "visibilityThreshold": 0.5
    }
  },
  "physics": {
    "enabled": true,
    "gravity": 0.1,
    "friction": 0.1
  },
  "network": {
    "reconnectDelay": 1000
  }
}
```

**Error Responses**:
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### Update User Settings

Updates the user's settings.

**Endpoint**: `/api/user-settings`
**Method**: `POST`
**Authentication**: Required

**Request Body**:
```json
{
  "visualization": {
    "nodes": {
      "size": 1.5,
      "color": "#FF4444"
    }
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "timestamp": 1615479082
}
```

**Error Responses**:
- `400 Bad Request`: Invalid settings data
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### Get Visualization Settings

Retrieves visualization-specific settings.

**Endpoint**: `/api/user-settings/visualization`
**Method**: `GET`
**Authentication**: Required

**Response Format**:
```json
{
  "nodes": {
    "size": 1,
    "color": "#4CAF50"
  },
  "edges": {
    "thickness": 0.25,
    "color": "#E0E0E0"
  },
  "labels": {
    "visible": true,
    "size": 1,
    "color": "#FFFFFF",
    "visibilityThreshold": 0.5
  }
}
```

**Error Responses**:
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### Update Visualization Settings

Updates visualization-specific settings.

**Endpoint**: `/api/user-settings/visualization`
**Method**: `POST`
**Authentication**: Required

**Request Body**:
```json
{
  "nodes": {
    "size": 1.5,
    "color": "#FF4444"
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "timestamp": 1615479082
}
```

**Error Responses**:
- `400 Bad Request`: Invalid settings data
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

### WebSocket Control Endpoints

#### Get WebSocket Settings

Retrieves WebSocket connection settings.

**Endpoint**: `/api/settings/websocket`
**Method**: `GET`
**Authentication**: Required

**Response Format**:
```json
{
  "reconnectDelay": 1000,
  "maxReconnectAttempts": 5,
  "compressionEnabled": true,
  "binaryProtocolEnabled": true
}
```

**Error Responses**:
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### Update WebSocket Settings

Updates WebSocket connection settings.

**Endpoint**: `/api/settings/websocket`
**Method**: `POST`
**Authentication**: Required

**Request Body**:
```json
{
  "reconnectDelay": 2000,
  "compressionEnabled": false
}
```

**Response Format**:
```json
{
  "success": true,
  "timestamp": 1615479082
}
```

**Error Responses**:
- `400 Bad Request`: Invalid settings data
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

#### WebSocket Control

Controls WebSocket server behavior.

**Endpoint**: `/api/websocket/control`
**Method**: `POST`
**Authentication**: Required

**Request Body**:
```json
{
  "command": "restart",
  "parameters": {
    "clearCache": true
  }
}
```

**Response Format**:
```json
{
  "success": true,
  "message": "WebSocket server restarting",
  "timestamp": 1615479082
}
```

**Error Responses**:
- `400 Bad Request`: Invalid command
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

### File Endpoints

#### Get File Content

Retrieves file content from the server.

**Endpoint**: `/api/files/{path}`
**Method**: `GET`
**Authentication**: Required

**Path Parameters**:
- `path` (string, required): Path to the file

**Response Format**: File content with appropriate MIME type

**Error Responses**:
- `401 Unauthorized`: Authentication token is missing or invalid
- `404 Not Found`: File not found
- `500 Internal Server Error`: Server-side error occurred

### Authentication Endpoints

#### Nostr Authentication

Initiates Nostr authentication.

**Endpoint**: `/api/auth/nostr`
**Method**: `POST`
**Authentication**: Not required

**Request Body**:
```json
{
  "pubkey": "npub..."
}
```

**Response Format**:
```json
{
  "challenge": "...",
  "timestamp": 1615479082
}
```

**Error Responses**:
- `400 Bad Request`: Invalid public key
- `500 Internal Server Error`: Server-side error occurred

#### Verify Nostr Authentication

Verifies Nostr authentication.

**Endpoint**: `/api/auth/nostr/verify`
**Method**: `POST`
**Authentication**: Not required

**Request Body**:
```json
{
  "pubkey": "npub...",
  "signature": "...",
  "challenge": "..."
}
```

**Response Format**:
```json
{
  "success": true,
  "token": "jwt-token-here",
  "expiresAt": 1615565482
}
```

**Error Responses**:
- `400 Bad Request`: Invalid verification data
- `401 Unauthorized`: Signature verification failed
- `500 Internal Server Error`: Server-side error occurred

#### Logout

Logs out the user by invalidating the current token.

**Endpoint**: `/api/auth/nostr/logout`
**Method**: `POST`
**Authentication**: Required

**Response Format**:
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

**Error Responses**:
- `401 Unauthorized`: Authentication token is missing or invalid
- `500 Internal Server Error`: Server-side error occurred

## Error Handling

All API responses follow a consistent error format:

```json
{
  "error": true,
  "code": "RESOURCE_NOT_FOUND",
  "message": "The requested resource was not found",
  "details": {
    "resource": "file",
    "path": "/path/to/file"
  }
}
```

Common error codes:
- `UNAUTHORIZED`: Authentication required or failed
- `INVALID_REQUEST`: Request format is invalid
- `RESOURCE_NOT_FOUND`: Requested resource does not exist
- `SERVER_ERROR`: Internal server error occurred
- `VALIDATION_ERROR`: Request validation failed

## Client Implementation

In the client code, API requests are typically made using the Fetch API with helper functions:

```typescript
async function fetchGraphData(page = 1, pageSize = 100) {
  const url = buildApiUrl(`${API_ENDPOINTS.GRAPH_PAGINATED}?page=${page}&pageSize=${pageSize}`);
  const response = await fetch(url, {
    method: 'GET',
    headers: getAuthHeaders()
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch graph data: ${response.status} ${response.statusText}`);
  }
  
  return await response.json();
}
```

## API Versioning

The current API does not use explicit versioning in the URL path. Future API versions may include version numbers in the path (e.g., `/api/v2/graph/data`).

## Next Sections

For more detailed information, refer to:
- [WebSocket Protocol](websocket-protocol.md) - WebSocket communication details
- [Payload Formats](payload-formats.md) - Request/response payload formats
- [Authentication](authentication.md) - Authentication mechanisms