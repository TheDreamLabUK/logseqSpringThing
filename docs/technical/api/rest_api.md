# REST API Documentation

## Overview
The REST API provides endpoints for graph data management, content operations, and system status.

## Base URL
```
https://api.webxr.dev/v1
```

## Authentication
All API requests require a JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

## Endpoints

### Graph Operations

#### GET /graph
Retrieves current graph state
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {...}
}
```

#### POST /graph/update
Updates graph data
```json
{
  "nodes": [
    {
      "id": "string",
      "position": {"x": 0, "y": 0, "z": 0},
      "mass": 1.0
    }
  ]
}
```

#### GET /graph/nodes/{nodeId}
Retrieves specific node data
```json
{
  "id": "string",
  "metadata": {...},
  "connections": [...]
}
```

### Content Operations

#### GET /content/{path}
Retrieves markdown content
```json
{
  "content": "string",
  "metadata": {...}
}
```

#### POST /content/upload
Uploads new content
```json
{
  "path": "string",
  "content": "string",
  "metadata": {...}
}
```

### System Status

#### GET /health
System health check
```json
{
  "status": "healthy",
  "services": {
    "gpu": "active",
    "github": "connected"
  }
}
```

## WebSocket API

### Connection
```
ws://api.webxr.dev/v1/ws
```

### Message Types

#### Graph Updates
```json
{
  "type": "graph_update",
  "data": {
    "nodes": [...],
    "edges": [...]
  }
}
```

#### Simulation Status
```json
{
  "type": "simulation_status",
  "data": {
    "phase": "dynamic",
    "metrics": {...}
  }
}
```

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {...}
  }
}
```

### Common Error Codes
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error