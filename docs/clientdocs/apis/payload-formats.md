# Payload Formats

This document provides a comprehensive specification of the payload formats used in API communications between the client and server. Well-defined payload formats are essential for ensuring consistent data exchange and interoperability between components.

## API Payload Format Specifications

### Common Format Conventions

All API payloads follow these general conventions:

1. **JSON Format**: All REST API requests and responses use JSON format
2. **Consistent Naming**: Snake_case is used for server responses, camelCase for client requests
3. **Error Format**: Standard error response format across all endpoints
4. **Timestamps**: ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)
5. **IDs**: String IDs for all resources, even if they are numeric

### Error Response Format

All API errors follow a standard format:

```json
{
  "error": true,
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    // Additional error-specific details
  }
}
```

Common error codes:
- `INVALID_REQUEST` - Request format or parameters are invalid
- `UNAUTHORIZED` - Authentication required or failed
- `FORBIDDEN` - Permission denied
- `NOT_FOUND` - Requested resource not found
- `SERVER_ERROR` - Internal server error
- `VALIDATION_ERROR` - Input validation failed

### Graph API Payloads

#### Graph Data Request

**Endpoint**: `GET /api/graph/data`

**Response Format**:
```json
{
  "nodes": [
    {
      "id": "1",
      "data": {
        "position": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "velocity": {
          "x": 0,
          "y": 0,
          "z": 0
        },
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

#### Paginated Graph Data Request

**Endpoint**: `GET /api/graph/data/paginated?page=1&pageSize=100`

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
  },
  "totalPages": 10,
  "currentPage": 1,
  "totalItems": 987,
  "pageSize": 100
}
```

#### Graph Update Request

**Endpoint**: `POST /api/graph/update`

**Request Format**:
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
  "updated_nodes": 1,
  "timestamp": 1615479082
}
```

### Settings API Payloads

#### Get Settings Request

**Endpoint**: `GET /api/user-settings`

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

#### Update Settings Request

**Endpoint**: `POST /api/user-settings`

**Request Format**:
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

### Authentication API Payloads

#### Nostr Authentication Request

**Endpoint**: `POST /api/auth/nostr`

**Request Format**:
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

#### Verify Nostr Authentication

**Endpoint**: `POST /api/auth/nostr/verify`

**Request Format**:
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
  "expires_at": 1615565482
}
```

## WebSocket Payload Formats

### Text Message Formats

WebSocket communication uses both text (JSON) and binary messages. Text messages are used for control messages and metadata:

#### Request Initial Data

**Client → Server**:
```json
{
  "type": "requestInitialData",
  "timestamp": 1615479082
}
```

#### Loading Status

**Server → Client**:
```json
{
  "type": "loading",
  "message": "Loading graph data...",
  "timestamp": 1615479082
}
```

#### Updates Started

**Server → Client**:
```json
{
  "type": "updatesStarted",
  "timestamp": 1615479082
}
```

#### Connection Established

**Server → Client**:
```json
{
  "type": "connection_established",
  "timestamp": 1615479082
}
```

#### Enable Randomization

**Client → Server**:
```json
{
  "type": "enableRandomization",
  "enabled": true,
  "timestamp": 1615479082
}
```

#### Heartbeat (Ping)

**Client → Server**:
```json
{
  "type": "ping",
  "timestamp": 1615479082
}
```

### Binary Message Format

Binary messages are used for efficient position updates:

#### Binary Position Update Format

Each node's position data is encoded in a fixed-length binary format:

| Field      | Type    | Size (bytes) | Description               |
|------------|---------|--------------|---------------------------|
| Node ID    | uint16  | 2            | Numeric ID for the node   |
| Position X | float32 | 4            | X coordinate              |
| Position Y | float32 | 4            | Y coordinate              |
| Position Z | float32 | 4            | Z coordinate              |
| Velocity X | float32 | 4            | X velocity component      |
| Velocity Y | float32 | 4            | Y velocity component      |
| Velocity Z | float32 | 4            | Z velocity component      |
| **Total**  |         | **26**       | **Bytes per node**        |

The binary message consists of multiple node entries concatenated together:

```
[Node1][Node2][Node3]...
```

Where each `[NodeN]` is 26 bytes as specified above.

## Data Transformation

The client applies transformations to convert between the API payload formats and the internal data structures:

### Graph Data Transformation

```typescript
// From client/core/types.ts
export function transformGraphData(data: any): GraphData {
  // Transform nodes
  const nodes = data.nodes.map(node => ({
    id: node.id,
    data: {
      position: new Vector3(
        node.data.position.x,
        node.data.position.y,
        node.data.position.z
      ),
      velocity: new Vector3(
        node.data.velocity?.x || 0,
        node.data.velocity?.y || 0,
        node.data.velocity?.z || 0
      ),
      metadata: node.data.metadata
    }
  }));
  
  // Transform edges
  const edges = data.edges.map(edge => ({
    source: edge.source,
    target: edge.target,
    data: edge.data || {}
  }));
  
  return {
    nodes,
    edges,
    metadata: data.metadata || {}
  };
}
```

### Settings Transformation

```typescript
// From client/state/settings.ts
export function transformSettings(data: any): Settings {
  // Apply default values for missing properties
  return deepMerge(defaultSettings, data);
}

// Helper function to perform deep merge
function deepMerge<T>(target: T, source: any): T {
  const result = { ...target };
  
  // For each property in source
  for (const key in source) {
    // If property is an object, recursively merge
    if (isObject(source[key]) && isObject(result[key])) {
      result[key] = deepMerge(result[key], source[key]);
    } else if (source[key] !== undefined) {
      // Otherwise directly assign if not undefined
      result[key] = source[key];
    }
  }
  
  return result;
}
```

## Validation

Payload validation is performed to ensure data integrity:

### Request Validation

```typescript
// Validate settings update payload
function validateSettingsPayload(data: any): ValidationResult {
  const errors = [];
  
  // Check that data is an object
  if (!data || typeof data !== 'object') {
    return {
      valid: false,
      errors: [{ message: 'Settings must be an object' }]
    };
  }
  
  // Validate visualization settings
  if (data.visualization) {
    if (data.visualization.nodes) {
      // Validate node size
      if (data.visualization.nodes.size !== undefined) {
        const size = data.visualization.nodes.size;
        if (typeof size !== 'number' || size < 0.1 || size > 10) {
          errors.push({
            path: 'visualization.nodes.size',
            message: 'Node size must be a number between 0.1 and 10'
          });
        }
      }
      
      // Validate node color
      if (data.visualization.nodes.color !== undefined) {
        const color = data.visualization.nodes.color;
        if (typeof color !== 'string' || !color.match(/^#[0-9A-F]{6}$/i)) {
          errors.push({
            path: 'visualization.nodes.color',
            message: 'Node color must be a valid hex color (e.g., "#FF0000")'
          });
        }
      }
    }
    
    // Similar validation for other settings...
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}
```

### Response Validation

```typescript
// Validate graph data response
function validateGraphDataResponse(data: any): ValidationResult {
  const errors = [];
  
  // Check that data is an object
  if (!data || typeof data !== 'object') {
    return {
      valid: false,
      errors: [{ message: 'Response must be an object' }]
    };
  }
  
  // Check nodes array
  if (!Array.isArray(data.nodes)) {
    errors.push({
      path: 'nodes',
      message: 'Nodes must be an array'
    });
  } else {
    // Validate each node
    data.nodes.forEach((node, index) => {
      if (!node.id) {
        errors.push({
          path: `nodes[${index}].id`,
          message: 'Node must have an ID'
        });
      }
      
      if (!node.data || typeof node.data !== 'object') {
        errors.push({
          path: `nodes[${index}].data`,
          message: 'Node must have data object'
        });
      } else {
        // Validate position
        if (!node.data.position || 
            typeof node.data.position !== 'object' ||
            typeof node.data.position.x !== 'number' ||
            typeof node.data.position.y !== 'number' ||
            typeof node.data.position.z !== 'number') {
          errors.push({
            path: `nodes[${index}].data.position`,
            message: 'Node position must be an object with numeric x, y, z properties'
          });
        }
      }
    });
  }
  
  // Check edges array
  if (!Array.isArray(data.edges)) {
    errors.push({
      path: 'edges',
      message: 'Edges must be an array'
    });
  } else {
    // Validate each edge
    data.edges.forEach((edge, index) => {
      if (!edge.source) {
        errors.push({
          path: `edges[${index}].source`,
          message: 'Edge must have a source'
        });
      }
      
      if (!edge.target) {
        errors.push({
          path: `edges[${index}].target`,
          message: 'Edge must have a target'
        });
      }
    });
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}
```

## Binary Data Encoding/Decoding

For efficient communication, binary data formats are used:

### Binary Position Encoding

```typescript
// Encode node positions to binary
function encodeNodePositions(nodes: NodePositionUpdate[]): ArrayBuffer {
  const buffer = new ArrayBuffer(nodes.length * 26); // 26 bytes per node
  const view = new DataView(buffer);
  
  nodes.forEach((node, index) => {
    const offset = index * 26;
    
    // Write node ID as u16
    view.setUint16(offset, parseInt(node.id, 10), true);
    
    // Write position (3 x f32)
    view.setFloat32(offset + 2, node.position.x, true);
    view.setFloat32(offset + 6, node.position.y, true);
    view.setFloat32(offset + 10, node.position.z, true);
    
    // Write velocity (3 x f32)
    view.setFloat32(offset + 14, node.velocity?.x || 0, true);
    view.setFloat32(offset + 18, node.velocity?.y || 0, true);
    view.setFloat32(offset + 22, node.velocity?.z || 0, true);
  });
  
  return buffer;
}
```

### Binary Position Decoding

```typescript
// Decode binary position data
function decodeNodePositions(buffer: ArrayBuffer): NodePositionUpdate[] {
  const view = new DataView(buffer);
  const nodeCount = Math.floor(buffer.byteLength / 26);
  const nodes: NodePositionUpdate[] = [];
  
  for (let i = 0; i < nodeCount; i++) {
    const offset = i * 26;
    
    // Read node ID
    const id = view.getUint16(offset, true).toString();
    
    // Read position
    const position = new Vector3(
      view.getFloat32(offset + 2, true),
      view.getFloat32(offset + 6, true),
      view.getFloat32(offset + 10, true)
    );
    
    // Read velocity
    const velocity = new Vector3(
      view.getFloat32(offset + 14, true),
      view.getFloat32(offset + 18, true),
      view.getFloat32(offset + 22, true)
    );
    
    nodes.push({
      id,
      position,
      velocity
    });
  }
  
  return nodes;
}
```

## Next Sections

For more detailed information, refer to:
- [REST Endpoints](rest-endpoints.md) - REST API details
- [WebSocket Protocol](websocket-protocol.md) - WebSocket protocol details
- [Authentication](authentication.md) - Authentication mechanisms