# Node ID Binding Between Binary Protocol and Metadata

## Overview

This document explains the critical binding between node IDs in the binary WebSocket protocol and metadata visualization in the client application. Understanding this binding is essential for maintaining consistent metadata labels and position updates.

## The Node ID Flow

The application uses a specific flow for node IDs that must be preserved across different systems:

1. Server generates numeric node IDs (u16) for each node in the graph
2. These IDs are sent to the client via the binary WebSocket protocol
3. Client uses these numeric IDs (as strings) to bind nodes with their metadata
4. Metadata visualization relies on these same IDs for position updates

```
┌──────────────────┐   Binary Protocol   ┌──────────────────┐
│                  │                     │                  │
│  Server (Rust)   │  u16 Numeric IDs    │  Client (TS)     │
│  Node IDs        │ ──────────────────> │  Node IDs        │
│  (numeric)       │                     │  (string)        │
│                  │                     │                  │
└──────────────────┘                     └──────────────────┘
                                                 │
                                                 │
                                                 ▼
                                         ┌──────────────────┐
                                         │  Metadata        │
                                         │  Visualization   │
                                         │  (same IDs)      │
                                         └──────────────────┘
```

## Binary Protocol Format

The binary WebSocket protocol uses a compact format optimized for performance:

- **Node data**: 26 bytes per node
  - **Node ID** (u16): 2 bytes
  - **Position** (3 x float32): 12 bytes
  - **Velocity** (3 x float32): 12 bytes

The node ID in this protocol is the single source of truth for node identification. All systems must reference this ID for proper binding.

## Common Issues and Solutions

### Problem: All Nodes Showing Same Metadata

When all nodes display the same metadata (e.g., all showing "1000B" file size):

**Root Cause**: Mismatch between IDs used in the binary protocol and metadata visualization.

**Solution**:
1. Ensure the numeric ID from the binary protocol is converted to a string consistently
2. Use this same string ID in MetadataVisualizer.createMetadataLabel()
3. Store the ID in userData for position updates

### Problem: Labels Not Following Node Movements

When node labels don't update position with their corresponding nodes:

**Root Cause**: IDs used in updateMetadataPosition() don't match those used in createMetadataLabel().

**Solution**:
1. Use the same ID format for position updates as was used for label creation
2. Verify the node ID mapping in NodeInstanceManager

### Problem: Missing Metadata Fields

When some metadata fields are missing or showing default values:

**Root Cause**: Improper metadata extraction in the transformNodeData function.

**Solution**:
1. Ensure proper type checking and fallbacks in transformNodeData
2. Verify that metadata is correctly populated from API responses
3. Check for null/undefined handling in metadata display logic

## Key Components and Their Roles

### 1. WebSocketService (client/websocket/websocketService.ts)

This service:
- Receives binary data from the server
- Converts u16 node IDs to strings
- Creates position updates with proper IDs
- Maintains node ID mappings

**Critical Code**:
```typescript
// Convert the numeric ID to a string to match our node ID storage format
const nodeId = id.toString();
```

### 2. VisualizationController (client/rendering/VisualizationController.ts)

This component:
- Coordinates between WebSocketService and visual components
- Routes position updates to NodeInstanceManager
- Initializes metadata visualization with correct IDs
- Ensures consistent ID usage across subsystems

**Critical Code**:
```typescript
// Using the correct node ID for metadata binding
this.metadataVisualizer?.createMetadataLabel(metadata, node.id);
```

### 3. MetadataVisualizer (client/rendering/MetadataVisualizer.ts)

This component:
- Creates and manages metadata labels
- Stores node IDs in userData for later position updates
- Updates label positions based on node movements

**Critical Code**:
```typescript
// Store the exact nodeId in userData for position updates
group.userData = { 
  isMetadata: true,
  nodeId
};
```

### 4. NodeInstanceManager (client/rendering/node/instance/NodeInstanceManager.ts)

This component:
- Manages the instanced rendering of nodes
- Maps between node IDs and instance indices
- Applies position updates from the binary protocol

## Best Practices

### 1. Node ID Consistency

- Always use string representations of the numeric u16 IDs from the server
- Maintain a single source of truth for node ID mapping
- Validate ID format when binding metadata to nodes

### 2. Type Safety

- Verify types when converting between numeric and string IDs
- Use strict ID validation to catch binding issues early
- Add debug logs for ID mismatch detection

### 3. Testing and Validation

- Create tests specifically for node ID binding
- Implement validation checks for metadata-node connections
- Log node ID mappings during initialization

### 4. Code Documentation

- Clearly mark sections that deal with node ID binding
- Comment on the ID format expectations for each component
- Document the origin and lifecycle of node IDs

## Implementation Details

### Server-Side (Rust)

The server generates numeric node IDs in two key files:

1. **src/models/node.rs**:
   - Generates sequential numeric IDs
   - Stores these IDs in the Node struct
   - Transmits them via binary protocol

2. **src/utils/socket_flow_messages.rs**:
   - Defines the BinaryNodeData struct with ID field
   - Handles binary encoding/decoding of node data

### Client-Side (TypeScript)

The client maintains the node ID binding in several key files:

1. **client/websocket/websocketService.ts**:
   - Converts binary u16 IDs to string format
   - Maps between node names and numeric IDs
   - Manages node registration

2. **client/rendering/VisualizationController.ts**:
   - Initializes metadata with appropriate IDs
   - Routes position updates to both NodeInstanceManager and MetadataVisualizer

3. **client/rendering/MetadataVisualizer.ts**:
   - Creates visual metadata with node IDs
   - Updates metadata positions when nodes move

4. **client/rendering/node/instance/NodeInstanceManager.ts**:
   - Manages node instance updates via binary data
   - Maps between node IDs and instance indices

## Debugging Node ID Binding Issues

### 1. Enable Debug Logging

```typescript
debugState.enable('node', 'websocket', 'data');
```

### 2. Check ID Consistency

In browser console:
```javascript
// Show node ID mappings
WebSocketDiagnostics.checkNodeIdMapping();

// Verify node ID to metadata mapping
MetadataDiagnostics.checkNodeBindings();
```

### 3. Validate Binary Protocol

Examine binary messages in the Network tab to verify:
- Node ID byte ordering (little-endian)
- Message size is a multiple of 26 bytes
- No duplicate IDs within a single message

## Conclusion

Proper node ID binding between the binary protocol and metadata visualization is critical for correct application behavior. By maintaining a consistent approach to node ID handling and following the guidelines in this document, similar issues should be avoided in the future.