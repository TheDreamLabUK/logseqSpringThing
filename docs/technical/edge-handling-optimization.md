# Edge Handling Optimization

## Overview

This document describes the optimized edge handling system in the VisionFlow application, focusing on the improvements made after refactoring to use Three.Vector3 throughout the codebase. The edge handling architecture has been optimized for performance and memory efficiency while maintaining visual quality.

## Architecture

The edge rendering system follows this high-level flow:

1. **Server**: Rust backend constructs edge data in `graph_service.rs`
2. **API**: Edges are sent via JSON in REST sync calls to the client
3. **Client Processing**: `graphData.ts` processes and manages edge data
4. **Rendering**: `EdgeManager.ts` handles the visualization of edges in the 3D scene

## Key Improvements

### 1. Consistent Vector3 Usage

- Refactored to use Three.js Vector3 objects consistently throughout the codebase
- Eliminated array-based position handling that previously required conversion
- Simplified code by leveraging Vector3 methods for operations like distance calculations

### 2. Memory Optimization

- Reused temporary Vector3 objects to reduce garbage collection pressure
- Added object pooling for frequently created objects
- Eliminated unnecessary object creation in hot paths
- Replaced `new Vector3()` calls with instance-level reusable vector objects

### 3. Rendering Efficiency

- Used in-place updates for edge geometry instead of recreating geometries
- Updated BufferAttributes directly instead of replacing entire geometries
- Tracked edge identity to avoid unnecessary recreation
- Implemented selective updating for edges that actually changed

### 4. Better Edge Management

- Maintained map of edges by ID for efficient lookup
- Cached source/target relationships to avoid redundant lookups
- Added edge identity tracking to prevent duplicate edges
- Improved validation logic for edge positions

## Implementation Details

### EdgeManager Optimizations

```typescript
// Reusable objects to avoid allocation during updates
private tempDirection = new Vector3();
private tempSourceVec = new Vector3();
private tempTargetVec = new Vector3();
```

The `EdgeManager` now maintains a set of reusable Vector3 objects that are used for calculations without creating new objects during each update. This significantly reduces memory churn.

### Incremental Edge Updates

Previously, the `updateEdges` method would clear all existing edges and recreate them from scratch. The improved approach:

1. Tracks which edges have been updated
2. Only creates new edges when needed
3. Updates existing edges in-place
4. Removes edges that no longer exist

This dramatically reduces the overhead when only a few edges change.

### In-place Geometry Updates

The edge position update logic now modifies existing geometries directly:

```typescript
// Update the existing BufferAttribute instead of recreating the geometry
posAttr.setXYZ(0, sourcePos.x, sourcePos.y, sourcePos.z);
posAttr.setXYZ(1, this.tempTargetVec.x, this.tempTargetVec.y, this.tempTargetVec.z);
posAttr.needsUpdate = true;
```

Instead of creating a new geometry, disposing the old one, and replacing it (which was expensive), we now update the existing buffer attributes directly.

## Performance Impact

The edge handling optimizations provide several performance benefits:

1. **Reduced Memory Allocation**: Less garbage collection pauses due to fewer object allocations
2. **Faster Updates**: Direct BufferAttribute updates are faster than geometry replacement
3. **Selective Processing**: Only processing edges that changed reduces unnecessary work
4. **Vector Math Efficiency**: Using Vector3 native methods is more efficient than manual calculations

## Server-side Edge Construction

On the server side (Rust), edges are constructed in `graph_service.rs` and include:

1. Source/target node IDs
2. Weight values
3. Optional metadata

The server builds edges through the following process:
1. Create nodes from metadata files
2. Analyze topic connections between files
3. Create edges based on relationships
4. Calculate initial layout using GPU physics
5. Send serialized edge data via API responses

## Client-side Edge Processing

The client processes edge data in `graphData.ts`:

1. Maps edges by ID (using a generated or provided ID)
2. Tracks source/target relationships
3. Updates edge data when new data is received
4. Ensures edges reference valid nodes

## Edge Rendering

The actual rendering of edges is handled by `EdgeManager.ts` which:

1. Creates line geometries for edges
2. Applies materials with appropriate colors and opacity
3. Manages edge visibility in different modes (desktop vs XR)
4. Updates edge positions based on node movements
5. Applies visual effects like pulsing animation

## Best Practices

When working with the edge system:

1. Always use Vector3 objects for positions
2. Reuse Vector3 instances when performing calculations in loops
3. Use the EdgeManager's API rather than directly manipulating edges
4. Ensure edges have valid source/target nodes
5. Be mindful of edge count for performance

## Future Improvements

Potential future optimizations include:

1. Instanced line rendering for better performance with large numbers of edges
2. Level of detail system for edges based on distance
3. Edge bundling for cleaner visualization
4. Edge importance scoring for selective display
5. WebGPU acceleration for edge physics and rendering