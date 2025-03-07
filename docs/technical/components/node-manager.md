# Node Management System

## Architecture

The node management system uses a modular architecture optimized for Meta Quest 3 performance.

### Core Components

1. NodeGeometryManager
   - Handles LOD and geometry optimization
   - Manages geometry levels for different distances
   - Optimizes for Quest hardware

2. NodeInstanceManager
   - Uses THREE.InstancedMesh for efficient rendering
   - Manages batched updates
   - Handles node indices and pending updates

3. NodeMetadataManager
   - Efficient label rendering
   - Distance-based visibility
   - Memory-optimized metadata storage

4. NodeInteractionManager
   - XR interaction handling
   - Gesture recognition
   - Haptic feedback support

## Scale Management

### AR Space Matching
- Room scale is the source of truth for AR mode
- Only xrSessionManager.ts handles room-scale adjustments
- All other scaling is relative to room scale

### Visualization Hierarchy
- Node scaling based on visual importance
- Hologram instance scaling for scene composition
- Spatial relationships preserved in AR mode

## Performance Optimizations

- Instanced rendering for improved GPU performance
- LOD system for distance-based detail
- Batched updates for efficient state management
- Memory-optimized data structures

## Configuration

```yaml
features:
  useInstancedNodes: true  # Enable instanced rendering
  enableLOD: true         # Enable Level of Detail system
```

## Integration Points

- WebSocket updates via binary protocol
- XR session management
- Visualization system
- Physics system