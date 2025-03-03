# Vec3 Alignment Across Stack

## Overview

This document describes the unified Vec3 representation implemented across the full stack, from CUDA computation to client-side rendering. The system has been fully refactored to use Three.js Vector3 objects throughout the client codebase, eliminating any custom Vec3 implementations in TypeScript.

## Components

### 1. Vec3Data (Rust)
Located in `src/types/vec3.rs`:
- CUDA-compatible struct (Pod, Zeroable)
- Direct memory layout: `x`, `y`, `z` fields
- Converts to/from glam::Vec3
- Implements array conversion for compatibility

### 2. CUDA Integration
Located in `src/utils/compute_forces.cu`:
- NodeData struct matches Vec3Data memory layout
- Direct field access eliminates array conversions
- Maintains CUDA float3 for computation efficiency
- Improved memory coherence

### 3. Binary Protocol
Located in `src/utils/binary_protocol.rs`:
- Direct serialization of Vec3Data
- Node IDs stored as u16 (2 bytes) for compactness
- Consistent 12-byte layout per vector
- Little-endian encoding
- 26 bytes per node (2 + 12 + 12)

### 4. WebSocket Protocol
Located in `src/utils/socket_flow_messages.rs`:
- BinaryNodeData uses Vec3Data for position/velocity
- Maintains CUDA compatibility
- Efficient binary transmission

### 5. Client Implementation (Updated)
Directly uses THREE.Vector3 throughout:
- Eliminated custom Vec3 implementation in TypeScript
- Direct usage of Three.js Vector3 objects for all vector operations
- Streamlined vector arithmetic using native Three.js methods
- Improved performance by removing conversion steps

## Memory Layout

All Vec3 representations share the same memory layout:
```
struct Vec3 {
    x: f32,  // 0-3   bytes
    y: f32,  // 4-7   bytes
    z: f32,  // 8-11  bytes
}           // Total: 12 bytes
```

## Benefits

1. Performance:
   - Reduced conversions
   - Better memory locality
   - SIMD-friendly layout

2. Consistency:
   - Single source of truth
   - Type safety across boundaries
   - Clear data flow

3. Maintainability:
   - Unified vector operations
   - Simplified debugging
   - Clear upgrade path

## Usage

### Rust Server
```rust
use crate::types::vec3::Vec3Data;

let vec = Vec3Data::new(x, y, z);
```

### CUDA Kernel
```cpp
struct NodeData {
    float x, y, z;        // position
    float vx, vy, vz;     // velocity
};
```

### TypeScript Client (Updated)
```typescript
import { Vector3 } from 'three';

const vec = new Vector3(x, y, z);
```

## Future Improvements

1. Optimized Binary Protocol:
   - Reduced node size from 28 to 26 bytes per node
   - Changed node ID from u32 to u16
   - ✅ Direct Vector3/Vec3Data representation throughout pipeline
   - ✅ Helper functions for GPU array compatibility
   - ✅ Improved type safety and code clarity
   - ✅ Eliminated custom Vec3 TypeScript implementation

1. SIMD Operations:
   - Add SIMD-optimized operations to Vec3Data
   - Leverage CPU vector instructions

2. Client-side Benefits (Achieved):
   - ✅ Eliminated unnecessary conversions between custom Vec3 and THREE.Vector3
   - ✅ Reduced memory footprint by removing duplicate vector representations
   - ✅ Improved performance of vector operations using Three.js optimized methods
   - ✅ Simplified codebase by standardizing on a single vector implementation

3. Zero-Copy:
   - Implement zero-copy transmission where possible
   - Reduce serialization overhead

4. GPU Direct:
   - Investigate direct GPU memory access
   - Reduce CPU-GPU transfers

## Edge Handling Optimization

The edge handling system has been significantly improved by the Vector3 refactoring:

1. Edges now directly use Vector3 objects for source and target positions
2. Edge geometries are updated in-place rather than recreated
3. Position calculations use native Three.js vector methods for better performance
4. Temporary vector objects are reused to minimize garbage collection

For detailed information on edge handling improvements, see [Edge Handling Optimization](./edge-handling-optimization.md).