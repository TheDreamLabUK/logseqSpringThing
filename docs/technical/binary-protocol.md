# WebSocket Binary Protocol

The WebSocket binary protocol has been optimized for efficient transmission of node position and velocity updates using a consistent Vec3 representation across the stack.

## Protocol Format

```
[4 bytes] message_type (u32)
[4 bytes] node_count (u32)
For each node:
  [4 bytes] node_id (u32)
  [12 bytes] Position (Vec3Data)
    - [4 bytes] x (f32)
    - [4 bytes] y (f32)
    - [4 bytes] z (f32)
  [12 bytes] Velocity (Vec3Data)
    - [4 bytes] vx (f32)
    - [4 bytes] vy (f32)
    - [4 bytes] vz (f32)
```

### Detailed Breakdown

#### Header (8 bytes)
- **Message Type (4 bytes):** 
  - u32 value indicating the type of message
  - 0x01 = PositionVelocityUpdate

- **Node Count (4 bytes):**
  - u32 value indicating the number of nodes in the message

#### Node Data (28 bytes per node)
For each node in the message:

- **Node ID (4 bytes):**
  - u32 identifier for the node

- **Position (Vec3Data, 12 bytes):**
  - Three 32-bit floats (f32) in x, y, z order
  - Direct memory layout matching Vec3Data struct
  - CUDA-compatible alignment
  - No padding between components

- **Velocity (Vec3Data, 12 bytes):**
  - Three 32-bit floats (f32) in x, y, z order
  - Same memory layout as position
  - CUDA-compatible alignment
  - No padding between components

#### Total Size
- **Header:** 8 bytes
- **Per node:** 28 bytes (4 + 12 + 12)
- **Message size:** 8 bytes + (28 bytes × number of nodes)

## Protocol Benefits

### 1. Unified Vec3 Representation
- Consistent memory layout across entire stack
- Direct mapping to CUDA structures
- Zero-copy potential between layers
- SIMD-friendly alignment

### 2. Direct Value Representation
- Uses native f32 format for positions and velocities
- No quantization/dequantization overhead
- Full floating-point precision maintained
- Consistent endianness (little-endian)

### 3. Efficient Processing
- Fixed-size format allows for fast, direct memory access
- CUDA-compatible memory layout
- Zero-copy potential on supported platforms
- Vectorization-friendly structure

### 4. Clear Message Structure
- Simple message type identification
- Explicit node count for buffer allocation
- Consistent node data format
- Self-describing layout

## Implementation Example

```typescript
import { Vec3 } from '../types/vec3';

// TypeScript implementation using Vec3 type
function parseNodeUpdate(buffer: ArrayBuffer): NodeUpdate[] {
    const view = new DataView(buffer);
    let offset = 0;

    // Read header
    const messageType = view.getUint32(offset, true);
    offset += 4;

    const nodeCount = view.getUint32(offset, true);
    offset += 4;

    const nodes: NodeUpdate[] = [];

    for (let i = 0; i < nodeCount; i++) {
        const nodeId = view.getUint32(offset, true);
        offset += 4;

        // Read position Vec3
        const position: Vec3 = {
            x: view.getFloat32(offset, true),
            y: view.getFloat32(offset + 4, true),
            z: view.getFloat32(offset + 8, true)
        };
        offset += 12;

        // Read velocity Vec3
        const velocity: Vec3 = {
            x: view.getFloat32(offset, true),
            y: view.getFloat32(offset + 4, true),
            z: view.getFloat32(offset + 8, true)
        };
        offset += 12;

        nodes.push({ nodeId, position, velocity });
    }

    return nodes;
}
```

## Performance Considerations

### Memory Layout Optimization
- Consistent Vec3 layout across stack
- CUDA-compatible structure alignment
- SIMD-friendly data organization
- Zero-copy potential where supported

### Network Optimization
- Fixed message size allows for efficient buffer allocation
- Binary format reduces overhead compared to JSON
- Direct floating-point values maintain precision
- Compression for large updates

### Client-Side Processing
- Fixed-size format enables efficient buffer allocation
- Direct TypedArray access for fast parsing
- No string processing or JSON parsing overhead
- Direct conversion to Three.js Vector3

### Server-Side Generation
- Efficient packing of data into binary format
- Direct use of Vec3Data memory layout
- CUDA-compatible structure
- Optimized for high-frequency updates

For more information about the Vec3 alignment across the stack, see [Vec3 Alignment](./vec3-alignment.md).
For performance optimization details, see [Performance Optimizations](./performance.md).