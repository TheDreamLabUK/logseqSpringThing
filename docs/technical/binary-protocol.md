# WebSocket Binary Protocol

The WebSocket binary protocol has been optimized for efficient transmission of node position and velocity updates.

## Protocol Format

```
[4 bytes] message_type (u32)
[4 bytes] node_count (u32)
For each node:
[4 bytes] node_id (u32)
[12 bytes] Position (3 × f32)
[12 bytes] Velocity (3 × f32)
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

- **Position (12 bytes):**
  - Three 32-bit floats (f32)
  - Each float represents x, y, and z coordinates
  - Direct floating-point values, no quantization

- **Velocity (12 bytes):**
  - Three 32-bit floats (f32)
  - Each float represents velocity along x, y, and z axes
  - Direct floating-point values, no quantization

#### Total Size
- **Header:** 8 bytes
- **Per node:** 28 bytes (4 + 12 + 12)
- **Message size:** 8 bytes + (28 bytes × number of nodes)

## Protocol Benefits

### 1. Minimal Overhead
- Fixed-size format eliminates need for delimiters
- No additional metadata within node data
- Compact header structure

### 2. Direct Value Representation
- Uses native f32 format for positions and velocities
- No quantization/dequantization overhead
- Full floating-point precision maintained

### 3. Efficient Parsing
- Fixed-size format allows for fast, direct memory access
- Standard data types (u32, f32) are efficiently handled
- Little-endian byte order for compatibility

### 4. Clear Message Structure
- Simple message type identification
- Explicit node count for buffer allocation
- Consistent node data format

## Implementation Example

```typescript
// TypeScript implementation of binary message parsing
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

        const position = {
            x: view.getFloat32(offset, true),
            y: view.getFloat32(offset + 4, true),
            z: view.getFloat32(offset + 8, true)
        };
        offset += 12;

        const velocity = {
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

### Network Optimization
- Fixed message size allows for efficient buffer allocation
- Binary format reduces overhead compared to JSON
- Direct floating-point values maintain precision

### Client-Side Processing
- Fixed-size format enables efficient buffer allocation
- Direct TypedArray access for fast parsing
- No string processing or JSON parsing overhead

### Server-Side Generation
- Efficient packing of data into binary format
- Minimal processing overhead for each update
- Optimized for high-frequency updates

For more information about performance optimizations, see [Performance Optimizations](./performance.md).