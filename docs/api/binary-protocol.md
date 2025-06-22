# Binary Protocol Documentation

## Overview

The LogseqSpringThing binary protocol is designed for efficient transmission of node position and velocity data over WebSocket connections. It uses a compact binary format to minimize bandwidth usage while maintaining precision for real-time graph visualization updates.

## Protocol Architecture

### Design Principles

1. **Efficiency**: Minimize bytes per node update
2. **Simplicity**: Fixed-size records for fast parsing
3. **Separation**: Wire format differs from server storage format
4. **Safety**: Type-safe serialization using Rust's bytemuck

### Components

- **Binary Protocol** (`src/utils/binary_protocol.rs`): Encoding/decoding logic
- **Socket Flow Messages** (`src/utils/socket_flow_messages.rs`): Data structures
- **Socket Flow Constants** (`src/utils/socket_flow_constants.rs`): Protocol constants

## Wire Format Specification

### Node Data Structure

Each node is transmitted as a fixed 28-byte structure:

```
┌─────────────┬────────────────┬────────────────┐
│  Node ID    │    Position    │    Velocity    │
│  (4 bytes)  │   (12 bytes)   │   (12 bytes)   │
└─────────────┴────────────────┴────────────────┘
```

### Field Details

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Node ID | u32 | 4 bytes | Unique node identifier |
| Position.x | f32 | 4 bytes | X coordinate |
| Position.y | f32 | 4 bytes | Y coordinate |
| Position.z | f32 | 4 bytes | Z coordinate |
| Velocity.x | f32 | 4 bytes | X velocity |
| Velocity.y | f32 | 4 bytes | Y velocity |
| Velocity.z | f32 | 4 bytes | Z velocity |

### Wire Format Structure (Rust)

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireNodeDataItem {
    pub id: u32,           // 4 bytes
    pub position: Vec3Data, // 12 bytes (3 × f32)
    pub velocity: Vec3Data, // 12 bytes (3 × f32)
    // Total: 28 bytes
}
```

## Message Format

### Binary Message Structure

A complete binary message consists of concatenated node data:

```
┌──────────────┬──────────────┬─────┬──────────────┐
│    Node 1    │    Node 2    │ ... │    Node N    │
│  (28 bytes)  │  (28 bytes)  │     │  (28 bytes)  │
└──────────────┴──────────────┴─────┴──────────────┘
```

### Message Size Calculation

```rust
pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    updates.len() * std::mem::size_of::<WireNodeDataItem>()
}
```

For example:
- 100 nodes = 2,800 bytes
- 1,000 nodes = 28,000 bytes (~27.3 KB)
- 10,000 nodes = 280,000 bytes (~273 KB)

## Encoding Process

### Server to Client

1. **Collect Updates**: Gather node positions and velocities
2. **Create Wire Format**: Convert to `WireNodeDataItem` structures
3. **Serialize**: Use bytemuck for zero-copy serialization
4. **Transmit**: Send as binary WebSocket frame

```rust
pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(
        nodes.len() * std::mem::size_of::<WireNodeDataItem>()
    );
    
    for (node_id, node) in nodes {
        let wire_item = WireNodeDataItem {
            id: *node_id,
            position: node.position,
            velocity: node.velocity,
        };
        
        // Safe, direct memory layout conversion
        let item_bytes = bytemuck::bytes_of(&wire_item);
        buffer.extend_from_slice(item_bytes);
    }
    
    buffer
}
```

### Important Notes

- Server-side fields (mass, flags, padding) are NOT transmitted
- Node IDs must be u32 for protocol compatibility
- All floating-point values use IEEE 754 single precision (f32)

## Decoding Process

### Client to Server

1. **Receive Binary**: Get binary WebSocket frame
2. **Validate Size**: Ensure data is multiple of 28 bytes
3. **Deserialize**: Parse fixed-size chunks
4. **Reconstruct**: Create server-side structures with defaults

```rust
pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    const WIRE_ITEM_SIZE: usize = std::mem::size_of::<WireNodeDataItem>();
    
    // Validate data size
    if data.len() % WIRE_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of wire item size {}",
            data.len(), WIRE_ITEM_SIZE
        ));
    }
    
    let mut updates = Vec::with_capacity(data.len() / WIRE_ITEM_SIZE);
    
    // Process fixed-size chunks
    for chunk in data.chunks_exact(WIRE_ITEM_SIZE) {
        let wire_item: WireNodeDataItem = *bytemuck::from_bytes(chunk);
        
        // Convert to server format with defaults
        let server_node_data = BinaryNodeData {
            position: wire_item.position,
            velocity: wire_item.velocity,
            mass: 100u8,     // Default, replaced from node_map
            flags: 0u8,      // Default, replaced from node_map
            padding: [0u8, 0u8],
        };
        
        updates.push((wire_item.id, server_node_data));
    }
    
    Ok(updates)
}
```

## Type Definitions

### Vec3Data Structure

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vec3Data {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
```

### BinaryNodeData (Server Format)

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BinaryNodeData {
    pub position: Vec3Data,    // 12 bytes
    pub velocity: Vec3Data,    // 12 bytes
    pub mass: u8,             // 1 byte (server-only)
    pub flags: u8,            // 1 byte (server-only)
    pub padding: [u8; 2],     // 2 bytes (server-only)
    // Total: 28 bytes (server-side)
}
```

## WebSocket Integration

### Message Types

The binary protocol is used for specific WebSocket message types:

```typescript
// Client-side message handling
socket.addEventListener('message', (event) => {
    if (event.data instanceof ArrayBuffer) {
        // Binary message - node updates
        handleBinaryUpdate(event.data);
    } else {
        // Text message - JSON protocol
        const message = JSON.parse(event.data);
        handleJsonMessage(message);
    }
});
```

### Binary Frame Format

WebSocket binary frames use opcode 0x2:
- FIN bit: 1 (complete message)
- Opcode: 0x2 (binary frame)
- Payload: Concatenated node data

## Performance Characteristics

### Bandwidth Usage

| Nodes | Message Size | Update Rate | Bandwidth |
|-------|--------------|-------------|-----------|
| 100 | 2.8 KB | 5 Hz | 14 KB/s |
| 500 | 14 KB | 5 Hz | 70 KB/s |
| 1,000 | 28 KB | 5 Hz | 140 KB/s |
| 5,000 | 140 KB | 5 Hz | 700 KB/s |
| 10,000 | 280 KB | 5 Hz | 1.4 MB/s |

### Optimization Strategies

1. **Delta Updates**: Only send changed nodes
2. **Throttling**: Limit update frequency based on client capacity
3. **Compression**: Apply gzip for large updates (>1KB)
4. **Chunking**: Split large updates across multiple frames

## Error Handling

### Common Errors

1. **Invalid Data Size**
   ```rust
   if data.len() % WIRE_ITEM_SIZE != 0 {
       return Err("Data size is not a multiple of wire item size");
   }
   ```

2. **Empty Data**
   ```rust
   if data.is_empty() {
       return Ok(Vec::new());
   }
   ```

3. **Deserialization Failure**
   - Handled by bytemuck's type safety
   - Panics on alignment issues (prevented by #[repr(C)])

### Client-Side Validation

```typescript
function decodeBinaryUpdate(buffer: ArrayBuffer): NodeUpdate[] {
    const BYTES_PER_NODE = 28;
    
    if (buffer.byteLength % BYTES_PER_NODE !== 0) {
        throw new Error('Invalid binary data size');
    }
    
    const view = new DataView(buffer);
    const nodeCount = buffer.byteLength / BYTES_PER_NODE;
    const updates: NodeUpdate[] = [];
    
    for (let i = 0; i < nodeCount; i++) {
        const offset = i * BYTES_PER_NODE;
        updates.push({
            id: view.getUint32(offset, true),
            position: {
                x: view.getFloat32(offset + 4, true),
                y: view.getFloat32(offset + 8, true),
                z: view.getFloat32(offset + 12, true),
            },
            velocity: {
                x: view.getFloat32(offset + 16, true),
                y: view.getFloat32(offset + 20, true),
                z: view.getFloat32(offset + 24, true),
            }
        });
    }
    
    return updates;
}
```

## Testing

### Unit Tests

```rust
#[test]
fn test_wire_format_size() {
    assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);
}

#[test]
fn test_encode_decode_roundtrip() {
    let nodes = vec![
        (1u32, BinaryNodeData {
            position: Vec3Data::new(1.0, 2.0, 3.0),
            velocity: Vec3Data::new(0.1, 0.2, 0.3),
            mass: 100,
            flags: 1,
            padding: [0, 0],
        }),
    ];
    
    let encoded = encode_node_data(&nodes);
    assert_eq!(encoded.len(), 28);
    
    let decoded = decode_node_data(&encoded).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].0, 1);
    assert_eq!(decoded[0].1.position.x, 1.0);
}
```

### Integration Testing

1. **Round-trip Test**: Encode on server, decode on client
2. **Performance Test**: Measure encoding/decoding time
3. **Stress Test**: Handle maximum node counts
4. **Error Test**: Verify handling of malformed data

## Protocol Constants

From `src/utils/socket_flow_constants.rs`:

```rust
// Binary message constants
pub const NODE_POSITION_SIZE: usize = 24; // 6 f32s * 4 bytes
pub const BINARY_HEADER_SIZE: usize = 4;  // 1 f32 for header

// Compression constants
pub const COMPRESSION_THRESHOLD: usize = 1024; // 1KB
pub const ENABLE_COMPRESSION: bool = true;

// WebSocket constants
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB
pub const BINARY_CHUNK_SIZE: usize = 64 * 1024; // 64KB
```

## Future Enhancements

### Planned Improvements

1. **Variable-Length Encoding**: Use varint for node IDs
2. **Differential Updates**: Send only position deltas
3. **Batch Compression**: Group similar updates
4. **Custom Float Precision**: Reduce to 16-bit floats where appropriate

### Protocol Versioning

Future versions may include:
- Version byte at message start
- Backwards compatibility for v1 clients
- Feature negotiation during handshake

## Security Considerations

### Data Validation

- Validate all array bounds before access
- Use type-safe deserialization (bytemuck)
- Limit maximum message size (100MB)
- Rate limit binary updates

### Memory Safety

- Fixed-size allocations prevent DoS
- No dynamic memory allocation during decode
- Bounded update counts per message

## Debugging

### Logging

Enable binary protocol logging:
```bash
RUST_LOG=logseq_spring_thing::utils::binary_protocol=trace
```

### Sample Output
```
[TRACE] Encoding 3 nodes for binary transmission
[TRACE] Encoding node 0: pos=[1.234,5.678,9.012], vel=[0.100,0.200,0.300]
[TRACE] Encoded binary data: 84 bytes for 3 nodes
```

### Binary Data Inspection

Use hex dump for debugging:
```bash
xxd -g 1 binary_message.bin
```

Example output:
```
00000000: 01 00 00 00 00 00 80 3f 00 00 00 40 00 00 40 40  .......?...@..@@
00000010: cd cc cc 3d cd cc 4c 3e 9a 99 99 3e              ...=..L>...>
```