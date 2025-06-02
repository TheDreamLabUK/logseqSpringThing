use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use bytemuck::{Pod, Zeroable};
use log::{trace, debug};

/// Explicit wire format struct for WebSocket binary protocol
/// This struct represents exactly what is sent over the wire
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireNodeDataItem {
    pub id: u32,           // 4 bytes (changed from u16)
    pub position: Vec3Data, // 12 bytes
    pub velocity: Vec3Data, // 12 bytes
    // Total: 28 bytes
}

// Compile-time assertion to ensure wire format is exactly 28 bytes
static_assertions::const_assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);

// Binary format (explicit):
// - For each node (28 bytes total):
//   - Node Index: 4 bytes (u32)
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
// Total: 28 bytes per node

pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoding {} nodes for binary transmission", nodes.len());
    }
    
    let mut buffer = Vec::with_capacity(nodes.len() * std::mem::size_of::<WireNodeDataItem>());
    
    // Log some samples of the encoded data
    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        trace!("Sample of nodes being encoded:");
    }
    
    for (node_id, node) in nodes {
        // Log the first few nodes for debugging
        if sample_size > 0 && *node_id < sample_size as u32 {
            trace!("Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                node_id,
                node.position.x, node.position.y, node.position.z,
                node.velocity.x, node.velocity.y, node.velocity.z);
        }
        
        // Create explicit wire format item
        let wire_item = WireNodeDataItem {
            id: *node_id,
            position: node.position,
            velocity: node.velocity,
        };
        
        // Use bytemuck for safe, direct memory layout conversion
        let item_bytes = bytemuck::bytes_of(&wire_item);
        buffer.extend_from_slice(item_bytes);
        
        // Mass, flags, and padding are server-side only and not transmitted over wire
    }

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoded binary data: {} bytes for {} nodes", buffer.len(), nodes.len());
    }
    buffer
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    const WIRE_ITEM_SIZE: usize = std::mem::size_of::<WireNodeDataItem>();
    
    // Check if data is properly sized
    if data.len() % WIRE_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of wire item size {}",
            data.len(),
            WIRE_ITEM_SIZE
        ));
    }
    
    if data.is_empty() {
        return Ok(Vec::new());
    }
    
    let expected_nodes = data.len() / WIRE_ITEM_SIZE;
    debug!(
        "Decoding binary data: size={} bytes, expected nodes={}",
        data.len(),
        expected_nodes
    );
    
    let mut updates = Vec::with_capacity(expected_nodes);
    
    // Set up sample logging
    let max_samples = 3;
    let mut samples_logged = 0;
    
    debug!("Starting binary data decode, expecting nodes with position and velocity data");
    
    // Process data in chunks of WIRE_ITEM_SIZE bytes
    for chunk in data.chunks_exact(WIRE_ITEM_SIZE) {
        // Use bytemuck for safe deserialization from bytes
        let wire_item: WireNodeDataItem = *bytemuck::from_bytes(chunk);
        
        // Log the first few decoded items as samples
        if samples_logged < max_samples {
            debug!(
                "Decoded node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                wire_item.id,
                wire_item.position.x, wire_item.position.y, wire_item.position.z,
                wire_item.velocity.x, wire_item.velocity.y, wire_item.velocity.z
            );
            samples_logged += 1;
        }
        
        // Convert wire format to server format (BinaryNodeData)
        // Server-side fields (mass, flags, padding) get default values
        let server_node_data = BinaryNodeData {
            position: wire_item.position,
            velocity: wire_item.velocity,
            mass: 100u8,     // Default mass - will be replaced with actual value from node_map
            flags: 0u8,      // Default flags - will be replaced with actual value from node_map
            padding: [0u8, 0u8], // Default padding
        };
        
        updates.push((wire_item.id, server_node_data));
    }
    
    debug!("Successfully decoded {} nodes from binary data", updates.len());
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // Each update uses WireNodeDataItem size
    updates.len() * std::mem::size_of::<WireNodeDataItem>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_format_size() {
        // Verify that WireNodeDataItem is exactly 28 bytes
        assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(1.0, 2.0, 3.0),
                velocity: crate::types::vec3::Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
            (2u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(4.0, 5.0, 6.0),
                velocity: crate::types::vec3::Vec3Data::new(0.4, 0.5, 0.6),
                mass: 200,
                flags: 1,
                padding: [0, 0],
            }),
        ];

        let encoded = encode_node_data(&nodes);
        
        // Verify encoded size matches expected wire format
        assert_eq!(encoded.len(), nodes.len() * std::mem::size_of::<WireNodeDataItem>());
        assert_eq!(encoded.len(), nodes.len() * 28);
        
        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id);
            assert_eq!(orig_data.position, dec_data.position);
            assert_eq!(orig_data.velocity, dec_data.velocity);
            // Note: mass, flags, and padding are not transmitted over the wire,
            // so decoded values will have defaults (100, 0, [0,0])
            assert_eq!(dec_data.mass, 100u8);
            assert_eq!(dec_data.flags, 0u8);
            assert_eq!(dec_data.padding, [0u8, 0u8]);
        }
    }

    #[test]
    fn test_decode_invalid_data() {
        // Test with data that's not a multiple of wire item size (28 bytes)
        let result = decode_node_data(&[0u8; 27]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple of wire item size"));
        
        // Test with data that's too short but multiple of wire size
        let result = decode_node_data(&[0u8; 0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_message_size_calculation() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(1.0, 2.0, 3.0),
                velocity: crate::types::vec3::Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
        ];

        let size = calculate_message_size(&nodes);
        assert_eq!(size, 28);
        
        let encoded = encode_node_data(&nodes);
        assert_eq!(encoded.len(), size);
    }
}
