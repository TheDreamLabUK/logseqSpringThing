use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::Cursor;
use crate::utils::socket_flow_messages::BinaryNodeData;

pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("Encoding {} nodes", nodes.len());
    }
    let mut buffer = Vec::new();
    
    for (node_id, node) in nodes {
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Encoding node {}: pos={:?}, vel={:?}", node_id, node.position, node.velocity);
        }
        // Write node ID (u32)
        buffer.write_u32::<LittleEndian>(*node_id).unwrap();
        
        // Write position (3 f32 values)
        for &pos in &node.position {
            buffer.write_f32::<LittleEndian>(pos).unwrap();
        }
        
        // Write velocity (3 f32 values)
        for &vel in &node.velocity {
            buffer.write_f32::<LittleEndian>(vel).unwrap();
        }
    }
    
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("Encoded data size: {} bytes", buffer.len());
    }
    buffer
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    let mut cursor = Cursor::new(data);
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("Decoding binary data of size: {} bytes", data.len());
    }

    let mut updates = Vec::new();
    
    while cursor.position() < data.len() as u64 {
        // Each update is 28 bytes: 4 (nodeId) + 12 (position) + 12 (velocity)
        if cursor.position() + 28 > data.len() as u64 {
            return Err("Unexpected end of data while reading node update".into());
        }
        
        // Read node ID (u32)
        let node_id = cursor.read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read node ID: {}", e))?;
        
        // Read position (3 f32 values)
        let mut position = [0.0; 3];
        for pos in &mut position {
            *pos = cursor.read_f32::<LittleEndian>()
                .map_err(|e| format!("Failed to read position component: {}", e))?;
        }
        
        // Read velocity (3 f32 values)
        let mut velocity = [0.0; 3];
        for vel in &mut velocity {
            *vel = cursor.read_f32::<LittleEndian>()
                .map_err(|e| format!("Failed to read velocity component: {}", e))?;
        }
        
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Decoded node {}: pos={:?}, vel={:?}", node_id, position, velocity);
        }
        
        updates.push((node_id, BinaryNodeData {
            position,
            velocity,
        }));
    }
    
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("Successfully decoded {} nodes", updates.len());
    }
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // Each update: u32 (node_id) + 3*f32 (position) + 3*f32 (velocity) = 4 + 12 + 12 = 28 bytes
    updates.len() * 28
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            (1, BinaryNodeData {
                position: [1.0, 2.0, 3.0],
                velocity: [0.1, 0.2, 0.3],
            }),
            (2, BinaryNodeData {
                position: [4.0, 5.0, 6.0],
                velocity: [0.4, 0.5, 0.6],
            }),
        ];

        let encoded = encode_node_data(&nodes);
        let decoded = decode_node_data(&encoded).unwrap();

        assert_eq!(nodes.len(), decoded.len());

        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id);
            assert_eq!(orig_data.position, dec_data.position);
            assert_eq!(orig_data.velocity, dec_data.velocity);
        }
    }

    #[test]
    fn test_decode_invalid_data() {
        // Test with data that's too short
        let result = decode_node_data(&[0u8; 27]);
        assert!(result.is_err());
    }
}
