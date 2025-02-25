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
            log::debug!(
                "Encoding node {}: pos={:?}, vel={:?}, mass={} (not transmitted)", 
                node_id, node.position, node.velocity, node.mass
            );
        }
        // Write node ID (u32)
        buffer.write_u32::<LittleEndian>(*node_id).unwrap();
        
        // Write position [f32; 3]
        buffer.write_f32::<LittleEndian>(node.position[0]).unwrap();
        buffer.write_f32::<LittleEndian>(node.position[1]).unwrap();
        buffer.write_f32::<LittleEndian>(node.position[2]).unwrap();
        
        // Write velocity [f32; 3]
        buffer.write_f32::<LittleEndian>(node.velocity[0]).unwrap();
        buffer.write_f32::<LittleEndian>(node.velocity[1]).unwrap();
        buffer.write_f32::<LittleEndian>(node.velocity[2]).unwrap();

        // Mass, flags, and padding are no longer sent to the client
        // They are still available in the BinaryNodeData struct for server-side use
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
        
        // Read position [f32; 3]
        let pos_x = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read position[0]: {}", e))?;
        let pos_y = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read position[1]: {}", e))?;
        let pos_z = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read position[2]: {}", e))?;
        
        // Read velocity [f32; 3]
        let vel_x = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read velocity[0]: {}", e))?;
        let vel_y = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read velocity[1]: {}", e))?;
        let vel_z = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read velocity[2]: {}", e))?;

        // Default mass value - this will be replaced with the actual mass from the node_map
        // in socket_flow_handler.rs for accurate physics calculations
        let mass = 100u8; // Default mass
        let flags = 0u8;  // Default flags
        let padding = [0u8, 0u8]; // Default padding
        
        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "Decoded node {}: pos=({:.3},{:.3},{:.3}), vel=({:.3},{:.3},{:.3}), default_mass={} (will be replaced)", 
                node_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass
            );
        }
        
        updates.push((node_id, BinaryNodeData {
            position: [pos_x, pos_y, pos_z],
            velocity: [vel_x, vel_y, vel_z],
            mass,
            flags,
            padding,
        }));
    }
    
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("Successfully decoded {} nodes", updates.len());
    }
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // Each update: u32 (node_id) + 3*f32 (position) + 3*f32 (velocity)
    // = 4 + 12 + 12 = 28 bytes
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
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
            (2, BinaryNodeData {
                position: [4.0, 5.0, 6.0],
                velocity: [0.4, 0.5, 0.6],
                mass: 200,
                flags: 1,
                padding: [0, 0],
            }),
        ];

        let encoded = encode_node_data(&nodes);
        let decoded = decode_node_data(&encoded).unwrap();

        assert_eq!(nodes.len(), decoded.len());

        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id);
            assert_eq!(orig_data.position, dec_data.position);
            assert_eq!(orig_data.velocity, dec_data.velocity);
            // Note: mass, flags, and padding are not compared as they're not transmitted
        }
    }

    #[test]
    fn test_decode_invalid_data() {
        // Test with data that's too short
        let result = decode_node_data(&[0u8; 27]);
        assert!(result.is_err());
    }
}
