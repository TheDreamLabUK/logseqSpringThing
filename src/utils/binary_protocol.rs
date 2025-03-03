use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::Cursor;
use crate::utils::socket_flow_messages::BinaryNodeData;
use log;

// Binary format (simplified):
// - For each node (24 bytes):
//   - Node Index: 2 bytes (u16)
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
// Total: 26 bytes per node

pub fn encode_node_data(nodes: &[(u16, BinaryNodeData)]) -> Vec<u8> {
    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        log::info!("Encoding {} nodes for binary transmission", nodes.len());
    }
    
    let mut buffer = Vec::new();
    
    // Log some samples of the encoded data
    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        log::info!("Sample of nodes being encoded:");
    }
    
    for (node_id, node) in nodes {
        // Log the first few nodes for debugging
        if sample_size > 0 && *node_id < sample_size as u16 {
            log::info!("Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]", 
                node_id, 
                node.position.x, node.position.y, node.position.z,
                node.velocity.x, node.velocity.y, node.velocity.z);
        }
        // Write node ID (u16)
        buffer.write_u16::<LittleEndian>(*node_id).unwrap();
        
        // Write position Vec3Data
        buffer.write_f32::<LittleEndian>(node.position.x).unwrap();
        buffer.write_f32::<LittleEndian>(node.position.y).unwrap();
        buffer.write_f32::<LittleEndian>(node.position.z).unwrap();
        
        // Write velocity Vec3Data
        buffer.write_f32::<LittleEndian>(node.velocity.x).unwrap();
        buffer.write_f32::<LittleEndian>(node.velocity.y).unwrap();
        buffer.write_f32::<LittleEndian>(node.velocity.z).unwrap();

        // Mass, flags, and padding are no longer sent to the client
        // They are still available in the BinaryNodeData struct for server-side use
    }

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        log::info!("Encoded binary data: {} bytes for {} nodes", buffer.len(), nodes.len());
    }
    buffer
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u16, BinaryNodeData)>, String> {
    let mut cursor = Cursor::new(data);
    
    // Check if data is empty
    if data.len() < 2 { // At least a node ID (2 bytes)
        return Err("Data too small to contain any nodes".into());
    }
    
    // Log header information
    log::info!(
        "Decoding binary data: size={} bytes, expected nodes={}",
        data.len(), data.len() / 26
    );
    
    // Always log this for visibility
    log::info!("Decoding binary data of size: {} bytes", data.len());
    
    let mut updates = Vec::new();
    
    // Set up sample logging
    let max_samples = 3;
    let mut samples_logged = 0;
    
    log::info!("Starting binary data decode, expecting nodes with position and velocity data");
    
    while cursor.position() < data.len() as u64 {
        // Each node update is 26 bytes: 2 (nodeId) + 12 (position) + 12 (velocity)
        if cursor.position() + 26 > data.len() as u64 {
            return Err("Unexpected end of data while reading node update".into());
        }
        
        // Read node ID (u16)
        let node_id = cursor.read_u16::<LittleEndian>()
            .map_err(|e| format!("Failed to read node ID: {}", e))?;
        
        // Read position Vec3Data
        let pos_x = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read position[0]: {}", e))?;
        let pos_y = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read position[1]: {}", e))?;
        let pos_z = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read position[2]: {}", e))?;

        let position = crate::types::vec3::Vec3Data::new(pos_x, pos_y, pos_z);
        
        // Read velocity Vec3Data
        let vel_x = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read velocity[0]: {}", e))?;
        let vel_y = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read velocity[1]: {}", e))?;
        let vel_z = cursor.read_f32::<LittleEndian>()
            .map_err(|e| format!("Failed to read velocity[2]: {}", e))?;
        let velocity = crate::types::vec3::Vec3Data::new(vel_x, vel_y, vel_z);

        // Default mass value - this will be replaced with the actual mass from the node_map
        // in socket_flow_handler.rs for accurate physics calculations
        let mass = 100u8; // Default mass
        let flags = 0u8;  // Default flags
        let padding = [0u8, 0u8]; // Default padding
        
        // Log the first few decoded items as samples
        if samples_logged < max_samples {
            log::info!(
                "Decoded node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]", 
                node_id, position.x, position.y, position.z, 
                velocity.x, velocity.y, velocity.z
            );
            samples_logged += 1;
        }
        updates.push((node_id, BinaryNodeData {
            position,
            velocity,
            mass,
            flags,
            padding,
        }));
    }
    
    log::info!("Successfully decoded {} nodes from binary data", updates.len());
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u16, BinaryNodeData)]) -> usize {
    // Each update: u16 (node_id) + 3*f32 (position) + 3*f32 (velocity)
    updates.len() * 26
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            (1u16, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(1.0, 2.0, 3.0),
                velocity: crate::types::vec3::Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
            (2u16, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(4.0, 5.0, 6.0),
                velocity: crate::types::vec3::Vec3Data::new(0.4, 0.5, 0.6),
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
        let result = decode_node_data(&[0u8; 25]);
        assert!(result.is_err());
    }
}
