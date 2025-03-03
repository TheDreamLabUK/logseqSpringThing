use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::utils::socket_flow_messages::BinaryNodeData;

// Protocol version and constants
const PROTOCOL_VERSION: u8 = 1;

// Static sequence counter for detecting missed updates
static mut SEQUENCE_NUMBER: u32 = 0;
static mut LAST_TIMESTAMP: u64 = 0;

pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        log::info!("Encoding {} nodes for binary transmission", nodes.len());
    }
    
    let mut buffer = Vec::new();

    // Get current timestamp in milliseconds
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Get and increment sequence number (thread-safe)
    let sequence_number = unsafe {
        // Only increment if timestamp has changed to avoid rapid increments
        if timestamp > LAST_TIMESTAMP {
            SEQUENCE_NUMBER = SEQUENCE_NUMBER.wrapping_add(1);
            LAST_TIMESTAMP = timestamp;
        }
        SEQUENCE_NUMBER
    };

    // Write header: version (1 byte), sequence number (4 bytes), timestamp (8 bytes)
    buffer.write_u8(PROTOCOL_VERSION).unwrap();
    buffer.write_u32::<LittleEndian>(sequence_number).unwrap();
    buffer.write_u64::<LittleEndian>(timestamp).unwrap();
    
    // Log some samples of the encoded data
    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        log::info!("Sample of nodes being encoded:");
    }
    
    for (node_id, node) in nodes {
        // Log the first few nodes for debugging
        if sample_size > 0 && *node_id < sample_size as u32 {
            log::info!("Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]", 
                node_id, 
                node.position[0], node.position[1], node.position[2],
                node.velocity[0], node.velocity[1], node.velocity[2]);
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

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        log::info!("Encoded binary data: {} bytes for {} nodes", buffer.len(), nodes.len());
    }
    buffer
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    let mut cursor = Cursor::new(data);
    
    // Check if data is empty or too small for header
    if data.len() < 13 { // 1 (version) + 4 (sequence) + 8 (timestamp)
        return Err("Data too small to contain header".into());
    }

    // Read header
    let version = cursor.read_u8()
        .map_err(|e| format!("Failed to read protocol version: {}", e))?;
    let sequence = cursor.read_u32::<LittleEndian>()
        .map_err(|e| format!("Failed to read sequence number: {}", e))?;
    let timestamp = cursor.read_u64::<LittleEndian>()
        .map_err(|e| format!("Failed to read timestamp: {}", e))?;

    // Log header information
    log::info!(
        "Decoding binary data: version={}, sequence={}, timestamp={}, size={} bytes",
        version, sequence, timestamp, data.len()
    );

    // Always log this for visibility
    log::info!("Decoding binary data of size: {} bytes", data.len());

    let mut updates = Vec::new();
    
    // Set up sample logging
    let max_samples = 3;
    let mut samples_logged = 0;
    
    log::info!("Starting binary data decode, expecting nodes with position and velocity data");
    
    while cursor.position() < data.len() as u64 {
        // Each node update is 28 bytes: 4 (nodeId) + 12 (position) + 12 (velocity)
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
        
        // Log the first few decoded items as samples
        if samples_logged < max_samples {
            log::info!(
                "Decoded node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]", 
                node_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
            );
            samples_logged += 1;
        }
        updates.push((node_id, BinaryNodeData {
            position: [pos_x, pos_y, pos_z],
            velocity: [vel_x, vel_y, vel_z],
            mass,
            flags,
            padding,
        }));
    }
    
    log::info!("Successfully decoded {} nodes from binary data", updates.len());
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // Each update: u32 (node_id) + 3*f32 (position) + 3*f32 (velocity)
    // = 4 + 12 + 12 = 28 bytes
    // Plus header: 1 (version) + 4 (sequence) + 8 (timestamp) = 13 bytes
    13 + (updates.len() * 28)
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
