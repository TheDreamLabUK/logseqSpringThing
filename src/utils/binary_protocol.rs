use byteorder::{ByteOrder, LittleEndian};
use glam::Vec3;

#[derive(Debug)]
pub enum MessageType {
    PositionUpdate = 0x01,
    VelocityUpdate = 0x02,
    FullStateUpdate = 0x03,
}

impl TryFrom<u32> for MessageType {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(MessageType::PositionUpdate),
            0x02 => Ok(MessageType::VelocityUpdate),
            0x03 => Ok(MessageType::FullStateUpdate),
            _ => Err(format!("Invalid message type: {}", value)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: u32,
    pub position: Vec3,  // 12 bytes (3 × f32)
    pub velocity: Vec3,  // 12 bytes (3 × f32)
}
// Total: 28 bytes per node (4 byte header + 24 bytes data)

pub fn encode_node_data(nodes: &[NodeData], msg_type: MessageType) -> Vec<u8> {
    // Calculate total size needed
    let mut total_size = 8; // 4 bytes for msg_type + 4 bytes for node count
    total_size += nodes.len() * 28; // 4 bytes for id + 24 bytes for position and velocity
    
    let mut buffer = vec![0u8; total_size];
    let mut offset = 0;
    
    // Write message type
    LittleEndian::write_u32(&mut buffer[offset..offset + 4], msg_type as u32);
    offset += 4;
    
    // Write number of nodes
    LittleEndian::write_u32(&mut buffer[offset..offset + 4], nodes.len() as u32);
    offset += 4;
    
    for node in nodes.iter() {
        // Write node ID
        LittleEndian::write_u32(&mut buffer[offset..offset + 4], node.id);
        offset += 4;
        
        // Write position
        for component in [node.position.x, node.position.y, node.position.z].iter() {
            LittleEndian::write_f32(&mut buffer[offset..offset + 4], *component);
            offset += 4;
        }
        
        // Write velocity
        for component in [node.velocity.x, node.velocity.y, node.velocity.z].iter() {
            LittleEndian::write_f32(&mut buffer[offset..offset + 4], *component);
            offset += 4;
        }
    }
    
    buffer
}

pub fn decode_node_data(data: &[u8]) -> Result<(MessageType, Vec<NodeData>), String> {
    // Read header
    if data.len() < 8 {
        return Err("Data buffer too small for header".into());
    }

    let mut offset = 0;

    // Read message type
    let msg_type = LittleEndian::read_u32(&data[offset..offset + 4]);
    offset += 4;
    let msg_type = MessageType::try_from(msg_type)?;
    
    // Read number of nodes
    let node_count = LittleEndian::read_u32(&data[offset..offset + 4]) as usize;
    offset += 4;
    
    let mut nodes = Vec::with_capacity(node_count);
    
    for _ in 0..node_count {
        // Check if we have enough bytes for the node (28 bytes: 4 for id + 24 for position/velocity)
        if offset + 28 > data.len() {
            return Err("Unexpected end of data while reading node".into());
        }
        
        // Read node ID
        let id = LittleEndian::read_u32(&data[offset..offset + 4]);
        offset += 4;
        
        // Read position
        let position = Vec3::new(
            LittleEndian::read_f32(&data[offset..offset + 4]),
            LittleEndian::read_f32(&data[offset + 4..offset + 8]),
            LittleEndian::read_f32(&data[offset + 8..offset + 12])
        );
        offset += 12;
        
        // Read velocity
        let velocity = Vec3::new(
            LittleEndian::read_f32(&data[offset..offset + 4]),
            LittleEndian::read_f32(&data[offset + 4..offset + 8]),
            LittleEndian::read_f32(&data[offset + 8..offset + 12])
        );
        offset += 12;
        
        nodes.push(NodeData {
            id,
            position,
            velocity,
        });
    }
    
    Ok((msg_type, nodes))
}

pub fn calculate_message_size(nodes: &[NodeData]) -> usize {
    let mut size = 8; // 4 bytes for msg_type + 4 bytes for node count
    size += nodes.len() * 28; // 28 bytes per node (4 for id + 24 for position/velocity)
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            NodeData {
                id: 1,
                position: Vec3::new(1.0, 2.0, 3.0),
                velocity: Vec3::new(0.1, 0.2, 0.3),
            },
            NodeData {
                id: 2,
                position: Vec3::new(4.0, 5.0, 6.0),
                velocity: Vec3::new(0.4, 0.5, 0.6),
            },
        ];

        let encoded = encode_node_data(&nodes, MessageType::FullStateUpdate);
        let (msg_type, decoded) = decode_node_data(&encoded).unwrap();

        assert!(matches!(msg_type, MessageType::FullStateUpdate));
        assert_eq!(nodes.len(), decoded.len());

        for (original, decoded) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(original.id, decoded.id);
            assert_eq!(original.position, decoded.position);
            assert_eq!(original.velocity, decoded.velocity);
        }
    }
}
