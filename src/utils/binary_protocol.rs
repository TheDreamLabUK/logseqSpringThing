use byteorder::{ByteOrder, LittleEndian};

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

#[derive(Debug)]
pub struct NodeData {
    pub id: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
}

pub fn encode_node_data(nodes: &[NodeData], msg_type: MessageType) -> Vec<u8> {
    // Calculate total size needed
    let mut total_size = 8; // 4 bytes for msg_type + 4 bytes for node count
    for node in nodes {
        total_size += 4 + node.id.len() + 24; // 4 bytes for id length + id bytes + 24 bytes for position and velocity
    }
    
    let mut buffer = vec![0u8; total_size];
    
    // Write header (message type)
    let mut offset = 0;
    
    // Write message type
    LittleEndian::write_u32(&mut buffer[offset..offset + 4], msg_type as u32);
    offset += 4;
    
    // Write number of nodes
    LittleEndian::write_u32(&mut buffer[offset..offset + 4], nodes.len() as u32);
    offset += 4;
    
    for node in nodes {
        // Write node ID length
        LittleEndian::write_u32(&mut buffer[offset..offset + 4], node.id.len() as u32);
        offset += 4;
        
        // Write node ID content
        buffer[offset..offset + node.id.len()].copy_from_slice(node.id.as_bytes());
        offset += node.id.len();
        
        // Write position
        for &pos in &node.position {
            LittleEndian::write_f32(&mut buffer[offset..offset + 4], pos);
            offset += 4;
        }
        
        // Write velocity
        for &vel in &node.velocity {
            LittleEndian::write_f32(&mut buffer[offset..offset + 4], vel);
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
        // Check if we have enough bytes for ID length
        if offset + 4 > data.len() {
            return Err("Unexpected end of data while reading ID length".into());
        }
        
        // Read node ID length
        let id_len = LittleEndian::read_u32(&data[offset..offset + 4]) as usize;
        offset += 4;
        
        // Check if we have enough bytes for ID content
        if offset + id_len > data.len() {
            return Err("Unexpected end of data while reading ID content".into());
        }
        
        // Read node ID
        let id_bytes = &data[offset..offset + id_len];
        let id = String::from_utf8(id_bytes.to_vec())
            .map_err(|e| format!("Invalid UTF-8 in node ID: {}", e))?;
        offset += id_len;
        
        // Check if we have enough bytes for position and velocity (24 bytes)
        if offset + 24 > data.len() {
            return Err("Unexpected end of data while reading position/velocity".into());
        }
        
        // Read position
        let mut position = [0.0; 3];
        for pos in &mut position {
            *pos = LittleEndian::read_f32(&data[offset..offset + 4]);
            offset += 4;
        }
        
        // Read velocity
        let mut velocity = [0.0; 3];
        for vel in &mut velocity {
            *vel = LittleEndian::read_f32(&data[offset..offset + 4]);
            offset += 4;
        }
        
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
    for node in nodes {
        size += 4 + node.id.len() + 24; // 4 bytes for id length + id bytes + 24 bytes for position and velocity
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            NodeData {
                id: "node1".to_string(),
                position: [1.0, 2.0, 3.0],
                velocity: [0.1, 0.2, 0.3],
            },
            NodeData {
                id: "node2".to_string(),
                position: [4.0, 5.0, 6.0],
                velocity: [0.4, 0.5, 0.6],
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
