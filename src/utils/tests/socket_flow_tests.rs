use std::sync::Arc;
use tokio::sync::RwLock;
use crate::utils::socket_flow_messages::{Node, NodeData};
use crate::models::edge::Edge;
use crate::config::Settings;
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::{ServerMessage, ClientMessage, BinaryNodeData, UpdatePositionsMessage};
use crate::AppState;

#[tokio::test]
async fn test_binary_node_data_conversion() {
    // Create test node data
    let node_data = NodeData {
        position: [1.0, 2.0, 3.0],
        velocity: [0.1, 0.2, 0.3],
        mass: 127,
        flags: 0,
        padding: [0; 2],
    };

    // Create test node
    let node = Node {
        id: "test_node".to_string(),
        label: "Test Node".to_string(),
        data: node_data,
        metadata: Default::default(),
        file_size: 0,
        node_type: None,
        size: None,
        color: None,
        weight: None,
        group: None,
        user_data: None,
    };

    // Convert to BinaryNodeData
    let binary_data = BinaryNodeData::from_node_data(&node.id, &node.data);

    // Verify conversion
    assert_eq!(binary_data.node_id, "test_node");
    assert_eq!(binary_data.data.position, [1.0, 2.0, 3.0]);
    assert_eq!(binary_data.data.velocity, [0.1, 0.2, 0.3]);
    assert_eq!(binary_data.data.mass, 127);
}

#[tokio::test]
async fn test_update_positions_message() {
    // Create test nodes
    let node1_data = NodeData {
        position: [1.0, 2.0, 3.0],
        velocity: [0.1, 0.2, 0.3],
        mass: 127,
        flags: 0,
        padding: [0; 2],
    };

    let node2_data = NodeData {
        position: [4.0, 5.0, 6.0],
        velocity: [0.4, 0.5, 0.6],
        mass: 127,
        flags: 0,
        padding: [0; 2],
    };

    let nodes = vec![
        Node {
            id: "node1".to_string(),
            label: "Node 1".to_string(),
            data: node1_data,
            metadata: Default::default(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        },
        Node {
            id: "node2".to_string(),
            label: "Node 2".to_string(),
            data: node2_data,
            metadata: Default::default(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        },
    ];

    // Create update message
    let binary_nodes: Vec<BinaryNodeData> = nodes.iter()
        .map(|node| BinaryNodeData::from_node_data(&node.id, &node.data))
        .collect();

    let update_msg = UpdatePositionsMessage {
        nodes: binary_nodes,
    };

    // Verify message
    assert_eq!(update_msg.nodes.len(), 2);
    assert_eq!(update_msg.nodes[0].node_id, "node1");
    assert_eq!(update_msg.nodes[0].data.position, [1.0, 2.0, 3.0]);
    assert_eq!(update_msg.nodes[1].node_id, "node2");
    assert_eq!(update_msg.nodes[1].data.position, [4.0, 5.0, 6.0]);
}
