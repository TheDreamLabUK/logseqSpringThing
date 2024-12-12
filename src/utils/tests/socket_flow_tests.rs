use super::*;
use crate::models::graph::GraphData;
use crate::models::node::{Node, NodeData};
use crate::models::edge::Edge;
use crate::config::Settings;
use crate::utils::socket_flow_messages::{ServerMessage, ClientMessage, BinaryNodeData, UpdatePositionsMessage};
use crate::AppState;
use actix::Actor;
use actix_web_actors::ws;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::{sleep, Duration};

// Helper function to create test settings
fn create_test_settings() -> Settings {
    Settings::new().unwrap()
}

// Helper function to create a test graph
fn create_test_graph() -> GraphData {
    GraphData {
        nodes: vec![
            Node {
                id: "1".to_string(),
                label: "Node 1".to_string(),
                data: NodeData {
                    position: [0.0, 0.0, 0.0],
                    velocity: [0.0, 0.0, 0.0],
                    mass: 127,
                    flags: 0,
                    padding: [0; 2],
                },
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
                id: "2".to_string(),
                label: "Node 2".to_string(),
                data: NodeData {
                    position: [1.0, 1.0, 1.0],
                    velocity: [0.0, 0.0, 0.0],
                    mass: 127,
                    flags: 0,
                    padding: [0; 2],
                },
                metadata: Default::default(),
                file_size: 0,
                node_type: None,
                size: None,
                color: None,
                weight: None,
                group: None,
                user_data: None,
            },
        ],
        edges: vec![
            Edge {
                source: "1".to_string(),
                target: "2".to_string(),
                weight: Some(1.0),
                ..Default::default()
            }
        ],
        metadata: Default::default(),
    }
}

#[tokio::test]
async fn test_binary_message_format() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new(
        Arc::new(tokio::sync::RwLock::new(create_test_settings())),
        Default::default(),
        Default::default(),
        Default::default(),
        None,
        None,
        String::new(),
        Default::default(),
    ));

    let server = SocketFlowServer::new(app_state.clone());
    
    // Create test data
    let binary_nodes: Vec<BinaryNodeData> = vec![
        BinaryNodeData::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        BinaryNodeData::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
    ];

    // Create binary message
    let mut binary_data = Vec::with_capacity(BINARY_HEADER_SIZE + binary_nodes.len() * NODE_POSITION_SIZE);
    binary_data.extend_from_slice(&1.0f32.to_le_bytes());
    for node in &binary_nodes {
        binary_data.extend_from_slice(bytemuck::bytes_of(node));
    }

    // Verify binary format
    assert_eq!(binary_data.len(), BINARY_HEADER_SIZE + 2 * NODE_POSITION_SIZE);
    assert_eq!(std::mem::size_of::<BinaryNodeData>(), NODE_POSITION_SIZE);

    Ok(())
}

#[tokio::test]
async fn test_position_updates() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new(
        Arc::new(tokio::sync::RwLock::new(create_test_settings())),
        Default::default(),
        Default::default(),
        Default::default(),
        None,
        None,
        String::new(),
        Default::default(),
    ));

    let server = SocketFlowServer::new(app_state.clone());
    
    // Create test position update
    let update_msg = UpdatePositionsMessage {
        nodes: vec![
            crate::utils::socket_flow_messages::Node {
                id: "1".to_string(),
                position: [1.0, 2.0, 3.0],
                velocity: Some([0.0, 0.0, 0.0]),
                data: None,
            }
        ],
    };

    // Verify message serialization
    let json = serde_json::to_string(&ClientMessage::UpdatePositions(update_msg))?;
    assert!(json.contains("updatePositions"));
    assert!(json.contains("position"));

    Ok(())
}

#[tokio::test]
async fn test_compression() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new(
        Arc::new(tokio::sync::RwLock::new(create_test_settings())),
        Default::default(),
        Default::default(),
        Default::default(),
        None,
        None,
        String::new(),
        Default::default(),
    ));

    let server = SocketFlowServer::new(app_state.clone());
    
    // Create large test data
    let mut large_graph = create_test_graph();
    for i in 0..1000 {
        large_graph.nodes.push(Node {
            id: format!("node_{}", i),
            label: format!("Node {}", i),
            data: NodeData {
                position: [i as f32, i as f32, i as f32],
                velocity: [0.0, 0.0, 0.0],
                mass: 127,
                flags: 0,
                padding: [0; 2],
            },
            metadata: Default::default(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        });
    }

    // Verify data serialization and compression
    let json = serde_json::to_string(&ServerMessage::GraphUpdate { graph_data: large_graph })?;
    assert!(json.len() > 1024); // Should be large enough to trigger compression

    Ok(())
}

#[tokio::test]
async fn test_binary_node_data() {
    // Verify BinaryNodeData memory layout
    assert_eq!(std::mem::size_of::<BinaryNodeData>(), 24);
    assert_eq!(std::mem::align_of::<BinaryNodeData>(), 4);

    // Test conversion from NodeData
    let node_data = NodeData {
        position: [1.0, 2.0, 3.0],
        velocity: [0.1, 0.2, 0.3],
        mass: 127,
        flags: 1,
        padding: [0; 2],
    };

    let binary_data = BinaryNodeData::from_node_data(&node_data);
    assert_eq!(binary_data.position, node_data.position);
    assert_eq!(binary_data.velocity, node_data.velocity);
}
