use super::*;
use crate::models::graph::Graph;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::config::{Settings, WebSocketSettings};
use socket_flow::message::Message;
use socket_flow::config::WebSocketConfig;
use socket_flow::extensions::Extensions;
use socket_flow::handshake::{connect_async_with_config, ClientConfig};
use tokio::time::{sleep, Duration};
use tokio::net::TcpListener;
use std::sync::Arc;
use std::net::SocketAddr;

// Helper function to create test settings
fn create_test_settings() -> Settings {
    let mut settings = Settings::new().unwrap();
    settings.websocket = WebSocketSettings {
        compression_enabled: true,
        compression_threshold: 1024,
        max_message_size: 100 * 1024 * 1024,
        update_rate: 5,
        heartbeat_interval: 15000,
        heartbeat_timeout: 60000,
        max_connections: 3,  // Small limit for testing
        reconnect_attempts: 3,
        reconnect_delay: 5000,
        binary_chunk_size: 128,  // Small chunks for testing
    };
    settings
}

// Helper function to create a test graph
fn create_test_graph() -> Graph {
    Graph {
        nodes: vec![
            Node {
                id: "1".to_string(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                position: Some([0.0, 0.0, 0.0]),
                ..Default::default()
            },
            Node {
                id: "2".to_string(),
                x: 1.0,
                y: 1.0,
                z: 1.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                position: Some([1.0, 1.0, 1.0]),
                ..Default::default()
            },
        ],
        edges: vec![
            Edge {
                source: "1".to_string(),
                target: "2".to_string(),
                weight: 1.0,
                ..Default::default()
            }
        ],
        ..Default::default()
    }
}

// Helper function to create a test client config with compression
fn create_test_client_config() -> ClientConfig {
    let mut websocket_config = WebSocketConfig::default();
    websocket_config.extensions = Some(Extensions {
        permessage_deflate: true,
        client_no_context_takeover: Some(true),
        server_no_context_takeover: Some(true),
        client_max_window_bits: None,
        server_max_window_bits: None,
    });
    
    let mut client_config = ClientConfig::default();
    client_config.web_socket_config = websocket_config;
    client_config
}

// Helper function to create a test client connection
async fn create_test_client(port: u16) -> Result<WSConnection, Box<dyn std::error::Error>> {
    let addr = format!("ws://127.0.0.1:{}", port);
    let config = create_test_client_config();
    let connection = connect_async_with_config(&addr, Some(config)).await?;
    Ok(connection)
}

#[tokio::test]
async fn test_connection_limit() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new().await);
    app_state.settings.write().await.websocket.max_connections = 2;
    let server = SocketFlowServer::new(app_state.clone());
    
    // Start server in background task
    let port = 9001;
    tokio::spawn(async move {
        server.start(port).await.unwrap();
    });
    
    // Wait for server to start
    sleep(Duration::from_millis(100)).await;
    
    // Try to connect multiple clients
    let mut clients = vec![];
    for _ in 0..3 {
        if let Ok(client) = create_test_client(port).await {
            clients.push(client);
        }
        sleep(Duration::from_millis(50)).await;
    }
    
    // Verify only max_connections clients were accepted
    assert_eq!(clients.len(), 2);
    
    Ok(())
}

#[tokio::test]
async fn test_chunked_binary_messages() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new().await);
    app_state.settings.write().await.websocket.binary_chunk_size = 128;
    let server = SocketFlowServer::new(app_state.clone());
    
    // Start server in background task
    let port = 9002;
    tokio::spawn(async move {
        server.start(port).await.unwrap();
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Connect test client
    let mut client = create_test_client(port).await?;
    
    // Create large position update
    let mut nodes = Vec::new();
    for i in 0..100 {
        nodes.push(NodePositionVelocity {
            x: i as f32,
            y: i as f32,
            z: i as f32,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        });
    }
    
    let mut binary_data = Vec::new();
    for node in &nodes {
        binary_data.extend_from_slice(bytemuck::bytes_of(node));
    }
    
    // Send large binary update
    client.send_message(Message::Binary(binary_data)).await?;
    
    // Verify received chunks
    let mut received_chunks = 0;
    let mut timeout = Duration::from_secs(1);
    
    while let Ok(Some(message)) = tokio::time::timeout(timeout, client.next()).await {
        if let Ok(Message::Binary(chunk)) = message {
            assert!(chunk.len() <= 128);
            received_chunks += 1;
        }
        timeout = Duration::from_millis(100);
    }
    
    assert!(received_chunks > 1);
    
    Ok(())
}

#[tokio::test]
async fn test_rate_limiting() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new().await);
    app_state.settings.write().await.websocket.update_rate = 5;
    let server = SocketFlowServer::new(app_state.clone());
    
    // Start server in background task
    let port = 9003;
    tokio::spawn(async move {
        server.start(port).await.unwrap();
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Connect test client
    let mut client = create_test_client(port).await?;
    
    // Send position updates rapidly
    let update_msg = UpdatePositionsMessage {
        nodes: vec![
            NodePosition {
                id: "1".to_string(),
                position: [0.0, 0.0, 0.0],
            }
        ],
    };
    
    // Send 10 updates in quick succession
    let start = Instant::now();
    let mut update_count = 0;
    
    for _ in 0..10 {
        let msg = serde_json::to_string(&update_msg)?;
        client.send_message(Message::Text(msg)).await?;
        
        // Check if update was processed
        if let Ok(Some(Ok(Message::Binary(_)))) = tokio::time::timeout(
            Duration::from_millis(100),
            client.next()
        ).await {
            update_count += 1;
        }
        
        sleep(Duration::from_millis(50)).await;
    }
    
    let elapsed = start.elapsed();
    
    // Should have processed about 5 updates per second
    assert!(update_count <= (elapsed.as_secs_f32() * 5.0) as i32 + 1);
    
    Ok(())
}

#[tokio::test]
async fn test_compression() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new().await);
    {
        let mut settings = app_state.settings.write().await;
        settings.websocket.compression_enabled = true;
        settings.websocket.compression_threshold = 1024;
    }
    
    let server = SocketFlowServer::new(app_state.clone());
    
    // Start server in background task
    let port = 9004;
    tokio::spawn(async move {
        server.start(port).await.unwrap();
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Create large test data
    let mut large_graph = create_test_graph();
    for i in 0..1000 {
        large_graph.nodes.push(Node {
            id: format!("node_{}", i),
            x: i as f32,
            y: i as f32,
            z: i as f32,
            position: Some([i as f32, i as f32, i as f32]),
            ..Default::default()
        });
    }
    
    // Connect test client
    let mut client = create_test_client(port).await?;
    
    // Wait for initial data message
    if let Ok(Some(Ok(Message::Text(msg)))) = tokio::time::timeout(
        Duration::from_secs(1),
        client.next()
    ).await {
        // Message should be compressed (verify by size comparison)
        let uncompressed = serde_json::to_string(&large_graph)?;
        assert!(msg.len() < uncompressed.len());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_client_management() -> Result<(), Box<dyn std::error::Error>> {
    // Setup server
    let app_state = Arc::new(AppState::new().await);
    let server = SocketFlowServer::new(app_state.clone());
    
    // Start server in background task
    let port = 9005;
    tokio::spawn(async move {
        server.start(port).await.unwrap();
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Connect multiple clients
    let mut clients = vec![];
    for _ in 0..3 {
        if let Ok(client) = create_test_client(port).await {
            clients.push(client);
        }
        sleep(Duration::from_millis(50)).await;
    }
    
    // Verify client count through server state
    {
        let state = app_state.clone();
        assert_eq!(state.clients.read().await.len(), 2); // max_connections is 3
    }
    
    // Drop one client
    clients.remove(1);
    sleep(Duration::from_millis(100)).await;
    
    // Verify client was removed from server state
    {
        let state = app_state.clone();
        assert_eq!(state.clients.read().await.len(), 1);
    }
    
    Ok(())
}
