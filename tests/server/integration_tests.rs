use webxr_graph::app_state::AppState;
use webxr_graph::models::graph::GraphData;
use webxr_graph::services::file_service::FileService;
use webxr_graph::services::graph_service::GraphService;
use webxr_graph::services::openwebui_service::OpenWebUiService;
use webxr_graph::services::ragflow_service::RAGFlowService;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

#[tokio::test]
async fn test_end_to_end_workflow() {
    // Initialize AppState
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let app_state = Arc::new(AppState::new(graph_data.clone(), file_cache.clone()));

    // Simulate file processing
    let file_content = "Test file content".to_string();
    let processed_content = OpenWebUiService::process_file(file_content.clone()).await.expect("File processing failed");

    // Update file cache
    {
        let mut cache = app_state.file_cache.write().await;
        cache.insert("test_file.md".to_string(), processed_content.clone());
    }

    // Generate graph
    GraphService::generate_graph(app_state.clone(), processed_content).await.expect("Graph generation failed");

    // Verify graph data
    {
        let graph = app_state.graph_data.read().await;
        assert!(graph.nodes.len() > 0, "Graph should contain nodes");
        assert!(graph.edges.len() > 0, "Graph should contain edges");
    }

    // Simulate RAGFlow interaction
    let conversation_id = RAGFlowService::create_conversation("test_user".to_string()).await.expect("Failed to create conversation");
    let response = RAGFlowService::send_message(conversation_id.clone(), "Test message".to_string()).await.expect("Failed to send message");
    assert!(!response.is_empty(), "RAGFlow response should not be empty");

    let chat_history = RAGFlowService::get_chat_history(conversation_id).await.expect("Failed to get chat history");
    assert!(chat_history.len() > 0, "Chat history should not be empty");

    // Verify file cache
    {
        let cache = app_state.file_cache.read().await;
        assert!(cache.contains_key("test_file.md"), "File cache should contain the processed file");
    }
}

#[tokio::test]
async fn test_graph_update_workflow() {
    // Initialize AppState
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let app_state = Arc::new(AppState::new(graph_data.clone(), file_cache.clone()));

    // Simulate initial file processing and graph generation
    let initial_content = "Initial content".to_string();
    let processed_initial = OpenWebUiService::process_file(initial_content.clone()).await.expect("Initial file processing failed");
    GraphService::generate_graph(app_state.clone(), processed_initial).await.expect("Initial graph generation failed");

    // Simulate file update
    let updated_content = "Updated content".to_string();
    let processed_updated = OpenWebUiService::process_file(updated_content.clone()).await.expect("Updated file processing failed");
    
    // Update graph
    GraphService::update_graph(app_state.clone(), "test_file.md".to_string(), processed_updated).await.expect("Graph update failed");

    // Verify updated graph data
    {
        let graph = app_state.graph_data.read().await;
        assert!(graph.nodes.len() > 0, "Updated graph should contain nodes");
        assert!(graph.edges.len() > 0, "Updated graph should contain edges");
        // Add more specific assertions based on expected graph structure after update
    }
}

// Add more integration tests as needed to cover different workflows and edge cases