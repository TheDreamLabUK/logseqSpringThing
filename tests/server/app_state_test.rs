use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use logseq_spring_thing::app_state::AppState;
use logseq_spring_thing::models::graph::GraphData;

#[tokio::test]
async fn test_app_state_creation() {
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    
    let app_state = AppState::new(graph_data.clone(), file_cache.clone());
    
    assert!(Arc::ptr_eq(&app_state.graph_data, &graph_data));
    assert!(Arc::ptr_eq(&app_state.file_cache, &file_cache));
}

#[tokio::test]
async fn test_graph_data_access() {
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    
    let app_state = AppState::new(graph_data, file_cache);
    
    // Test writing to graph_data
    {
        let mut graph = app_state.graph_data.write().await;
        graph.add_node(String::from("test_node"), HashMap::new());
    }
    
    // Test reading from graph_data
    {
        let graph = app_state.graph_data.read().await;
        assert!(graph.nodes.contains_key("test_node"));
    }
}

#[tokio::test]
async fn test_file_cache_access() {
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    
    let app_state = AppState::new(graph_data, file_cache);
    
    // Test writing to file_cache
    {
        let mut cache = app_state.file_cache.write().await;
        cache.insert(String::from("test_file"), String::from("test_content"));
    }
    
    // Test reading from file_cache
    {
        let cache = app_state.file_cache.read().await;
        assert_eq!(cache.get("test_file"), Some(&String::from("test_content")));
    }
}

#[tokio::test]
async fn test_concurrent_access() {
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    
    let app_state = Arc::new(AppState::new(graph_data, file_cache));
    
    let handles: Vec<_> = (0..10).map(|i| {
        let app_state = app_state.clone();
        tokio::spawn(async move {
            let mut graph = app_state.graph_data.write().await;
            graph.add_node(format!("node_{}", i), HashMap::new());
            
            let mut cache = app_state.file_cache.write().await;
            cache.insert(format!("file_{}", i), format!("content_{}", i));
        })
    }).collect();
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    let graph = app_state.graph_data.read().await;
    let cache = app_state.file_cache.read().await;
    
    assert_eq!(graph.nodes.len(), 10);
    assert_eq!(cache.len(), 10);
}