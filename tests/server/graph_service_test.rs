#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_get_graph_data() {
        let app_state = AppState {
            graph_data: Arc::new(RwLock::new(GraphData {
                nodes: vec![],
                edges: vec![],
            })),
            file_cache: Arc::new(RwLock::new(Default::default())),
        };

        let result = get_graph_data(&app_state).await;
        assert!(result.is_ok());
        let graph_data = result.unwrap();
        // Add assertions based on expected graph data
    }

    #[tokio::test]
    async fn test_refresh_graph_data() {
        let app_state = AppState {
            graph_data: Arc::new(RwLock::new(GraphData {
                nodes: vec![],
                edges: vec![],
            })),
            file_cache: Arc::new(RwLock::new(Default::default())),
        };

        let result = refresh_graph_data(&app_state).await;
        assert!(result.is_ok());
        let graph_data = result.unwrap();
        // Add assertions based on expected refreshed graph data
    }

    #[tokio::test]
    async fn test_build_edges() {
        let app_state = AppState {
            graph_data: Arc::new(RwLock::new(GraphData {
                nodes: vec![],
                edges: vec![],
            })),
            file_cache: Arc::new(RwLock::new(Default::default())),
        };

        let result = build_edges(&app_state).await;
        assert!(result.is_ok());
        let edges = result.unwrap();
        // Add assertions based on expected edges
    }
}