#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_get_graph_data() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(Default::default())),
            Arc::new(RwLock::new(Default::default())),
        );

        let result = GraphService::get_graph_data(&app_state).await;
        assert!(result.is_ok());
        let graph_data = result.unwrap();
        assert!(!graph_data.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_refresh_graph_data() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(Default::default())),
            Arc::new(RwLock::new(Default::default())),
        );

        let result = GraphService::refresh_graph_data(&app_state).await;
        assert!(result.is_ok());
        let graph_data = result.unwrap();
        assert!(!graph_data.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_build_edges() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(Default::default())),
            Arc::new(RwLock::new(Default::default())),
        );

        let result = GraphService::build_edges(&app_state).await;
        assert!(result.is_ok());
        let edges = result.unwrap();
        assert!(!edges.is_empty());
    }
}