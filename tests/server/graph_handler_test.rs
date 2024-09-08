#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_get_graph_data() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(GraphData {
                nodes: vec![Node { id: "1".to_string(), label: "Test Node", metadata: Default::default() }],
                edges: vec![Edge { source: "1", target: "2" }],
            })),
            Arc::new(RwLock::new(Default::default())),
        );

        let result = get_graph_data(app_state).await;
        assert!(result.is_ok());
        let graph_data = result.unwrap();
        assert_eq!(graph_data.nodes.len(), 1);
        assert_eq!(graph_data.edges.len(), 1);
    }

    #[tokio::test]
    async fn test_refresh_graph() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(GraphData {
                nodes: vec![],
                edges: vec![],
            })),
            Arc::new(RwLock::new(Default::default())),
        );

        let result = refresh_graph(app_state).await;
        assert!(result.is_ok());
        let graph_data = result.unwrap();
        assert!(graph_data.nodes.len() > 0);
        assert!(graph_data.edges.len() > 0);
    }
}