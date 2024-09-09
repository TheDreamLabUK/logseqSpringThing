use crate::*;
use services::graph_service::GraphService;

#[tokio::test]
async fn test_get_graph_data() {
    let app_state = AppState::new(
        Arc::new(RwLock::new(Default::default())),
        Arc::new(RwLock::new(Default::default())),
    );

    let result = GraphService::get_graph_data(&app_state);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_refresh_graph_data() {
    let app_state = AppState::new(
        Arc::new(RwLock::new(Default::default())),
        Arc::new(RwLock::new(Default::default())),
    );

    let result = GraphService::refresh_graph_data(&app_state);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_build_edges() {
    let app_state = AppState::new(
        Arc::new(RwLock::new(Default::default())),
        Arc::new(RwLock::new(Default::default())),
    );

    let result = GraphService::build_edges(&app_state);
    assert!(result.is_ok());
}