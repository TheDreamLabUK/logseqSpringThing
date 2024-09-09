use crate::*;
use handlers::graph_handler::{get_graph_data, refresh_graph};
use actix_web::web;

#[actix_web::test]
async fn test_get_graph_data() {
    let app_state = AppState::new(
        Arc::new(RwLock::new(GraphData {
            nodes: vec![Node { id: "1".to_string(), label: "Test Node".to_string(), metadata: Default::default() }],
            edges: vec![Edge { source: "1".to_string(), target: "2".to_string() }],
        })),
        Arc::new(RwLock::new(Default::default())),
    );

    let app_state = web::Data::new(app_state);
    let result = get_graph_data(app_state).await;
    assert_eq!(result.status(), actix_web::http::StatusCode::OK);
}

#[actix_web::test]
async fn test_refresh_graph() {
    let app_state = AppState::new(
        Arc::new(RwLock::new(GraphData {
            nodes: vec![],
            edges: vec![],
        })),
        Arc::new(RwLock::new(Default::default())),
    );

    let app_state = web::Data::new(app_state);
    let result = refresh_graph(app_state).await;
    assert_eq!(result.status(), actix_web::http::StatusCode::OK);
}