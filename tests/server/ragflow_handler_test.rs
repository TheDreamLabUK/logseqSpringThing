use crate::handlers::ragflow_handler::{send_message, Message, Response};
use crate::app_state::AppState;
use std::sync::{Arc, RwLock};

#[tokio::test]
async fn test_send_message() {
    let app_state = AppState {
        graph_data: Arc::new(RwLock::new(Default::default())),
        file_cache: Arc::new(RwLock::new(Default::default())),
    };

    let message = Message {
        content: "Test message".to_string(),
    };

    let result = send_message(&app_state, message).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(!response.content.is_empty());
    // Add more specific assertions based on the expected behavior of send_message
}