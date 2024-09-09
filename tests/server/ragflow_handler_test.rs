use crate::*;
use handlers::ragflow_handler::{send_message, Message};
use actix_web::web;

#[actix_web::test]
async fn test_send_message() {
    let app_state = AppState::new(
        Arc::new(RwLock::new(Default::default())),
        Arc::new(RwLock::new(Default::default())),
    );

    let app_state = web::Data::new(app_state);
    let message = web::Json(Message { content: "Test message".to_string() });
    let result = send_message(app_state, message).await;
    assert_eq!(result.status(), actix_web::http::StatusCode::OK);
}