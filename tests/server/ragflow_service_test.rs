use crate::*;
use services::ragflow_service::RAGFlowService;

#[tokio::test]
async fn test_create_conversation() {
    let user_id = "test_user".to_string();
    let result = RAGFlowService::create_conversation(user_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_send_message() {
    let conversation_id = "test_conversation".to_string();
    let message = "Test message".to_string();
    let result = RAGFlowService::send_message(conversation_id, message).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_get_chat_history() {
    let conversation_id = "test_conversation".to_string();
    let result = RAGFlowService::get_chat_history(conversation_id).await;
    assert!(result.is_ok());
}