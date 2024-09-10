use mockall::predicate::*;
use mockall::automock;
use async_trait::async_trait;

#[automock]
#[async_trait]
trait RAGFlowServiceTrait {
    async fn create_conversation(&self, user_id: String) -> Result<String, Box<dyn std::error::Error>>;
    async fn send_message(&self, conversation_id: String, message: String) -> Result<String, Box<dyn std::error::Error>>;
    async fn get_chat_history(&self, conversation_id: String) -> Result<Vec<String>, Box<dyn std::error::Error>>;
}

#[tokio::test]
async fn test_create_conversation_success() {
    let mut mock = MockRAGFlowServiceTrait::new();
    mock.expect_create_conversation()
        .with(eq("test_user".to_string()))
        .times(1)
        .returning(|_| Ok("new_conversation_id".to_string()));

    let result = mock.create_conversation("test_user".to_string()).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "new_conversation_id");
}

#[tokio::test]
async fn test_create_conversation_error() {
    let mut mock = MockRAGFlowServiceTrait::new();
    mock.expect_create_conversation()
        .with(eq("test_user".to_string()))
        .times(1)
        .returning(|_| Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "API Error"))));

    let result = mock.create_conversation("test_user".to_string()).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "API Error");
}

#[tokio::test]
async fn test_send_message_success() {
    let mut mock = MockRAGFlowServiceTrait::new();
    mock.expect_send_message()
        .with(eq("test_conversation".to_string()), eq("Test message".to_string()))
        .times(1)
        .returning(|_, _| Ok("Message sent successfully".to_string()));

    let result = mock.send_message("test_conversation".to_string(), "Test message".to_string()).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Message sent successfully");
}

#[tokio::test]
async fn test_send_message_error() {
    let mut mock = MockRAGFlowServiceTrait::new();
    mock.expect_send_message()
        .with(eq("test_conversation".to_string()), eq("Test message".to_string()))
        .times(1)
        .returning(|_, _| Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Message send failed"))));

    let result = mock.send_message("test_conversation".to_string(), "Test message".to_string()).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "Message send failed");
}

#[tokio::test]
async fn test_get_chat_history_success() {
    let mut mock = MockRAGFlowServiceTrait::new();
    mock.expect_get_chat_history()
        .with(eq("test_conversation".to_string()))
        .times(1)
        .returning(|_| Ok(vec!["Message 1".to_string(), "Message 2".to_string()]));

    let result = mock.get_chat_history("test_conversation".to_string()).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec!["Message 1".to_string(), "Message 2".to_string()]);
}

#[tokio::test]
async fn test_get_chat_history_error() {
    let mut mock = MockRAGFlowServiceTrait::new();
    mock.expect_get_chat_history()
        .with(eq("test_conversation".to_string()))
        .times(1)
        .returning(|_| Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Failed to fetch chat history"))));

    let result = mock.get_chat_history("test_conversation".to_string()).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "Failed to fetch chat history");
}