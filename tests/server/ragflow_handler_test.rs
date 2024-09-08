#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_send_message() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(Default::default())),
            Arc::new(RwLock::new(Default::default())),
        );

        let message = Message { content: "Test message".to_string() };

        let result = send_message(app_state, message).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.content.is_empty());
    }
}