#[cfg(test)]
mod tests {
    use super::*;
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
    }
}