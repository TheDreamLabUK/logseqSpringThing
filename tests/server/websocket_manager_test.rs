#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_setup_websocket() {
        let ws_manager = WebSocketManager::new();
        let result = ws_manager.setup_websocket().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_broadcast_message() {
        let ws_manager = WebSocketManager::new();
        let result = ws_manager.broadcast_message("Test message".to_string()).await;
        assert!(result.is_ok());
    }
}