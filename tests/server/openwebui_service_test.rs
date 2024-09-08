#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_process_file() {
        let file = "Test content".to_string();
        let result = OpenWebUiService::process_file(file).await;
        assert!(result.is_ok());
        let processed_file = result.unwrap();
        assert!(!processed_file.is_empty());
    }
}