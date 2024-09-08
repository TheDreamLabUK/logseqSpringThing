#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_fetch_and_process_files() {
        let app_state = AppState::new(
            Arc::new(RwLock::new(Default::default())),
            Arc::new(RwLock::new(Default::default())),
        );

        let result = fetch_and_process_files(app_state).await;
        assert!(result.is_ok());
        let processed_files = result.unwrap();
        // Assuming we expect two processed files
        assert_eq!(processed_files.len(), 2);
        assert!(processed_files.contains(&"expected_file_name_1".to_string()));
        assert!(processed_files.contains(&"expected_file_name_2".to_string()));
    }
}