#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_fetch_files_from_github() {
        let result = FileService::fetch_files_from_github().await;
        assert!(result.is_ok());
        let files = result.unwrap();
        assert!(!files.is_empty());
    }

    #[tokio::test]
    async fn test_compare_and_identify_updates() {
        let github_files = vec![GithubFile { name: "test1.md".to_string(), content: "Test content 1".to_string() }];
        let result = FileService::compare_and_identify_updates(github_files).unwrap();
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_send_to_openwebui() {
        let file = "Test content for OpenWebUI".to_string();
        let result = FileService::send_to_openwebui(file).await;
        assert!(result.is_ok());
        let processed_file = result.unwrap();
        assert!(!processed_file.is_empty());
    }

    #[tokio::test]
    async fn test_save_file_metadata() {
        let metadata = Metadata::default();
        let result = FileService::save_file_metadata(metadata);
        assert!(result.is_ok());
    }
}