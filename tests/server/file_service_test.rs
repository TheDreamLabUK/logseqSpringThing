use crate::services::file_service::{
    fetch_files_from_github,
    compare_and_identify_updates,
    send_to_openwebui,
    save_file_metadata,
    GithubFile,
    ProcessedFile,
};
use crate::models::metadata::Metadata;

#[tokio::test]
async fn test_fetch_files_from_github() {
    let result = fetch_files_from_github().await;
    assert!(result.is_ok());
    let files = result.unwrap();
    assert!(!files.is_empty());
    // Add more specific assertions
}

#[tokio::test]
async fn test_compare_and_identify_updates() {
    let github_files = vec![
        GithubFile {
            name: "test1.md".to_string(),
            content: "Test content 1".to_string(),
        },
        GithubFile {
            name: "test2.md".to_string(),
            content: "Test content 2".to_string(),
        },
    ];
    let result = compare_and_identify_updates(github_files).await;
    assert!(result.is_ok());
    let updates = result.unwrap();
    // Add assertions based on expected updates
}

#[tokio::test]
async fn test_send_to_openwebui() {
    let file = "Test content for OpenWebUI".to_string();
    let result = send_to_openwebui(file).await;
    assert!(result.is_ok());
    let processed_file = result.unwrap();
    assert!(!processed_file.content.is_empty());
    // Add more assertions for processed file and metadata
}

#[tokio::test]
async fn test_save_file_metadata() {
    let metadata = Metadata {
        file_name: "test.md".to_string(),
        last_modified: chrono::Utc::now(),
        processed_file: "Processed content".to_string(),
        original_file: "Original content".to_string(),
    };
    let result = save_file_metadata(metadata).await;
    assert!(result.is_ok());
    // Add more assertions if needed
}