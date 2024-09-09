use crate::*;
use services::file_service::{FileService, GithubFile};

#[tokio::test]
async fn test_fetch_files_from_github() {
    let result = FileService::fetch_files_from_github().await;
    assert!(result.is_ok());
}

#[test]
fn test_compare_and_identify_updates() {
    let github_files = vec![
        GithubFile { name: "test1.md".to_string(), content: "Test content 1".to_string() },
        GithubFile { name: "test2.md".to_string(), content: "Test content 2".to_string() },
    ];
    let result = FileService::compare_and_identify_updates(github_files).unwrap();
    assert_eq!(result.len(), 2);
    assert!(result.contains(&"test1.md".to_string()));
    assert!(result.contains(&"test2.md".to_string()));
}

#[tokio::test]
async fn test_send_to_openwebui() {
    let file = "Test file content".to_string();
    let result = FileService::send_to_openwebui(file).await;
    assert!(result.is_ok());
}

#[test]
fn test_save_file_metadata() {
    let metadata = Metadata::default();
    let result = FileService::save_file_metadata(metadata);
    assert!(result.is_ok());
}