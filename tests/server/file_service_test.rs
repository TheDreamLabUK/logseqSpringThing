use webxr_graph::services::file_service::{FileService, GithubFile, GitHubService, OpenWebUIService, ProcessedFile};
use webxr_graph::models::metadata::Metadata;
use mockall::predicate::*;
use mockall::mock;
use async_trait::async_trait;

mock! {
    pub GitHubServiceMock {}

    #[async_trait]
    impl GitHubService for GitHubServiceMock {
        async fn fetch_files() -> Result<Vec<GithubFile>, reqwest::Error>;
    }
}

mock! {
    pub OpenWebUIServiceMock {}

    #[async_trait]
    impl OpenWebUIService for OpenWebUIServiceMock {
        async fn process_file(file: String) -> Result<ProcessedFile, reqwest::Error>;
    }
}

#[tokio::test]
async fn test_fetch_files_from_github() {
    let ctx = MockGitHubServiceMock::fetch_files_context();
    ctx.expect()
        .times(1)
        .returning(|| Ok(vec![
            GithubFile { name: "test1.md".to_string(), content: "Test content 1".to_string() },
            GithubFile { name: "test2.md".to_string(), content: "Test content 2".to_string() },
        ]));

    let result = FileService::fetch_files_from_github::<MockGitHubServiceMock>().await;
    assert!(result.is_ok());
    let files = result.unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].name, "test1.md");
    assert_eq!(files[1].name, "test2.md");
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
    let ctx = MockOpenWebUIServiceMock::process_file_context();
    ctx.expect()
        .with(eq("Test file content".to_string()))
        .times(1)
        .returning(|_| Ok(ProcessedFile { content: "Processed content".to_string() }));

    let file = "Test file content".to_string();
    let result = FileService::send_to_openwebui::<MockOpenWebUIServiceMock>(file).await;
    assert!(result.is_ok());
    let processed_file = result.unwrap();
    assert_eq!(processed_file.content, "Processed content");
}

#[test]
fn test_save_file_metadata() {
    let metadata = Metadata {
        file_name: "test.md".to_string(),
        last_modified: chrono::Utc::now(),
        processed_file: "Processed content".to_string(),
        original_file: "Original content".to_string(),
    };
    let result = FileService::save_file_metadata(metadata);
    assert!(result.is_ok());
}