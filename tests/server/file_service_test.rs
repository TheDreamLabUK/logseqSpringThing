use webxr_graph::services::file_service::*;
use webxr_graph::models::metadata::Metadata;
use webxr_graph::services::perplexity_service::PerplexityService;
use webxr_graph::services::perplexity_service::PerplexityError;
use async_trait::async_trait;
use mockall::predicate::*;

mockall::mock! {
    pub GitHubService {}

    #[async_trait]
    impl GitHubService for GitHubService {
        async fn fetch_files() -> Result<Vec<GithubFile>, reqwest::Error>;
    }
}

mockall::mock! {
    pub PerplexityService {}

    #[async_trait]
    impl PerplexityService for PerplexityService {
        async fn process_file(file_content: String) -> Result<ProcessedFile, PerplexityError>;
    }
}

#[tokio::test]
async fn test_fetch_files_from_github() {
    let ctx = MockGitHubService::fetch_files_context();
    ctx.expect()
        .times(1)
        .returning(|| Ok(vec![
            GithubFile {
                name: "file1.md".to_string(),
                content: "content1".to_string(),
            },
            GithubFile {
                name: "file2.md".to_string(),
                content: "content2".to_string(),
            },
        ]));

    let files = FileService::fetch_files_from_github::<MockGitHubService>().await.unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].name, "file1.md");
    assert_eq!(files[0].content, "content1");
    assert_eq!(files[1].name, "file2.md");
    assert_eq!(files[1].content, "content2");
}

#[test]
fn test_compare_and_identify_updates() {
    let github_files = vec![
        GithubFile {
            name: "file1.md".to_string(),
            content: "content1".to_string(),
        },
        GithubFile {
            name: "file2.md".to_string(),
            content: "content2".to_string(),
        },
    ];

    let updated_files = FileService::compare_and_identify_updates(github_files).unwrap();
    assert_eq!(updated_files, vec!["file1.md", "file2.md"]);
}

#[tokio::test]
async fn test_process_with_perplexity() {
    let ctx = MockPerplexityService::process_file_context();
    let file_content = "test content".to_string();
    ctx.expect()
        .with(eq(file_content.clone()))
        .times(1)
        .returning(|_| Ok(ProcessedFile { content: "processed content".to_string() }));

    let result = FileService::process_with_perplexity::<MockPerplexityService>(file_content.clone()).await.unwrap();
    assert_eq!(result.content, "processed content");
}

#[test]
fn test_save_file_metadata() {
    let metadata = Metadata {
        file_name: "test.md".to_string(),
        last_modified: chrono::Utc::now(),
        processed_file: "processed content".to_string(),
        original_file: "original content".to_string(),
    };

    let result = FileService::save_file_metadata(metadata);
    assert!(result.is_ok());
}
