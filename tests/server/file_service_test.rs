use webxr_graph::services::file_service::*;
use webxr_graph::models::metadata::Metadata;
use webxr_graph::services::perplexity_service::{PerplexityService, PerplexityError, ApiClient};
use webxr_graph::{GithubFile, ProcessedFile, PerplexityRequest};
use async_trait::async_trait;
use mockall::predicate::*;
use std::sync::Arc;
use webxr_graph::config::{Settings, PerplexityConfig};

mockall::mock! {
    pub GitHubService {}

    #[async_trait]
    impl webxr_graph::services::file_service::GitHubService for GitHubService {
        async fn fetch_files() -> Result<Vec<GithubFile>, reqwest::Error>;
    }
}

struct MockPerplexityService;

#[async_trait]
impl PerplexityService for MockPerplexityService {
    async fn process_file(file_content: String, _settings: &Settings, _api_client: &dyn ApiClient) -> Result<ProcessedFile, PerplexityError> {
        Ok(ProcessedFile { content: format!("Processed: {}", file_content) })
    }
}

#[async_trait]
impl ApiClient for MockPerplexityService {
    async fn post_json(&self, _url: &str, _body: &PerplexityRequest, _api_key: &str) -> Result<String, PerplexityError> {
        Ok(String::from("mock"))
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
    let file_content = "test content".to_string();
    let settings = Settings {
        prompt: "Test Prompt".to_string(),
        topics: vec!["topic1".to_string()],
        perplexity: PerplexityConfig {
            api_key: "test_key".to_string(),
            model: "test_model".to_string(),
            api_base_url: "http://localhost".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            presence_penalty: 0.5,
            frequency_penalty: 0.5,
        },
    };
    let api_client = Arc::new(MockPerplexityService);

    let result = FileService::process_with_perplexity::<MockPerplexityService>(file_content.clone(), &settings, &*api_client).await.unwrap();
    assert_eq!(result.content, "Processed: test content");
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
