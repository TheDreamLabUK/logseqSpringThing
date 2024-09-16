use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};
use tokio::time::timeout;
use std::time::Duration;
use serde_json::json;
use async_trait::async_trait;

use webxr_graph::services::perplexity_service::{
    call_perplexity_api,
    process_markdown,
    ApiClient,
    PerplexityRequest,
    PerplexityError,
    RealApiClient,
};
use webxr_graph::config::{Settings, PerplexityConfig};

struct MockApiClient {
    mock_server: MockServer,
}

impl MockApiClient {
    async fn new() -> Self {
        Self {
            mock_server: MockServer::start().await,
        }
    }

    fn uri(&self) -> String {
        self.mock_server.uri()
    }

    async fn mock(&self, status: u16, body: serde_json::Value) {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&self.mock_server)
            .await;
    }
}

#[async_trait]
impl ApiClient for MockApiClient {
    async fn post_json(
        &self,
        url: &str,
        body: &PerplexityRequest,
        _api_key: &str,
    ) -> Result<String, PerplexityError> {
        let client = reqwest::Client::new();
        let response = client
            .post(url)
            .json(body)
            .send()
            .await?
            .text()
            .await?;
        Ok(response)
    }
}

fn setup_mock_settings(mock_server_uri: &str) -> Settings {
    Settings {
        prompt: "Test prompt".to_string(),
        topics: vec!["topic1".to_string(), "topic2".to_string()],
        perplexity: PerplexityConfig {
            api_key: "test_api_key".to_string(),
            model: "test_model".to_string(),
            api_base_url: format!("{}/chat/completions", mock_server_uri),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            presence_penalty: 0.6,
            frequency_penalty: 0.6,
        },
    }
}

#[tokio::test]
async fn test_call_perplexity_api_success() {
    let mock_client = MockApiClient::new().await;
    mock_client.mock(200, json!({"choices":[{"message":{"content":"API Response"}}]})).await;

    let settings = setup_mock_settings(&mock_client.uri());

    let prompt = &settings.prompt;
    let context = vec!["Test context".to_string()];
    let topics = settings.topics.clone();

    let result = call_perplexity_api(prompt, &context, &topics, &mock_client, &settings.perplexity).await;

    match result {
        Ok(response) => {
            assert_eq!(response, "API Response");
        },
        Err(e) => {
            panic!("call_perplexity_api failed with error: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_call_perplexity_api_error() {
    let mock_client = MockApiClient::new().await;
    mock_client.mock(500, json!({"error": "Internal Server Error"})).await;

    let settings = setup_mock_settings(&mock_client.uri());

    let prompt = &settings.prompt;
    let context = vec!["Test context".to_string()];
    let topics = settings.topics.clone();

    let result = call_perplexity_api(prompt, &context, &topics, &mock_client, &settings.perplexity).await;

    assert!(result.is_err(), "Expected an error due to 500 response, but call succeeded");
}

#[tokio::test]
async fn test_call_perplexity_api_timeout() {
    let mock_client = MockApiClient::new().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200)
            .set_delay(Duration::from_secs(3))
            .set_body_json(json!({"choices":[{"message":{"content":"Delayed Response"}}]})))
        .mount(&mock_client.mock_server)
        .await;

    let settings = setup_mock_settings(&mock_client.uri());

    let prompt = &settings.prompt;
    let context = vec!["Test context".to_string()];
    let topics = settings.topics.clone();

    let result = timeout(
        Duration::from_secs(2),
        call_perplexity_api(prompt, &context, &topics, &mock_client, &settings.perplexity)
    ).await;

    assert!(result.is_err(), "Expected timeout error, but call succeeded");
}

#[tokio::test]
async fn test_process_markdown_success() {
    let mock_client = MockApiClient::new().await;
    mock_client.mock(200, json!({"choices":[{"message":{"content":"Processed block"}}]})).await;

    let settings = setup_mock_settings(&mock_client.uri());

    let file_content = "Test content\n\nAnother block";

    let result = process_markdown(file_content, &settings, &mock_client).await;

    match result {
        Ok(processed_content) => {
            assert!(processed_content.contains("Test content"), "Processed content does not contain 'Test content'");
            assert!(processed_content.contains("Another block"), "Processed content does not contain 'Another block'");
            assert!(processed_content.contains("Processed block"), "Processed content does not contain 'Processed block'");
        },
        Err(e) => {
            panic!("process_markdown failed with error: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_process_markdown_api_error() {
    let mock_client = MockApiClient::new().await;
    mock_client.mock(500, json!({"error": "Internal Server Error"})).await;

    let settings = setup_mock_settings(&mock_client.uri());

    let file_content = "Test content";

    let result = process_markdown(file_content, &settings, &mock_client).await;

    assert!(result.is_err(), "Expected an error due to 500 response, but process_markdown succeeded");
}