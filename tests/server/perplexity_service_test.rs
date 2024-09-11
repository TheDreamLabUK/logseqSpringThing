use webxr_graph::services::perplexity_service::*;
use mockall::predicate::*;
use mockall::mock;
use std::fs;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::sync::Semaphore;
use tokio::time::timeout;
use lazy_static::lazy_static;

const TEST_MAX_CONCURRENT_REQUESTS: usize = 5;

lazy_static! {
    static ref TEST_SEMAPHORE: Semaphore = Semaphore::new(TEST_MAX_CONCURRENT_REQUESTS);
}

mock! {
    pub Client {}
    impl Clone for Client {
        fn clone(&self) -> Self;
    }
    #[async_trait::async_trait]
    impl Client {
        async fn post(&self, url: &str) -> reqwest::RequestBuilder;
    }
}

#[tokio::test]
async fn test_clean_logseq_links() {
    let test_cases = vec![
        ("This is a [[test]] with [[multiple]] [[links]]", "This is a test with multiple links"),
        ("No links here", "No links here"),
        ("[[Empty]] [[]] [[links]]", "Empty  links"),
        ("", ""),
    ];

    for (input, expected) in test_cases {
        assert_eq!(clean_logseq_links(input), expected);
    }
}

#[tokio::test]
async fn test_process_markdown_block() {
    let input = "- This is a [[test]] block";
    let prompt = "Summarize";
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];
    let api_response = "Processed content";
    
    let expected = "- ```\nThis is a test block\n```\nProcessed content";
    assert_eq!(process_markdown_block(input, prompt, &topics, api_response), expected);
}

#[tokio::test]
async fn test_select_context_blocks() {
    let content = "Block 1\nBlock 2\nBlock 3";
    let active_block = "Block 2";
    let result = select_context_blocks(content, active_block);
    assert_eq!(result, vec!["Block 2".to_string()]);
}

#[tokio::test]
async fn test_call_perplexity_api_success() {
    let _permit = TEST_SEMAPHORE.acquire().await.unwrap();
    let mut mock_client = MockClient::new();
    mock_client.expect_post()
        .with(eq("https://api.perplexity.ai/chat/completions"))
        .times(1)
        .returning(|_| {
            let mut req = reqwest::Request::new(reqwest::Method::POST, "https://api.perplexity.ai/chat/completions".parse().unwrap());
            req.headers_mut().insert("Authorization", "Bearer test_key".parse().unwrap());
            reqwest::RequestBuilder::from_parts(mock_client, req)
        });

    std::env::set_var("PERPLEXITY_API_KEY", "test_key");

    let prompt = "Test prompt";
    let context = vec!["Context 1".to_string(), "Context 2".to_string()];
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];

    let result = call_perplexity_api(prompt, &context, &topics).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_call_perplexity_api_error() {
    let _permit = TEST_SEMAPHORE.acquire().await.unwrap();
    let mut mock_client = MockClient::new();
    mock_client.expect_post()
        .with(eq("https://api.perplexity.ai/chat/completions"))
        .times(3) // Use the actual MAX_RETRIES value from perplexity_service.rs
        .returning(|_| {
            let mut req = reqwest::Request::new(reqwest::Method::POST, "https://api.perplexity.ai/chat/completions".parse().unwrap());
            req.headers_mut().insert("Authorization", "Bearer test_key".parse().unwrap());
            reqwest::RequestBuilder::from_parts(mock_client, req)
        });

    std::env::set_var("PERPLEXITY_API_KEY", "test_key");

    let prompt = "Test prompt";
    let context = vec!["Context 1".to_string(), "Context 2".to_string()];
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];

    let result = call_perplexity_api(prompt, &context, &topics).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_call_perplexity_api_timeout() {
    let _permit = TEST_SEMAPHORE.acquire().await.unwrap();
    let mut mock_client = MockClient::new();
    mock_client.expect_post()
        .with(eq("https://api.perplexity.ai/chat/completions"))
        .times(1)
        .returning(|_| {
            let mut req = reqwest::Request::new(reqwest::Method::POST, "https://api.perplexity.ai/chat/completions".parse().unwrap());
            req.headers_mut().insert("Authorization", "Bearer test_key".parse().unwrap());
            reqwest::RequestBuilder::from_parts(mock_client, req)
        });

    std::env::set_var("PERPLEXITY_API_KEY", "test_key");

    let prompt = "Test prompt";
    let context = vec!["Context 1".to_string(), "Context 2".to_string()];
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];

    let result = timeout(Duration::from_millis(100), call_perplexity_api(prompt, &context, &topics)).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_process_markdown_parallel() {
    let _permit = TEST_SEMAPHORE.acquire().await.unwrap();
    let file_content = "- Block 1\n- Block 2\n- Block 3\n- Block 4\n- Block 5";
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), file_content).unwrap();

    let prompt = "Test prompt";
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];

    let mut mock_client = MockClient::new();
    mock_client.expect_post()
        .times(5)
        .returning(|_| {
            let mut req = reqwest::Request::new(reqwest::Method::POST, "https://api.perplexity.ai/chat/completions".parse().unwrap());
            req.headers_mut().insert("Authorization", "Bearer test_key".parse().unwrap());
            reqwest::RequestBuilder::from_parts(mock_client, req)
        });

    std::env::set_var("PERPLEXITY_API_KEY", "test_key");

    let start = std::time::Instant::now();
    let result = process_markdown(temp_file.path().to_str().unwrap(), prompt, &topics).await;
    let duration = start.elapsed();

    assert!(result.is_ok());
    assert!(duration < Duration::from_secs(2), "Parallel processing should be faster than sequential");
    
    let processed_content = result.unwrap();
    for i in 1..=5 {
        assert!(processed_content.contains(&format!("Block {}", i)));
    }
}

#[tokio::test]
async fn test_process_markdown_complex() {
    let _permit = TEST_SEMAPHORE.acquire().await.unwrap();
    let file_content = "- Block 1 with [[link]]\n- Block 2 with *emphasis*\n- [[Block 3]] with **strong**";
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), file_content).unwrap();

    let prompt = "Test prompt";
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];

    let mut mock_client = MockClient::new();
    mock_client.expect_post()
        .times(3)
        .returning(|_| {
            let mut req = reqwest::Request::new(reqwest::Method::POST, "https://api.perplexity.ai/chat/completions".parse().unwrap());
            req.headers_mut().insert("Authorization", "Bearer test_key".parse().unwrap());
            reqwest::RequestBuilder::from_parts(mock_client, req)
        });

    std::env::set_var("PERPLEXITY_API_KEY", "test_key");

    let result = process_markdown(temp_file.path().to_str().unwrap(), prompt, &topics).await;
    assert!(result.is_ok());
    
    let processed_content = result.unwrap();
    assert!(processed_content.contains("Block 1 with link"));
    assert!(processed_content.contains("Block 2 with *emphasis*"));
    assert!(processed_content.contains("Block 3 with **strong**"));
}

#[tokio::test]
async fn test_process_markdown_file_not_found() {
    let prompt = "Test prompt";
    let topics = vec!["Topic1".to_string(), "Topic2".to_string()];

    let result = process_markdown("non_existent_file.md", prompt, &topics).await;
    assert!(result.is_err());
}

#[test]
fn test_load_prompt() {
    let prompt_content = "This is a test prompt";
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), prompt_content).unwrap();

    let result = load_prompt(temp_file.path().to_str().unwrap());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), prompt_content);
}

#[test]
fn test_load_prompt_file_not_found() {
    let result = load_prompt("non_existent_file.txt");
    assert!(result.is_err());
}

#[test]
fn test_load_topics() {
    let topics_content = "Topic1, Topic2, Topic3";
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), topics_content).unwrap();

    let result = load_topics(temp_file.path().to_str().unwrap());
    assert!(result.is_ok());
    let topics = result.unwrap();
    assert_eq!(topics, vec!["Topic1", "Topic2", "Topic3"]);
}

#[test]
fn test_load_topics_empty_file() {
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), "").unwrap();

    let result = load_topics(temp_file.path().to_str().unwrap());
    assert!(result.is_ok());
    let topics = result.unwrap();
    assert!(topics.is_empty());
}

#[test]
fn test_load_topics_file_not_found() {
    let result = load_topics("non_existent_file.txt");
    assert!(result.is_err());
}

#[test]
fn test_perplexity_request_serialization() {
    let request = PerplexityRequest {
        model: "test-model".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "System message".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "User message".to_string(),
            },
        ],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        return_citations: Some(true),
        search_domain_filter: Some(vec!["domain1".to_string(), "domain2".to_string()]),
        return_images: Some(false),
        return_related_questions: Some(true),
        search_recency_filter: Some("day".to_string()),
        top_k: Some(5),
        stream: Some(false),
        presence_penalty: Some(0.5),
        frequency_penalty: Some(0.5),
    };

    let serialized = serde_json::to_string(&request).unwrap();
    let deserialized: PerplexityRequest = serde_json::from_str(&serialized).unwrap();

    assert_eq!(request.model, deserialized.model);
    assert_eq!(request.messages.len(), deserialized.messages.len());
    assert_eq!(request.max_tokens, deserialized.max_tokens);
    assert_eq!(request.temperature, deserialized.temperature);
    assert_eq!(request.top_p, deserialized.top_p);
    assert_eq!(request.return_citations, deserialized.return_citations);
    assert_eq!(request.search_domain_filter, deserialized.search_domain_filter);
    assert_eq!(request.return_images, deserialized.return_images);
    assert_eq!(request.return_related_questions, deserialized.return_related_questions);
    assert_eq!(request.search_recency_filter, deserialized.search_recency_filter);
    assert_eq!(request.top_k, deserialized.top_k);
    assert_eq!(request.stream, deserialized.stream);
    assert_eq!(request.presence_penalty, deserialized.presence_penalty);
    assert_eq!(request.frequency_penalty, deserialized.frequency_penalty);
}

#[test]
fn test_perplexity_response_deserialization() {
    let json_response = serde_json::json!({
        "id": "response_id",
        "model": "test-model",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "delta": {
                    "content": "Test delta"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    });

    let response: PerplexityResponse = serde_json::from_value(json_response).unwrap();

    assert_eq!(response.id, Some("response_id".to_string()));
    assert_eq!(response.model, Some("test-model".to_string()));
    assert_eq!(response.object, Some("chat.completion".to_string()));
    assert_eq!(response.created, Some(1234567890));
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    assert_eq!(response.choices[0].message.role, "assistant");
    assert_eq!(response.choices[0].message.content, "Test response");
    assert_eq!(response.choices[0].delta, Some(Delta { content: Some("Test delta".to_string()) }));
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
}