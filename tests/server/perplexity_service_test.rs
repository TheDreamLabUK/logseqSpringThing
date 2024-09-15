use std::env;
use wiremock::{MockServer Mock ResponseTemplate};
use wiremock::matchers::{method path header};
#[tokio::test]
async fn test_clean_logseq_links() {
    let test_cases = vec![
        ("[[Link]]", "Link"),
        ("This is a [[nested link]]", "This is a nested link"),
        ("Multiple [[links]] in [[one]] string", "Multiple links in one string"),
        ("No links here", "No links here"),
        ("[[]]", ""),
    ];

    for (input, expected) in test_cases {
        assert_eq!(clean_logseq_links(input), expected);
    }
}

#[tokio::test]
async fn test_process_markdown_block() {
    let input = "This is a test block";
    let prompt = "Summarize the following:";
    let topics = vec!["topic1".to_string(), "topic2".to_string()];
    let api_response = "Summarized content";
    let expected = format!("- ```\nThis is a test block```\nPrompt: {}\nTopics: {}\nResponse: {}", prompt, topics.join(", "), api_response);

    assert_eq!(process_markdown_block(input, prompt, &topics, api_response), expected);
}

#[tokio::test]
async fn test_select_context_blocks() {
    let content = "Block 1\n\nBlock 2\n\nBlock 3";
    let active_block = "Block 2";

    let result = select_context_blocks(content, active_block);
    assert_eq!(result, vec!["Block 2".to_string()]);
}

#[tokio::test]
async fn test_call_perplexity_api_success() {
    let _m = mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"choices":[{"message":{"content":"API Response"}}]}"#)
        .create();

    env::set_var("PERPLEXITY_API_URL", &mockito::server_url());
    env::set_var("PERPLEXITY_API_KEY", "test_api_key");

    let prompt = "Test prompt";
    let context = vec!["Test context".to_string()];
    let topics = vec!["Test topic".to_string()];

    let result = call_perplexity_api(prompt, &context, &topics).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "API Response");
}

#[tokio::test]
async fn test_call_perplexity_api_error() {
    let _m = mock("POST", "/chat/completions")
        .with_status(500)
        .create();

    env::set_var("PERPLEXITY_API_URL", &mockito::server_url());
    env::set_var("PERPLEXITY_API_KEY", "test_api_key");

    let prompt = "Test prompt";
    let context = vec!["Test context".to_string()];
    let topics = vec!["Test topic".to_string()];

    let result = call_perplexity_api(prompt, &context, &topics).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_call_perplexity_api_timeout() {
    let _m = mock("POST", "/chat/completions")
        .expect(1)
        .with_header("content-type", "application/json")
        .with_body(r#"{"choices":[{"message":{"content":"Delayed Response"}}]}"#)
        // .delay(Duration::from_secs(3))
        .create();

    env::set_var("PERPLEXITY_API_URL", &mockito::server_url());
    env::set_var("PERPLEXITY_API_KEY", "test_api_key");

    let prompt = "Test prompt";
    let context = vec!["Test context".to_string()];
    let topics = vec!["Test topic".to_string()];

    let result = timeout(Duration::from_secs(2), call_perplexity_api(prompt, &context, &topics)).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_process_markdown_success() {
    let _m = mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"choices":[{"message":{"content":"Processed block"}}]}"#)
        .create();

    env::set_var("PERPLEXITY_API_URL", &mockito::server_url());
    env::set_var("PERPLEXITY_API_KEY", "test_api_key");

    let file_content = "Test content\n\nAnother block";

    let result = process_markdown(&file_content).await;
    assert!(result.is_ok());
    let processed_content = result.unwrap();
    assert!(processed_content.contains("Test content"));
    assert!(processed_content.contains("Another block"));
    assert!(processed_content.contains("Processed block"));
}

#[tokio::test]
async fn test_process_markdown_api_error() {
    let _m = mock("POST", "/chat/completions")
        .with_status(500)
        .create();

    env::set_var("PERPLEXITY_API_URL", &mockito::server_url());
    env::set_var("PERPLEXITY_API_KEY", "test_api_key");

    let file_content = "Test content";

    let result = process_markdown(&file_content).await;
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
        return_citations: Some(false),
        stream: Some(false),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
    };

    let serialized = serde_json::to_string(&request).unwrap();
    let deserialized: PerplexityRequest = serde_json::from_str(&serialized).unwrap();

    assert_eq!(request.model, deserialized.model);
    assert_eq!(request.messages.len(), deserialized.messages.len());
    assert_eq!(request.max_tokens, deserialized.max_tokens);
    assert_eq!(request.temperature, deserialized.temperature);
    assert_eq!(request.top_p, deserialized.top_p);
    assert_eq!(request.return_citations, deserialized.return_citations);
    assert_eq!(request.stream, deserialized.stream);
    assert_eq!(request.presence_penalty, deserialized.presence_penalty);
    assert_eq!(request.frequency_penalty, deserialized.frequency_penalty);
}

#[test]
fn test_perplexity_response_deserialization() {
    let json_response = json!({
        "id": "test-id",
        "model": "test-model",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Test response content"
                },
                "delta": {
                    "content": "Delta content"
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

    assert_eq!(response.id, Some("test-id".to_string()));
    assert_eq!(response.model, Some("test-model".to_string()));
    assert_eq!(response.object, Some("chat.completion".to_string()));
    assert_eq!(response.created, Some(1234567890));
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    assert_eq!(response.choices[0].message.role, "assistant");
    assert_eq!(response.choices[0].message.content, "Test response content");
    assert_eq!(response.choices[0].delta.as_ref().and_then(|d| d.content.as_ref()), Some(&"Delta content".to_string()));
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
}
