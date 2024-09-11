use std::fs;
use std::io::{self, Write};
use regex::Regex;
use serde::{Serialize, Deserialize};
use reqwest::Client;
use tokio::time::{sleep, Duration};
use tokio::sync::Semaphore;
use log::error;
use thiserror::Error;
use futures::stream::{self, StreamExt};
use lazy_static::lazy_static;

const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: u64 = 5;
const MAX_CONCURRENT_REQUESTS: usize = 5;

lazy_static! {
    static ref API_CLIENT: Client = Client::new();
    static ref REQUEST_SEMAPHORE: Semaphore = Semaphore::new(MAX_CONCURRENT_REQUESTS);
}

#[derive(Error, Debug)]
pub enum PerplexityError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("API error: {0}")]
    Api(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub async fn process_markdown(file_path: &str, prompt: &str, topics: &[String]) -> Result<String, PerplexityError> {
    let content = fs::read_to_string(file_path).map_err(PerplexityError::Io)?;
    let blocks: Vec<&str> = content.split("\n- ").filter(|s| !s.trim().is_empty()).collect();
    
    let results = stream::iter(blocks)
        .map(|block| {
            let prompt = prompt.to_string();
            let topics = topics.to_vec();
            let content = content.clone();
            async move {
                let trimmed_block = block.trim();
                let context = select_context_blocks(&content, trimmed_block);
                let api_response = call_perplexity_api(&prompt, &context, &topics).await?;
                let processed_block = process_markdown_block(trimmed_block, &prompt, &topics, &api_response);
                Ok::<String, PerplexityError>(processed_block)
            }
        })
        .buffer_unordered(MAX_CONCURRENT_REQUESTS)
        .collect::<Vec<Result<String, PerplexityError>>>()
        .await;

    let processed_content = results.into_iter()
        .collect::<Result<Vec<String>, PerplexityError>>()?
        .join("\n");

    Ok(processed_content)
}

pub async fn call_perplexity_api(prompt: &str, context: &[String], topics: &[String]) -> Result<String, PerplexityError> {
    let _permit = REQUEST_SEMAPHORE.acquire().await.unwrap();
    let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|e| PerplexityError::Api(e.to_string()))?;

    let system_message = format!(
        "You are an AI assistant building summaries of web links and text. You will visit any web links found in the text and integrate \
        a summary with web citations, aiming for up to two citations explicitly returned in context as raw web hyperlinks. \
        Ensure to return web links as citations separated by new lines. \
        You should aim to select one or more of these topics in this form appropriate to the created summary, \
        embedding the topic in Logseq double square brackets once in the returned text. \
        Relevant category topics are: {}.",
        topics.join(", ")
    );
    
    let request = PerplexityRequest {
        model: "llama-3.1-sonar-small-128k-online".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: system_message,
            },
            Message {
                role: "user".to_string(),
                content: format!("Prompt: {}\n\nContext:\n{}", prompt, context.join("\n")),
            },
        ],
        max_tokens: Some(4096),
        temperature: Some(0.5),
        top_p: Some(0.9),
        return_citations: Some(false),
        search_domain_filter: Some(vec!["all".to_string()]),
        return_images: Some(false),
        return_related_questions: Some(false),
        search_recency_filter: Some("year".to_string()),
        top_k: Some(0),
        stream: Some(false),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(1.0),
    };

    for attempt in 1..=MAX_RETRIES {
        let response = API_CLIENT
            .post("https://api.perplexity.ai/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request)
            .send()
            .await
            .map_err(PerplexityError::Reqwest)?;

        if response.status().is_success() {
            let response_text = response.text().await.map_err(PerplexityError::Reqwest)?;
            return match serde_json::from_str::<PerplexityResponse>(&response_text) {
                Ok(parsed_response) => {
                    if let Some(message) = parsed_response.choices.first().map(|choice| &choice.message) {
                        Ok(message.content.clone())
                    } else {
                        Err(PerplexityError::Api("No content in API response".to_string()))
                    }
                },
                Err(e) => {
                    error!("Failed to parse API response: {}", e);
                    error!("Raw response: {}", response_text);
                    Err(PerplexityError::Serialization(e))
                }
            };
        } else if response.status().as_u16() == 524 || response.status().is_server_error() {
            error!("API request failed with status {}, attempt {} of {}", response.status(), attempt, MAX_RETRIES);
            if attempt < MAX_RETRIES {
                sleep(Duration::from_secs(RETRY_DELAY)).await;
            }
        } else {
            let error_message = format!("API request failed with status {}", response.status());
            return Err(PerplexityError::Api(error_message));
        }
    }

    Err(PerplexityError::Api("Max retries reached, API request failed".to_string()))
}

pub fn select_context_blocks(_content: &str, active_block: &str) -> Vec<String> {
    vec![active_block.to_string()] // Only return the current block
}

pub fn clean_logseq_links(input: &str) -> String {
    let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
    re.replace_all(input, "$1").to_string()
}

pub fn process_markdown_block(input: &str, _prompt: &str, _topics: &[String], api_response: &str) -> String {
    let cleaned_input = clean_logseq_links(input);
    let mut output = Vec::new();
    writeln!(output, "- ```").unwrap();
    writeln!(output, "{}", cleaned_input.trim_start_matches("- ").trim_end()).unwrap();
    writeln!(output, "```").unwrap();
    write!(output, "{}", api_response).unwrap();
    String::from_utf8(output).unwrap()
}

pub fn load_prompt(file_path: &str) -> io::Result<String> {
    fs::read_to_string(file_path)
}

pub fn load_topics(file_path: &str) -> io::Result<Vec<String>> {
    let content = fs::read_to_string(file_path)?;
    Ok(content.split(',').map(|s| s.trim().to_string()).collect())
}

#[derive(Debug, Serialize)]
pub struct PerplexityRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub return_citations: Option<bool>,
    pub search_domain_filter: Option<Vec<String>>,
    pub return_images: Option<bool>,
    pub return_related_questions: Option<bool>,
    pub search_recency_filter: Option<String>,
    pub top_k: Option<u32>,
    pub stream: Option<bool>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct PerplexityResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub finish_reason: Option<String>,
    pub message: Message,
    pub delta: Option<Delta>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}