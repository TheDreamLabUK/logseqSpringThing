use std::io;
use regex::Regex;
use serde::{Serialize, Deserialize};
use reqwest::Client;
use tokio::time::{sleep, Duration};
use tokio::sync::Semaphore;
use log::error;
use thiserror::Error;
use futures::stream::{self, StreamExt};
use lazy_static::lazy_static;
use std::env;
use pulldown_cmark::{Parser, Event, Tag};
use async_trait::async_trait;
use config::ConfigError;

use crate::config::Settings;
use crate::services::file_service::ProcessedFile;

#[derive(Error, Debug)]
pub enum PerplexityError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("HTTP request error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("API error: {0}")]
    Api(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Environment variable error: {0}")]
    EnvVar(#[from] env::VarError),
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
}

lazy_static! {
    static ref API_CLIENT: Client = Client::builder()
        .timeout(Duration::from_secs(
            env::var("API_CLIENT_TIMEOUT")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .expect("API_CLIENT_TIMEOUT must be a valid u64")
        ))
        .build()
        .expect("Failed to build API client");

    static ref REQUEST_SEMAPHORE: Semaphore = Semaphore::new(
        env::var("MAX_CONCURRENT_REQUESTS")
            .unwrap_or_else(|_| "5".to_string())
            .parse::<usize>()
            .expect("MAX_CONCURRENT_REQUESTS must be a valid usize")
    );
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerplexityRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub return_citations: Option<bool>,
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

#[async_trait]
pub trait ApiClient {
    async fn post_json(
        &self,
        url: &str,
        body: &PerplexityRequest,
        api_key: &str,
    ) -> Result<String, PerplexityError>;
}

pub struct RealApiClient {
    client: Client,
}

impl RealApiClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

#[async_trait]
impl ApiClient for RealApiClient {
    async fn post_json(
        &self,
        url: &str,
        body: &PerplexityRequest,
        api_key: &str,
    ) -> Result<String, PerplexityError> {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(body)
            .send()
            .await?
            .text()
            .await?;
        Ok(response)
    }
}

pub async fn process_markdown(file_content: &str, settings: &Settings, api_client: &dyn ApiClient) -> Result<String, PerplexityError> {
    let blocks = split_markdown_blocks(file_content);

    let results = stream::iter(blocks.into_iter())
        .map(|block| {
            let prompt = settings.prompt.clone();
            let topics = settings.topics.clone();
            let content = file_content.to_string();
            async move {
                let trimmed_block = block.trim().to_string();
                let context = select_context_blocks(&content, &trimmed_block);

                let api_response = call_perplexity_api(&prompt, &context, &topics, api_client, &settings.perplexity).await?;
                let processed_block = process_markdown_block(&trimmed_block, &prompt, &topics, &api_response);
                Ok::<String, PerplexityError>(processed_block)
            }
        })
        .buffer_unordered(
            env::var("MAX_CONCURRENT_REQUESTS")
                .unwrap_or_else(|_| "5".to_string())
                .parse::<usize>()
                .unwrap_or(5)
        )
        .collect::<Vec<Result<String, PerplexityError>>>()
        .await;

    let processed_content = results.into_iter()
        .collect::<Result<Vec<String>, PerplexityError>>()?
        .join("\n");

    Ok(processed_content)
}

pub async fn call_perplexity_api(
    prompt: &str,
    context: &[String],
    topics: &[String],
    api_client: &dyn ApiClient,
    perplexity_config: &crate::config::PerplexityConfig,
) -> Result<String, PerplexityError> {
    let _permit = REQUEST_SEMAPHORE.acquire().await.unwrap();

    let max_retries: u32 = env::var("MAX_RETRIES").unwrap_or_else(|_| "3".to_string()).parse().unwrap_or(3);
    let retry_delay: u64 = env::var("RETRY_DELAY").unwrap_or_else(|_| "5".to_string()).parse().unwrap_or(5);

    let system_message = format!(
        "{}\nRelevant category topics are: {}.",
        prompt.trim(),
        topics.join(", ")
    );

    let request = PerplexityRequest {
        model: perplexity_config.model.clone(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: system_message,
            },
            Message {
                role: "user".to_string(),
                content: format!(
                    "Context:\n{}",
                    context.join("\n")
                ),
            },
        ],
        max_tokens: Some(perplexity_config.max_tokens),
        temperature: Some(perplexity_config.temperature),
        top_p: Some(perplexity_config.top_p),
        return_citations: Some(false),
        stream: Some(false),
        presence_penalty: Some(perplexity_config.presence_penalty),
        frequency_penalty: Some(perplexity_config.frequency_penalty),
    };

    for attempt in 1..=max_retries {
        match api_client.post_json(&perplexity_config.api_base_url, &request, &perplexity_config.api_key).await {
            Ok(response_text) => {
                return parse_perplexity_response(&response_text);
            }
            Err(e) => {
                error!("API request encountered an error: {} on attempt {} of {}", e, attempt, max_retries);
                if attempt < max_retries {
                    sleep(Duration::from_secs(retry_delay)).await;
                    continue;
                } else {
                    return Err(e);
                }
            }
        }
    }

    Err(PerplexityError::Api("Max retries reached, API request failed".to_string()))
}

fn parse_perplexity_response(response_text: &str) -> Result<String, PerplexityError> {
    match serde_json::from_str::<PerplexityResponse>(response_text) {
        Ok(parsed_response) => {
            if let Some(message) = parsed_response.choices.first().map(|choice| &choice.message) {
                Ok(message.content.clone())
            } else {
                Err(PerplexityError::Api("No content in API response".to_string()))
            }
        }
        Err(e) => {
            error!("Failed to parse API response: {}", e);
            error!("Raw response: {}", response_text);
            Err(PerplexityError::Serialization(e))
        }
    }
}

fn split_markdown_blocks(content: &str) -> Vec<String> {
    let parser = Parser::new(content);
    let mut blocks = Vec::new();
    let mut current_block = String::new();

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading(_, _, _) | Tag::Item => {
                    if !current_block.is_empty() {
                        blocks.push(current_block.clone());
                        current_block.clear();
                    }
                },
                _ => {},
            },
            Event::Text(text) => {
                current_block.push_str(&text);
            },
            Event::End(tag) => match tag {
                Tag::Paragraph | Tag::Heading(_, _, _) | Tag::Item => {
                    if !current_block.is_empty() {
                        blocks.push(current_block.clone());
                        current_block.clear();
                    }
                },
                _ => {},
            },
            _ => {},
        }
    }

    if !current_block.is_empty() {
        blocks.push(current_block);
    }

    blocks
}

pub fn select_context_blocks(_content: &str, active_block: &str) -> Vec<String> {
    vec![active_block.to_string()]
}

pub fn clean_logseq_links(input: &str) -> String {
    let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
    re.replace_all(input, "$1").to_string()
}

pub fn process_markdown_block(input: &str, prompt: &str, topics: &[String], api_response: &str) -> String {
    let cleaned_input = clean_logseq_links(input);

    format!(
        "- ```\n{}```\nPrompt: {}\nTopics: {}\nResponse: {}",
        cleaned_input.trim_start_matches("- ").trim_end(),
        prompt,
        topics.join(", "),
        api_response
    )
}

#[async_trait]
pub trait PerplexityService {
    async fn process_file(file_content: String, settings: &Settings, api_client: &dyn ApiClient) -> Result<ProcessedFile, PerplexityError>;
}

pub struct RealPerplexityService;

#[async_trait]
impl PerplexityService for RealPerplexityService {
    async fn process_file(file_content: String, settings: &Settings, api_client: &dyn ApiClient) -> Result<ProcessedFile, PerplexityError> {
        let processed_content = process_markdown(&file_content, settings, api_client).await?;
        Ok(ProcessedFile { content: processed_content })
    }
}