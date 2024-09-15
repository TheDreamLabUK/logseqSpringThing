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

/// Custom error type for Perplexity operations
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
    /// HTTP client used for making API requests
    static ref API_CLIENT: Client = Client::builder()
        .timeout(Duration::from_secs(
            env::var("API_CLIENT_TIMEOUT")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .expect("API_CLIENT_TIMEOUT must be a valid u64")
        ))
        .build()
        .expect("Failed to build API client");

    /// Semaphore to limit the number of concurrent API requests
    static ref REQUEST_SEMAPHORE: Semaphore = Semaphore::new(
        env::var("MAX_CONCURRENT_REQUESTS")
            .unwrap_or_else(|_| "5".to_string())
            .parse::<usize>()
            .expect("MAX_CONCURRENT_REQUESTS must be a valid usize")
    );
}

/// Processes a markdown content string by splitting it into blocks and sending each block to the Perplexity API.
///
/// # Arguments
///
/// * `file_content` - The content of the markdown file as a string.
///
/// # Returns
///
/// A `Result` containing the processed content or a `PerplexityError`.
pub async fn process_markdown(file_content: &str) -> Result<String, PerplexityError> {
    // Load settings from settings.toml
    let settings = Settings::new()?;

    // Split the content into markdown blocks
    let blocks = split_markdown_blocks(file_content);

    // Process each block concurrently, respecting the semaphore limit
    let results = stream::iter(blocks.into_iter())
        .map(|block| {
            let prompt = settings.prompt.clone();
            let topics = settings.topics.clone();
            let content = file_content.to_string();
            async move {
                let trimmed_block = block.trim().to_string();
                let context = select_context_blocks(&content, &trimmed_block);

                // Call the Perplexity API with retries
                let api_response = call_perplexity_api(&prompt, &context, &topics).await?;
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

    // Collect and join the processed blocks
    let processed_content = results.into_iter()
        .collect::<Result<Vec<String>, PerplexityError>>()?
        .join("\n");

    Ok(processed_content)
}

/// Calls the Perplexity API with the given prompt and context.
///
/// # Arguments
///
/// * `prompt` - The prompt to provide to the AI.
/// * `context` - The context for the AI to consider.
/// * `topics` - A list of topics to embed in the summary.
///
/// # Returns
///
/// A `Result` containing the API response as a string or a `PerplexityError`.
pub async fn call_perplexity_api(prompt: &str, context: &[String], topics: &[String]) -> Result<String, PerplexityError> {
    // Acquire a permit from the semaphore to limit concurrent requests
    let _permit = REQUEST_SEMAPHORE.acquire().await.unwrap();

    // Load Perplexity API configuration
    let settings = Settings::new()?;
    let perplexity_config = &settings.perplexity;

    // Retrieve retry configurations
    let max_retries: u32 = env::var("MAX_RETRIES").unwrap_or_else(|_| "3".to_string()).parse().unwrap_or(3);
    let retry_delay: u64 = env::var("RETRY_DELAY").unwrap_or_else(|_| "5".to_string()).parse().unwrap_or(5);

    // Construct the system message with topics
    let system_message = format!(
        "{}\nRelevant category topics are: {}.",
        prompt.trim(),
        topics.join(", ")
    );

    // Build the request payload
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

    // Attempt the API call with retries
    for attempt in 1..=max_retries {
        match API_CLIENT
            .post("https://api.perplexity.ai/chat/completions")
            .header("Authorization", format!("Bearer {}", perplexity_config.api_key))
            .json(&request)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    let response_text = response.text().await?;
                    return parse_perplexity_response(&response_text);
                } else {
                    error!(
                        "API request failed with status {} on attempt {} of {}",
                        response.status(),
                        attempt,
                        max_retries
                    );
                    if attempt < max_retries {
                        sleep(Duration::from_secs(retry_delay)).await;
                        continue;
                    } else {
                        return Err(PerplexityError::Api(format!("API request failed with status {}", response.status())));
                    }
                }
            }
            Err(e) => {
                error!("API request encountered an error: {} on attempt {} of {}", e, attempt, max_retries);
                if attempt < max_retries {
                    sleep(Duration::from_secs(retry_delay)).await;
                    continue;
                } else {
                    return Err(PerplexityError::Reqwest(e));
                }
            }
        }
    }

    Err(PerplexityError::Api("Max retries reached, API request failed".to_string()))
}

/// Parses the Perplexity API response.
///
/// # Arguments
///
/// * `response_text` - The raw response text from the API.
///
/// # Returns
///
/// A `Result` containing the content of the AI's message or a `PerplexityError`.
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

/// Splits markdown content into blocks using a markdown parser.
///
/// # Arguments
///
/// * `content` - The markdown content as a string slice.
///
/// # Returns
///
/// A vector of strings representing individual blocks.
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

/// Selects context blocks related to the active block.
///
/// # Arguments
///
/// * `content` - The full content of the markdown file.
/// * `active_block` - The block currently being processed.
///
/// # Returns
///
/// A vector of strings representing the context blocks.
pub fn select_context_blocks(_content: &str, active_block: &str) -> Vec<String> {
    // For now, only return the active block
    vec![active_block.to_string()]
}

/// Cleans Logseq links from the input text.
///
/// # Arguments
///
/// * `input` - The input text containing Logseq links.
///
/// # Returns
///
/// A string with Logseq links cleaned.
pub fn clean_logseq_links(input: &str) -> String {
    let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
    re.replace_all(input, "$1").to_string()
}

/// Processes an individual markdown block by integrating the API response.
///
/// # Arguments
///
/// * `input` - The original markdown block.
/// * `prompt` - The prompt used for processing.
/// * `topics` - The list of topics.
/// * `api_response` - The response from the Perplexity API.
///
/// # Returns
///
/// A string representing the processed markdown block.
pub fn process_markdown_block(input: &str, prompt: &str, topics: &[String], api_response: &str) -> String {
    let cleaned_input = clean_logseq_links(input);

    // Use the format! macro for better readability
    format!(
        "- ```\n{}```\nPrompt: {}\nTopics: {}\nResponse: {}",
        cleaned_input.trim_start_matches("- ").trim_end(),
        prompt,
        topics.join(", "),
        api_response
    )
}

/// Struct representing a Perplexity API request
#[derive(Debug, Serialize, Deserialize)]
pub struct PerplexityRequest {
    /// The model to use
    pub model: String,
    /// The messages to send in the conversation
    pub messages: Vec<Message>,
    /// Maximum number of tokens
    pub max_tokens: Option<u32>,
    /// Temperature setting
    pub temperature: Option<f32>,
    /// Top-p setting
    pub top_p: Option<f32>,
    /// Whether to return citations
    pub return_citations: Option<bool>,
    /// Whether to stream responses
    pub stream: Option<bool>,
    /// Presence penalty setting
    pub presence_penalty: Option<f32>,
    /// Frequency penalty setting
    pub frequency_penalty: Option<f32>,
}

/// Struct representing a message in the Perplexity API conversation
#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender (e.g., "system", "user", "assistant")
    pub role: String,
    /// The content of the message
    pub content: String,
}

/// Struct representing the Perplexity API response
#[derive(Debug, Deserialize)]
pub struct PerplexityResponse {
    /// Unique identifier for the response
    pub id: Option<String>,
    /// Model used for the response
    pub model: Option<String>,
    /// Type of the object returned
    pub object: Option<String>,
    /// Timestamp of creation
    pub created: Option<u64>,
    /// List of choices returned by the API
    pub choices: Vec<Choice>,
    /// Usage information
    pub usage: Option<Usage>,
}

/// Struct representing a choice in the Perplexity API response
#[derive(Debug, Deserialize)]
pub struct Choice {
    /// Index of the choice
    pub index: u32,
    /// Reason for finishing
    pub finish_reason: Option<String>,
    /// Message content
    pub message: Message,
    /// Delta updates (if streaming)
    pub delta: Option<Delta>,
}

/// Struct representing delta updates in the Perplexity API response
#[derive(Debug, Deserialize)]
pub struct Delta {
    /// Content of the delta
    pub content: Option<String>,
}

/// Struct representing usage information in the Perplexity API response
#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens
    pub total_tokens: u32,
}

/// Trait representing the PerplexityService interface
#[async_trait]
pub trait PerplexityService {
    /// Processes a file's content using the Perplexity API
    async fn process_file(file_content: String) -> Result<ProcessedFile, PerplexityError>;
}

/// Real implementation of the PerplexityService trait.
pub struct RealPerplexityService;

#[async_trait]
impl PerplexityService for RealPerplexityService {
    /// Processes the given file content using the Perplexity API
    ///
    /// # Arguments
    ///
    /// * `file_content` - The content of the markdown file as a string.
    ///
    /// # Returns
    ///
    /// A `Result` containing the processed content wrapped in `ProcessedFile`, or a `PerplexityError`.
    async fn process_file(file_content: String) -> Result<ProcessedFile, PerplexityError> {
        // Process the markdown content
        let processed_content = process_markdown(&file_content).await?;
        Ok(ProcessedFile { content: processed_content })
    }
}
