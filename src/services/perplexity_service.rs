use crate::config::Settings;
use crate::models::metadata::Metadata;
use crate::services::file_service::ProcessedFile;
use chrono::Utc;
use log::{error, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

const MARKDOWN_DIR: &str = "data/markdown";

#[derive(Debug, Serialize, Deserialize)]
struct PerplexityResponse {
    content: String,
    link: String,
}

#[derive(Debug, Serialize)]
struct QueryRequest {
    query: String,
    conversation_id: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

pub struct PerplexityService {
    client: Client,
    settings: Arc<RwLock<Settings>>,
}

impl PerplexityService {
    pub async fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let timeout = {
            let settings_read = settings.read().await;
            settings_read.perplexity.timeout
        };

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()?;

        Ok(Self { 
            client,
            settings: Arc::clone(&settings)
        })
    }

    pub async fn query(&self, query: &str, conversation_id: &str) -> Result<String, Box<dyn StdError + Send + Sync>> {
        let settings = self.settings.read().await;
        let api_url = &settings.perplexity.api_url;
        info!("Sending query to Perplexity API: {}", api_url);

        let request = QueryRequest {
            query: query.to_string(),
            conversation_id: conversation_id.to_string(),
            model: settings.perplexity.model.clone(),
            max_tokens: settings.perplexity.max_tokens,
            temperature: settings.perplexity.temperature,
            top_p: settings.perplexity.top_p,
            presence_penalty: settings.perplexity.presence_penalty,
            frequency_penalty: settings.perplexity.frequency_penalty,
        };

        let response = self.client
            .post(api_url)
            .header("Authorization", format!("Bearer {}", settings.perplexity.api_key))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            error!("Perplexity API error: Status: {}, Error: {}", status, error_text);
            return Err(format!("Perplexity API error: {}", error_text).into());
        }

        let perplexity_response: PerplexityResponse = response.json().await?;
        Ok(perplexity_response.content)
    }

    pub async fn process_file(&self, file_name: &str) -> Result<ProcessedFile, Box<dyn StdError + Send + Sync>> {
        let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
        if !Path::new(&file_path).exists() {
            return Err(format!("File not found: {}", file_name).into());
        }

        let content = fs::read_to_string(&file_path)?;
        let settings = self.settings.read().await;
        
        let api_url = &settings.perplexity.api_url;
        info!("Sending request to Perplexity API: {}", api_url);

        let response = self.client
            .post(api_url)
            .header("Authorization", format!("Bearer {}", settings.perplexity.api_key))
            .json(&content)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            error!("Perplexity API error: Status: {}, Error: {}", status, error_text);
            return Err(format!("Perplexity API error: {}", error_text).into());
        }

        let perplexity_response: PerplexityResponse = response.json().await?;
        
        // Create metadata for processed file
        let metadata = Metadata {
            file_name: file_name.to_string(),
            file_size: perplexity_response.content.len(),
            node_size: 10.0, // Default size
            hyperlink_count: 0,
            sha1: String::new(),
            last_modified: Utc::now(),
            perplexity_link: perplexity_response.link,
            last_perplexity_process: Some(Utc::now()),
            topic_counts: HashMap::new(),
        };

        Ok(ProcessedFile {
            file_name: file_name.to_string(),
            content: perplexity_response.content,
            is_public: true,
            metadata,
        })
    }
}
