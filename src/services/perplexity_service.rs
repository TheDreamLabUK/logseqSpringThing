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

pub struct PerplexityService {
    client: Client,
    settings: Arc<RwLock<Settings>>,
}

impl PerplexityService {
    pub fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self { client, settings })
    }

    pub async fn process_file(&self, file_name: &str) -> Result<ProcessedFile, Box<dyn StdError + Send + Sync>> {
        let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
        if !Path::new(&file_path).exists() {
            return Err(format!("File not found: {}", file_name).into());
        }

        let content = fs::read_to_string(&file_path)?;
        let settings = self.settings.read().await;
        
        let api_url = format!("{}/process", settings.network.domain);
        info!("Sending request to Perplexity API: {}", api_url);

        let response = self.client
            .post(&api_url)
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
