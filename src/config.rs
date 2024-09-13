// src/config.rs

use serde::Deserialize;
use std::env;
use config::{Config, ConfigError, File};
use dotenv::dotenv;

/// Struct representing the settings loaded from settings.toml
#[derive(Debug, Deserialize)]
pub struct Settings {
    /// The prompt to provide to the AI assistant
    pub prompt: String,
    /// A list of topics to embed in the summary
    pub topics: Vec<String>,
    /// Perplexity API configuration
    pub perplexity: PerplexityConfig,
}

impl Settings {
    /// Loads the settings from the settings.toml file
    pub fn new() -> Result<Self, ConfigError> {
        let config = Config::builder()
            .add_source(File::with_name("settings.toml"))
            .build()?;

        config.try_deserialize()
    }
}

/// Struct representing the Perplexity API configuration
#[derive(Debug, Deserialize)]
pub struct PerplexityConfig {
    /// Perplexity API key
    pub api_key: String,
    /// Model to use for the API
    pub model: String,
    /// Maximum tokens for the API response
    pub max_tokens: u32,
    /// Temperature setting for the API
    pub temperature: f32,
    /// Top-p setting for the API
    pub top_p: f32,
    /// Presence penalty setting for the API
    pub presence_penalty: f32,
    /// Frequency penalty setting for the API
    pub frequency_penalty: f32,
}

impl PerplexityConfig {
    /// Loads the Perplexity API configuration from environment variables
    pub fn from_env() -> Result<Self, env::VarError> {
        dotenv().ok();

        Ok(PerplexityConfig {
            api_key: env::var("PERPLEXITY_API_KEY")?,
            model: env::var("PERPLEXITY_MODEL")
                .unwrap_or_else(|_| "llama-3.1-sonar-small-128k-online".to_string()),
            max_tokens: env::var("PERPLEXITY_MAX_TOKENS")
                .unwrap_or_else(|_| "4096".to_string())
                .parse()
                .map_err(|_| env::VarError::NotPresent)?,
            temperature: env::var("PERPLEXITY_TEMPERATURE")
                .unwrap_or_else(|_| "0.5".to_string())
                .parse()
                .map_err(|_| env::VarError::NotPresent)?,
            top_p: env::var("PERPLEXITY_TOP_P")
                .unwrap_or_else(|_| "0.9".to_string())
                .parse()
                .map_err(|_| env::VarError::NotPresent)?,
            presence_penalty: env::var("PERPLEXITY_PRESENCE_PENALTY")
                .unwrap_or_else(|_| "0.0".to_string())
                .parse()
                .map_err(|_| env::VarError::NotPresent)?,
            frequency_penalty: env::var("PERPLEXITY_FREQUENCY_PENALTY")
                .unwrap_or_else(|_| "1.0".to_string())
                .parse()
                .map_err(|_| env::VarError::NotPresent)?,
        })
    }
}
