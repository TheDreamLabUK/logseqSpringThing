use serde::Deserialize;
use config::{Config, ConfigError, File, Environment};
use dotenv::dotenv;
use std::path::Path;
use log::{info, error};

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub prompt: String,
    #[serde(default)]
    pub topics: Vec<String>,
    pub perplexity: PerplexityConfig,
    pub default: DefaultConfig,
}

#[derive(Debug, Deserialize)]
pub struct PerplexityConfig {
    pub api_key: String,
    pub model: String,
    pub api_base_url: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

#[derive(Debug, Deserialize)]
pub struct DefaultConfig {
    pub max_concurrent_requests: u32,
    pub max_retries: u32,
    pub retry_delay: u32,
    pub api_client_timeout: u32,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        dotenv().ok();

        info!("Current working directory: {:?}", std::env::current_dir());

        let mut builder = Config::builder();

        // Load default settings from settings.toml
        let base_settings_path = Path::new("/app/settings.toml");
        if base_settings_path.exists() {
            info!("Loading default settings from {:?}", base_settings_path);
            builder = builder.add_source(File::from(base_settings_path).required(true));
        } else {
            error!("Default settings file not found at {:?}", base_settings_path);
            return Err(ConfigError::NotFound("settings.toml".into()));
        }

        // Load environment variables (overriding settings.toml)
        builder = builder.add_source(Environment::default().separator("__"));
        info!("Loading environment variables");

        let config_map = builder.build()?;
        info!("Raw configuration: {:#?}", config_map);

        // Deserialize into Settings struct
        let mut settings: Settings = config_map.try_deserialize()?;

        // Special handling for topics if it's a single string in env var
        if let Ok(topics_str) = std::env::var("TOPICS") {
            info!("Overriding topics from environment variable");
            // Split the string into individual topics
            settings.topics = topics_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else {
            info!("Using topics from settings.toml");
        }

        info!("Final parsed configuration: {:#?}", settings);

        Ok(settings)
    }
}
