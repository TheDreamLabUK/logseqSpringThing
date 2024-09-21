use serde::Deserialize;
use config::{Config, ConfigError, File, Environment};
use dotenv::dotenv;
use std::path::Path;
use std::fs::File as StdFile;
use std::io::{BufRead, BufReader};
use log::{info, error};

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub prompt: String,
    #[serde(skip)]
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

        // Load default settings from settings.toml in the current directory
        let base_settings_path = Path::new("settings.toml");
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

        // Load topics from CSV file in the new volume
        settings.topics = Self::load_topics_from_csv("/app/data/topics.csv")?;
        info!("Loaded topics: {:?}", settings.topics);

        info!("Final parsed configuration: {:#?}", settings);

        Ok(settings)
    }

    fn load_topics_from_csv(file_path: &str) -> Result<Vec<String>, ConfigError> {
        let file = StdFile::open(file_path).map_err(|e| ConfigError::Message(format!("Failed to open topics.csv: {}", e)))?;
        let reader = BufReader::new(file);
        let topics: Vec<String> = reader.lines()
            .filter_map(Result::ok)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        if topics.is_empty() {
            Err(ConfigError::Message("No topics found in topics.csv".to_string()))
        } else {
            Ok(topics)
        }
    }
}
