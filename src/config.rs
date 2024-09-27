// config.rs

use serde::Deserialize;
use config::{Config, ConfigError, File, Environment};
use dotenv::dotenv;
use std::path::Path;
use std::fs::File as StdFile;
use std::io::{BufRead, BufReader};
use log::{info, error, debug};
use std::env;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub prompt: String,
    #[serde(skip)]
    pub topics: Vec<String>,
    pub perplexity: PerplexityConfig,
    #[serde(default)]
    pub ragflow: RAGFlowConfig,
    #[serde(default)]
    pub github: GitHubConfig,
    pub default: DefaultConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PerplexityConfig {
    pub perplexity_api_key: String,
    pub perplexity_model: String,
    pub perplexity_api_base_url: String,
    pub perplexity_max_tokens: u32,
    pub perplexity_temperature: f32,
    pub perplexity_top_p: f32,
    pub perplexity_presence_penalty: f32,
    pub perplexity_frequency_penalty: f32,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct RAGFlowConfig {
    pub ragflow_api_key: String,
    pub ragflow_api_base_url: String,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct GitHubConfig {
    pub github_access_token: String,
    pub github_owner: String,
    pub github_repo: String,
    pub github_directory: String,
}

#[derive(Debug, Deserialize)]
pub struct DefaultConfig {
    pub max_concurrent_requests: u32,
    pub max_retries: u32,
    pub retry_delay: u64,
    pub api_client_timeout: u64,
}

impl Settings {
    /// Creates a new `Settings` instance by loading configurations from files and environment variables.
    pub fn new() -> Result<Self, ConfigError> {
        // Load environment variables from .env file if it exists
        dotenv().ok();
        debug!("Loaded .env file");

        // Log the current working directory for debugging purposes
        info!("Current working directory: {:?}", std::env::current_dir());

        // Initialize the configuration builder
        let mut builder = Config::builder();

        // Path to the default settings file
        let base_settings_path = Path::new("settings.toml");
        if base_settings_path.exists() {
            info!("Loading default settings from {:?}", base_settings_path);
            builder = builder.add_source(File::from(base_settings_path).required(true));
        } else {
            error!("Default settings file not found at {:?}", base_settings_path);
            return Err(ConfigError::NotFound("settings.toml".into()));
        }

        // Add environment variables to the configuration, using '_' as a separator
        builder = builder.add_source(Environment::default().separator("_"));
        info!("Loading environment variables");

        // Build the configuration
        let config = builder.build()?;
        info!("Raw configuration: {:#?}", config);

        // Clone the config before deserializing to retain ownership for later use
        let config_clone = config.clone();
        let mut settings: Settings = config_clone.try_deserialize()?;

        // Load and override specific configurations from environment variables or other sources
        settings.load_github_config(&config)?;
        settings.load_ragflow_config(&config)?;
        settings.load_perplexity_config(&config)?;
        settings.load_default_config(&config)?;
        settings.load_topics_from_csv("data/topics.csv")?;

        info!("Loaded topics: {:?}", settings.topics);
        info!("Final parsed configuration: {:#?}", settings);

        Ok(settings)
    }

    /// Loads and updates the GitHub configuration.
    fn load_github_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading GitHub config...");

        // Attempt to load each GitHub configuration from environment variables first, then from the config file
        let access_token = env::var("GITHUB_ACCESS_TOKEN")
            .or_else(|_| config.get_string("github.github_access_token"))
            .unwrap_or_default();

        let github_owner = env::var("GITHUB_OWNER")
            .or_else(|_| config.get_string("github.github_owner"))
            .unwrap_or_default();

        let github_repo = env::var("GITHUB_REPO")
            .or_else(|_| config.get_string("github.github_repo"))
            .unwrap_or_default();

        let github_directory = env::var("GITHUB_DIRECTORY")
            .or_else(|_| config.get_string("github.github_directory"))
            .unwrap_or_default();

        // Update the `github` field of `Settings`
        self.github = GitHubConfig {
            github_access_token: access_token,
            github_owner,
            github_repo,
            github_directory,
        };

        debug!("Loaded GitHub config: {:?}", self.github);

        // Validate that the GitHub access token is present
        if self.github.github_access_token.is_empty() {
            error!("GitHub Access Token is empty");
            return Err(ConfigError::NotFound("github.github_access_token".into()));
        }

        Ok(())
    }

    /// Loads and updates the RAGFlow configuration.
    fn load_ragflow_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading RAGFlow config...");

        // Attempt to load RAGFlow API key and base URL from environment variables first, then from the config file
        let api_key = env::var("RAGFLOW_API_KEY")
            .or_else(|_| config.get_string("ragflow.ragflow_api_key"))
            .unwrap_or_default();

        let base_url = env::var("RAGFLOW_API_BASE_URL") // Ensure the environment variable name matches your setup
            .or_else(|_| config.get_string("ragflow.ragflow_api_base_url"))
            .unwrap_or_default();

        // Update the `ragflow` field of `Settings`
        self.ragflow = RAGFlowConfig {
            ragflow_api_key: api_key,
            ragflow_api_base_url: base_url,
        };

        debug!("Loaded RAGFlow config: {:?}", self.ragflow);

        // Validate that the RAGFlow API key is present
        if self.ragflow.ragflow_api_key.is_empty() {
            error!("RAGFlow API key is empty");
            return Err(ConfigError::NotFound("ragflow.ragflow_api_key".into()));
        }

        Ok(())
    }

    /// Loads and updates the Perplexity configuration.
    fn load_perplexity_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading Perplexity config...");

        // Attempt to load each Perplexity configuration from environment variables first, then from the config file
        let api_key = env::var("PERPLEXITY_API_KEY")
            .or_else(|_| config.get_string("perplexity.perplexity_api_key"))
            .unwrap_or_default();

        let perplexity_model = env::var("PERPLEXITY_MODEL")
            .or_else(|_| config.get_string("perplexity.perplexity_model"))
            .unwrap_or_default();

        let perplexity_api_base_url = env::var("PERPLEXITY_API_URL") // Ensure the environment variable name matches your setup
            .or_else(|_| config.get_string("perplexity.perplexity_api_base_url"))
            .unwrap_or_default();

        let perplexity_max_tokens = env::var("PERPLEXITY_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("perplexity.perplexity_max_tokens").unwrap_or(4096) as u32);

        let perplexity_temperature = env::var("PERPLEXITY_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_float("perplexity.perplexity_temperature").unwrap_or(0.7) as f32);

        let perplexity_top_p = env::var("PERPLEXITY_TOP_P")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_float("perplexity.perplexity_top_p").unwrap_or(1.0) as f32);

        let perplexity_presence_penalty = env::var("PERPLEXITY_PRESENCE_PENALTY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_float("perplexity.perplexity_presence_penalty").unwrap_or(0.0) as f32);

        let perplexity_frequency_penalty = env::var("PERPLEXITY_FREQUENCY_PENALTY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_float("perplexity.perplexity_frequency_penalty").unwrap_or(0.0) as f32);

        // Update the `perplexity` field of `Settings`
        self.perplexity = PerplexityConfig {
            perplexity_api_key: api_key,
            perplexity_model,
            perplexity_api_base_url,
            perplexity_max_tokens,
            perplexity_temperature,
            perplexity_top_p,
            perplexity_presence_penalty,
            perplexity_frequency_penalty,
        };

        debug!("Loaded Perplexity config: {:?}", self.perplexity);

        // Validate that the Perplexity API key is present
        if self.perplexity.perplexity_api_key.is_empty() {
            error!("Perplexity API key is empty");
            return Err(ConfigError::NotFound("perplexity.perplexity_api_key".into()));
        }

        Ok(())
    }

    /// Loads and updates the Default configuration.
    fn load_default_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading Default config...");

        // Attempt to load each Default configuration from environment variables first, then from the config file
        let max_concurrent_requests = env::var("DEFAULT_MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.max_concurrent_requests").unwrap_or(10) as u32);

        let max_retries = env::var("DEFAULT_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.max_retries").unwrap_or(3) as u32);

        let retry_delay = env::var("DEFAULT_RETRY_DELAY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.retry_delay").unwrap_or(1000) as u64);

        let api_client_timeout = env::var("DEFAULT_API_CLIENT_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.api_client_timeout").unwrap_or(5000) as u64);

        // Update the `default` field of `Settings`
        self.default = DefaultConfig {
            max_concurrent_requests,
            max_retries,
            retry_delay,
            api_client_timeout,
        };

        debug!("Loaded Default config: {:?}", self.default);

        Ok(())
    }

    /// Loads topics from a CSV file and updates the `topics` field.
    fn load_topics_from_csv(&mut self, file_path: &str) -> Result<(), ConfigError> {
        debug!("Loading topics from CSV at path: {}", file_path);

        // Log the absolute path
        let absolute_path = std::fs::canonicalize(file_path).map_err(|e| {
            error!("Failed to get absolute path for {}: {}", file_path, e);
            ConfigError::Message(format!("Failed to get absolute path: {}", e))
        })?;
        debug!("Absolute path of topics.csv: {:?}", absolute_path);

        // Check if the file exists
        if !absolute_path.exists() {
            error!("topics.csv does not exist at {:?}", absolute_path);
            return Err(ConfigError::Message(format!("topics.csv does not exist at {:?}", absolute_path)));
        }

        // Attempt to open the CSV file
        let file = StdFile::open(&absolute_path).map_err(|e| {
            error!("Failed to open topics.csv at {:?}: {}", absolute_path, e);
            ConfigError::Message(format!(
                "Failed to open topics.csv at {:?}: {}. Make sure the file exists and has correct permissions.",
                absolute_path, e
            ))
        })?;

        let reader = BufReader::new(file);
        let topics: Vec<String> = reader
            .lines()
            .enumerate()
            .filter_map(|(i, line)| {
                match line {
                    Ok(l) => {
                        let trimmed = l.trim().to_string();
                        if trimmed.is_empty() {
                            debug!("Skipping empty line {} in topics.csv", i + 1);
                            None
                        } else {
                            Some(trimmed)
                        }
                    },
                    Err(e) => {
                        error!("Error reading line {} from topics.csv: {}", i + 1, e);
                        None
                    }
                }
            })
            .collect();

        // Update the `topics` field of `Settings`
        self.topics = topics;

        // Validate that at least one topic was loaded
        if self.topics.is_empty() {
            error!("No topics found in topics.csv at {:?}", absolute_path);
            Err(ConfigError::Message(
                format!("No topics found in topics.csv at {:?}", absolute_path)
            ))
        } else {
            debug!("Successfully loaded {} topics from {:?}", self.topics.len(), absolute_path);
            Ok(())
        }
    }
}
