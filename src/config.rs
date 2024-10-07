use serde::Deserialize;
use config::{Config, ConfigError, File as ConfigFile, Environment};
use dotenv::dotenv;
use std::path::Path;
use log::{info, error, debug};
use std::env;
use std::io::{BufReader, BufRead};
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub prompt: String,
    #[serde(skip)]
    pub topics: Vec<String>,
    pub perplexity: PerplexityConfig,
    pub ragflow: RagFlowConfig,
    pub github: GitHubConfig,
    pub default: DefaultConfig,
    pub sonata: SonataSettings,
    pub openai: Option<OpenAISettings>,
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
pub struct RagFlowConfig {
    pub api_key: String,
    pub base_url: String,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct GitHubConfig {
    pub github_access_token: String,
    pub github_owner: String,
    pub github_repo: String,
    pub github_directory: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DefaultConfig {
    pub max_concurrent_requests: u32,
    pub max_retries: u32,
    pub retry_delay: u64,
    pub api_client_timeout: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SonataSettings {
    pub voice_config_path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAISettings {
    pub openai_api_key: String,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        dotenv().ok();
        debug!("Loaded .env file");

        info!("Current working directory: {:?}", std::env::current_dir());

        let mut builder = Config::builder();

        let base_settings_path = Path::new("settings.toml");
        if base_settings_path.exists() {
            info!("Loading default settings from {:?}", base_settings_path);
            builder = builder.add_source(ConfigFile::from(base_settings_path).required(true));
        } else {
            error!("Default settings file not found at {:?}", base_settings_path);
            return Err(ConfigError::NotFound("settings.toml".into()));
        }

        builder = builder.add_source(Environment::default().separator("_"));
        info!("Loading environment variables");

        let config = builder.build()?;
        info!("Raw configuration: {:#?}", config);

        let config_clone = config.clone();
        let mut settings: Settings = config_clone.try_deserialize()?;

        settings.load_github_config(&config)?;
        settings.load_ragflow_config(&config)?;
        settings.load_perplexity_config(&config)?;
        settings.load_default_config(&config)?;
        settings.load_openai_config(&config);
        settings.load_topics_from_csv("data/topics.csv")?;

        info!("Loaded topics: {:?}", settings.topics);
        info!("Final parsed configuration: {:#?}", settings);

        Ok(settings)
    }

    fn load_github_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading GitHub config...");

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

        self.github = GitHubConfig {
            github_access_token: access_token,
            github_owner,
            github_repo,
            github_directory,
        };

        debug!("Loaded GitHub config: {:?}", self.github);

        if self.github.github_access_token.is_empty() {
            error!("GitHub Access Token is empty");
            return Err(ConfigError::NotFound("github.github_access_token".into()));
        }

        Ok(())
    }

    fn load_ragflow_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading RAGFlow config...");

        let api_key = env::var("RAGFLOW_API_KEY")
            .or_else(|_| config.get_string("ragflow.api_key"))
            .unwrap_or_default();

        let base_url = env::var("RAGFLOW_BASE_URL")
            .or_else(|_| config.get_string("ragflow.base_url"))
            .unwrap_or_default();

        self.ragflow = RagFlowConfig {
            api_key,
            base_url,
        };

        debug!("Loaded RAGFlow config: {:?}", self.ragflow);

        if self.ragflow.api_key.is_empty() {
            error!("RAGFlow API key is empty");
            return Err(ConfigError::NotFound("ragflow.api_key".into()));
        }

        Ok(())
    }

    fn load_perplexity_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading Perplexity config...");

        let api_key = env::var("PERPLEXITY_API_KEY")
            .or_else(|_| config.get_string("perplexity.perplexity_api_key"))
            .unwrap_or_default();

        let perplexity_model = env::var("PERPLEXITY_MODEL")
            .or_else(|_| config.get_string("perplexity.perplexity_model"))
            .unwrap_or_default();

        let perplexity_api_base_url = env::var("PERPLEXITY_API_URL")
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

        if self.perplexity.perplexity_api_key.is_empty() {
            error!("Perplexity API key is empty");
            return Err(ConfigError::NotFound("perplexity.perplexity_api_key".into()));
        }

        Ok(())
    }

    fn load_default_config(&mut self, config: &Config) -> Result<(), ConfigError> {
        debug!("Loading Default config...");

        let max_concurrent_requests = env::var("MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.max_concurrent_requests").unwrap_or(10) as u32);

        let max_retries = env::var("MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.max_retries").unwrap_or(3) as u32);

        let retry_delay = env::var("RETRY_DELAY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.retry_delay").unwrap_or(1000) as u64);

        let api_client_timeout = env::var("API_CLIENT_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| config.get_int("default.api_client_timeout").unwrap_or(5000) as u64);

        self.default = DefaultConfig {
            max_concurrent_requests,
            max_retries,
            retry_delay,
            api_client_timeout,
        };

        debug!("Loaded Default config: {:?}", self.default);

        Ok(())
    }

    fn load_openai_config(&mut self, config: &Config) {
        debug!("Loading OpenAI config...");

        let openai_api_key = env::var("OPENAI_API_KEY")
            .or_else(|_| config.get_string("openai.openai_api_key"))
            .ok();

        if let Some(openai_api_key) = openai_api_key {
            self.openai = Some(OpenAISettings { openai_api_key });
            debug!("Loaded OpenAI config: {:?}", self.openai);
        } else {
            debug!("OpenAI API key not found, skipping OpenAI configuration");
        }
    }

    fn load_topics_from_csv(&mut self, file_path: &str) -> Result<(), ConfigError> {
        debug!("Loading topics from CSV at path: {}", file_path);

        let absolute_path = std::fs::canonicalize(file_path).map_err(|e| {
            error!("Failed to get absolute path for {}: {}", file_path, e);
            ConfigError::Message(format!("Failed to get absolute path: {}", e))
        })?;
        debug!("Absolute path of topics.csv: {:?}", absolute_path);

        if !absolute_path.exists() {
            error!("topics.csv does not exist at {:?}", absolute_path);
            return Err(ConfigError::Message(format!("topics.csv does not exist at {:?}", absolute_path)));
        }

        let file = File::open(&absolute_path).map_err(|e| {
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

        self.topics = topics;

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