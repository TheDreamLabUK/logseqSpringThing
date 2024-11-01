use config::{Config, ConfigError, Environment, File};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    #[serde(default = "default_prompt")]
    pub prompt: String,
    #[serde(skip_deserializing)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub github: GithubSettings,
    #[serde(default)]
    pub ragflow: RagFlowSettings,
    #[serde(default)]
    pub perplexity: PerplexitySettings,
    #[serde(default)]
    pub openai: OpenAISettings,
    #[serde(default = "default_settings")]
    pub default: DefaultSettings,
    #[serde(default = "default_visualization")]
    pub visualization: VisualizationSettings,
    #[serde(default = "default_bloom")]
    pub bloom: BloomSettings,
}

fn default_prompt() -> String {
    "Your default prompt here".to_string()
}

fn default_settings() -> DefaultSettings {
    DefaultSettings {
        max_concurrent_requests: 2,
        max_retries: 3,
        retry_delay: 5,
        api_client_timeout: 30,
    }
}

fn default_visualization() -> VisualizationSettings {
    VisualizationSettings {
        node_color: "0x1A0B31".to_string(),
        edge_color: "0xff0000".to_string(),
        hologram_color: "0xFFD700".to_string(),
        node_size_scaling_factor: 5,
        hologram_scale: 5,
        hologram_opacity: 0.1,
        edge_opacity: 0.3,
        label_font_size: 36,
        fog_density: 0.002,
        force_directed_iterations: 100,
        force_directed_repulsion: 1.0,
        force_directed_attraction: 0.01,
    }
}

fn default_bloom() -> BloomSettings {
    BloomSettings {
        node_bloom_strength: 0.1,
        node_bloom_radius: 0.1,
        node_bloom_threshold: 0.0,
        edge_bloom_strength: 0.2,
        edge_bloom_radius: 0.3,
        edge_bloom_threshold: 0.0,
        environment_bloom_strength: 0.5,
        environment_bloom_radius: 0.1,
        environment_bloom_threshold: 0.0,
    }
}

fn default_openai_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_perplexity_api_url() -> String {
    "https://api.perplexity.ai/chat/completions".to_string()
}

fn default_topics() -> Vec<String> {
    vec!["default_topic".to_string()]
}

fn load_topics_from_markdown() -> Vec<String> {
    let markdown_dir = Path::new("/app/data/markdown");
    if !markdown_dir.exists() {
        log::info!("Markdown directory not found at {:?}, using default topics", markdown_dir);
        return default_topics();
    }

    match fs::read_dir(markdown_dir) {
        Ok(entries) => {
            let mut topics: Vec<String> = entries
                .filter_map(|entry| {
                    entry.ok().and_then(|e| {
                        let path = e.path();
                        if let Some(ext) = path.extension() {
                            if ext == "md" {
                                // Get the filename without extension
                                path.file_stem()
                                    .and_then(|s| s.to_str())
                                    .map(|s| s.trim_end_matches(".md").to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if topics.is_empty() {
                log::info!("No markdown files found in {:?}, using default topics", markdown_dir);
                default_topics()
            } else {
                // Sort topics for consistent ordering
                topics.sort();
                log::debug!("Loaded {} topics from markdown files", topics.len());
                log::debug!("Topics: {:?}", topics);
                topics
            }
        }
        Err(e) => {
            log::error!("Failed to read markdown directory: {}", e);
            default_topics()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GithubSettings {
    #[serde(alias = "GITHUB_ACCESS_TOKEN", alias = "github_access_token")]
    pub github_access_token: String,
    #[serde(alias = "GITHUB_OWNER", alias = "github_owner")]
    pub github_owner: String,
    #[serde(alias = "GITHUB_REPO", alias = "github_repo")]
    pub github_repo: String,
    #[serde(alias = "GITHUB_DIRECTORY", alias = "github_directory")]
    pub github_directory: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RagFlowSettings {
    #[serde(alias = "RAGFLOW_API_KEY", alias = "ragflow_api_key")]
    pub ragflow_api_key: String,
    #[serde(alias = "RAGFLOW_BASE_URL", alias = "ragflow_api_base_url")]
    pub ragflow_api_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct OpenAISettings {
    #[serde(alias = "OPENAI_API_KEY", alias = "openai_api_key")]
    pub openai_api_key: String,
    #[serde(alias = "OPENAI_BASE_URL", alias = "openai_base_url")]
    pub openai_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PerplexitySettings {
    #[serde(alias = "PERPLEXITY_API_KEY", alias = "perplexity_api_key")]
    pub perplexity_api_key: String,
    #[serde(alias = "PERPLEXITY_MODEL", alias = "perplexity_model")]
    pub perplexity_model: String,
    #[serde(alias = "PERPLEXITY_API_URL", alias = "perplexity_api_base_url")]
    pub perplexity_api_base_url: String,
    #[serde(alias = "PERPLEXITY_MAX_TOKENS", alias = "perplexity_max_tokens")]
    pub perplexity_max_tokens: u32,
    #[serde(alias = "PERPLEXITY_TEMPERATURE", alias = "perplexity_temperature")]
    pub perplexity_temperature: f32,
    #[serde(alias = "PERPLEXITY_TOP_P", alias = "perplexity_top_p")]
    pub perplexity_top_p: f32,
    #[serde(alias = "PERPLEXITY_PRESENCE_PENALTY", alias = "perplexity_presence_penalty")]
    pub perplexity_presence_penalty: f32,
    #[serde(alias = "PERPLEXITY_FREQUENCY_PENALTY", alias = "perplexity_frequency_penalty")]
    pub perplexity_frequency_penalty: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DefaultSettings {
    #[serde(alias = "MAX_CONCURRENT_REQUESTS", alias = "max_concurrent_requests")]
    pub max_concurrent_requests: u32,
    #[serde(alias = "MAX_RETRIES", alias = "max_retries")]
    pub max_retries: u32,
    #[serde(alias = "RETRY_DELAY", alias = "retry_delay")]
    pub retry_delay: u32,
    #[serde(alias = "API_CLIENT_TIMEOUT", alias = "api_client_timeout")]
    pub api_client_timeout: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct VisualizationSettings {
    #[serde(alias = "NODE_COLOR", alias = "node_color")]
    pub node_color: String,
    #[serde(alias = "EDGE_COLOR", alias = "edge_color")]
    pub edge_color: String,
    #[serde(alias = "HOLOGRAM_COLOR", alias = "hologram_color")]
    pub hologram_color: String,
    #[serde(alias = "NODE_SIZE_SCALING_FACTOR", alias = "node_size_scaling_factor")]
    pub node_size_scaling_factor: u32,
    #[serde(alias = "HOLOGRAM_SCALE", alias = "hologram_scale")]
    pub hologram_scale: u32,
    #[serde(alias = "HOLOGRAM_OPACITY", alias = "hologram_opacity")]
    pub hologram_opacity: f32,
    #[serde(alias = "EDGE_OPACITY", alias = "edge_opacity")]
    pub edge_opacity: f32,
    #[serde(alias = "LABEL_FONT_SIZE", alias = "label_font_size")]
    pub label_font_size: u32,
    #[serde(alias = "FOG_DENSITY", alias = "fog_density")]
    pub fog_density: f32,
    #[serde(alias = "FORCE_DIRECTED_ITERATIONS", alias = "force_directed_iterations")]
    pub force_directed_iterations: u32,
    #[serde(alias = "FORCE_DIRECTED_REPULSION", alias = "force_directed_repulsion")]
    pub force_directed_repulsion: f32,
    #[serde(alias = "FORCE_DIRECTED_ATTRACTION", alias = "force_directed_attraction")]
    pub force_directed_attraction: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BloomSettings {
    #[serde(alias = "NODE_BLOOM_STRENGTH", alias = "node_bloom_strength")]
    pub node_bloom_strength: f32,
    #[serde(alias = "NODE_BLOOM_RADIUS", alias = "node_bloom_radius")]
    pub node_bloom_radius: f32,
    #[serde(alias = "NODE_BLOOM_THRESHOLD", alias = "node_bloom_threshold")]
    pub node_bloom_threshold: f32,
    #[serde(alias = "EDGE_BLOOM_STRENGTH", alias = "edge_bloom_strength")]
    pub edge_bloom_strength: f32,
    #[serde(alias = "EDGE_BLOOM_RADIUS", alias = "edge_bloom_radius")]
    pub edge_bloom_radius: f32,
    #[serde(alias = "EDGE_BLOOM_THRESHOLD", alias = "edge_bloom_threshold")]
    pub edge_bloom_threshold: f32,
    #[serde(alias = "ENVIRONMENT_BLOOM_STRENGTH", alias = "environment_bloom_strength")]
    pub environment_bloom_strength: f32,
    #[serde(alias = "ENVIRONMENT_BLOOM_RADIUS", alias = "environment_bloom_radius")]
    pub environment_bloom_radius: f32,
    #[serde(alias = "ENVIRONMENT_BLOOM_THRESHOLD", alias = "environment_bloom_threshold")]
    pub environment_bloom_threshold: f32,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        // Load .env file first
        dotenv().ok();

        let run_mode = std::env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        log::debug!("Loading configuration for mode: {}", run_mode);

        // Log environment variables for debugging
        log::debug!("Environment variables:");
        for (key, value) in std::env::vars() {
            if key.starts_with("GITHUB_") || key.starts_with("RAGFLOW_") || 
               key.starts_with("PERPLEXITY_") || key.starts_with("OPENAI_") {
                log::debug!("{}={}", key, if key.contains("TOKEN") || key.contains("KEY") { "[REDACTED]" } else { &value });
            }
        }

        // Create a builder and add sources in order of precedence (later sources override earlier ones)
        let mut builder = Config::builder();

        // 1. Add settings.toml for default values
        let settings_file = File::with_name("settings.toml").required(false);
        builder = builder.add_source(settings_file);
        log::debug!("Attempted to add settings.toml to configuration sources");

        // 2. Add environment variables with higher precedence
        builder = builder.add_source(
            Environment::default()
                .separator("_")
                .try_parsing(true)
                .ignore_empty(true)  // Ignore empty environment variables
        );

        // Build the config
        let config = builder.build()?;

        // Try to deserialize and log the results
        match config.try_deserialize::<Settings>() {
            Ok(mut s) => {
                log::debug!("Successfully loaded settings");
                
                // Load topics from markdown files
                s.topics = load_topics_from_markdown();
                
                // Log non-sensitive settings
                log::debug!("GitHub settings: owner={}, repo={}, directory={}", 
                    s.github.github_owner,
                    s.github.github_repo,
                    s.github.github_directory
                );
                log::debug!("RAGFlow base URL: {}", s.ragflow.ragflow_api_base_url);
                log::debug!("Perplexity model: {}", s.perplexity.perplexity_model);

                // Log presence of sensitive values without showing them
                if !s.github.github_access_token.is_empty() {
                    log::debug!("GitHub access token is present");
                } else {
                    log::warn!("GitHub access token is missing");
                }
                if !s.ragflow.ragflow_api_key.is_empty() {
                    log::debug!("RAGFlow API key is present");
                }
                if !s.perplexity.perplexity_api_key.is_empty() {
                    log::debug!("Perplexity API key is present");
                }
                if !s.openai.openai_api_key.is_empty() {
                    log::debug!("OpenAI API key is present");
                }

                Ok(s)
            }
            Err(e) => {
                log::error!("Failed to deserialize settings: {}", e);
                Err(e)
            }
        }
    }
}

impl fmt::Display for GithubSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GithubSettings {{ access_token: [REDACTED], owner: {}, repo: {} }}", 
               self.github_owner, self.github_repo)
    }
}

impl fmt::Display for RagFlowSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RagFlowSettings {{ api_base_url: {}, api_key: [REDACTED] }}", 
               self.ragflow_api_base_url)
    }
}

impl fmt::Display for OpenAISettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenAISettings {{ base_url: {}, api_key: [REDACTED] }}", 
               self.openai_base_url)
    }
}

impl fmt::Display for PerplexitySettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PerplexitySettings {{ api_base_url: {}, api_key: [REDACTED], model: {} }}", 
               self.perplexity_api_base_url, self.perplexity_model)
    }
}

impl fmt::Display for DefaultSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DefaultSettings {{ max_concurrent_requests: {}, max_retries: {}, retry_delay: {}, api_client_timeout: {} }}", 
               self.max_concurrent_requests,
               self.max_retries,
               self.retry_delay,
               self.api_client_timeout)
    }
}

impl fmt::Display for VisualizationSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VisualizationSettings {{ node_color: {}, edge_color: {}, iterations: {}, repulsion: {}, attraction: {} }}", 
               self.node_color,
               self.edge_color,
               self.force_directed_iterations,
               self.force_directed_repulsion,
               self.force_directed_attraction)
    }
}

impl fmt::Display for BloomSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomSettings {{ node_strength: {}, edge_strength: {}, environment_strength: {} }}", 
               self.node_bloom_strength,
               self.edge_bloom_strength,
               self.environment_bloom_strength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use std::io::Write;

    #[test]
    fn test_settings_from_files() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path();

        let default_config = r#"
prompt = "Test prompt"

[github]
github_access_token = "test_token"
github_owner = "test_owner"
github_repo = "test_repo"
github_directory = "test_directory"

[ragflow]
ragflow_api_key = "test_key"
ragflow_api_base_url = "http://test.example.com"

[openai]
openai_api_key = "test_openai_key"
openai_base_url = "https://api.openai.com/v1"

[perplexity]
perplexity_api_key = "test_perplexity_key"
perplexity_model = "test_model"
perplexity_api_base_url = "test_url"
perplexity_max_tokens = 4096
perplexity_temperature = 0.7
perplexity_top_p = 1.0
perplexity_presence_penalty = 0.0
perplexity_frequency_penalty = 0.0

[default]
max_concurrent_requests = 5
max_retries = 3
retry_delay = 5
api_client_timeout = 30

[visualization]
node_color = "0x1A0B31"
edge_color = "0xff0000"
hologram_color = "0xFFD700"
node_size_scaling_factor = 5
hologram_scale = 5
hologram_opacity = 0.1
edge_opacity = 0.3
label_font_size = 36
fog_density = 0.002
force_directed_iterations = 100
force_directed_repulsion = 1.0
force_directed_attraction = 0.01

[bloom]
node_bloom_strength = 0.1
node_bloom_radius = 0.1
node_bloom_threshold = 0.0
edge_bloom_strength = 0.2
edge_bloom_radius = 0.3
edge_bloom_threshold = 0.0
environment_bloom_strength = 0.5
environment_bloom_radius = 0.1
environment_bloom_threshold = 0.0
"#;

        let settings_path = config_path.join("settings.toml");
        let mut file = fs::File::create(&settings_path)?;
        file.write_all(default_config.as_bytes())?;

        std::env::set_var("CONFIG_DIR", config_path.to_str().unwrap());

        // Test environment variable override
        std::env::set_var("GITHUB_OWNER", "env_owner");
        let settings = Settings::new()?;
        assert_eq!(settings.github.github_owner, "env_owner");

        Ok(())
    }
}
