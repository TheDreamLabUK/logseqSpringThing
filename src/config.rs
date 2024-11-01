use config::{Config, ConfigError, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub prompt: String,
    pub topics: Vec<String>,
    pub github: GithubSettings,
    pub ragflow: RagFlowSettings,
    pub perplexity: PerplexitySettings,
    pub openai: OpenAISettings,
    pub default: DefaultSettings,
    pub visualization: VisualizationSettings,
    pub bloom: BloomSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GithubSettings {
    #[serde(alias = "GITHUB_ACCESS_TOKEN")]
    pub github_access_token: String,
    #[serde(alias = "GITHUB_OWNER")]
    pub github_owner: String,
    #[serde(alias = "GITHUB_REPO")]
    pub github_repo: String,
    #[serde(alias = "GITHUB_DIRECTORY")]
    pub github_directory: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagFlowSettings {
    #[serde(alias = "RAGFLOW_API_KEY")]
    pub ragflow_api_key: String,
    #[serde(alias = "RAGFLOW_BASE_URL")]
    pub ragflow_api_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    #[serde(alias = "OPENAI_API_KEY")]
    pub openai_api_key: String,
    #[serde(alias = "OPENAI_BASE_URL")]
    pub openai_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerplexitySettings {
    #[serde(alias = "PERPLEXITY_API_KEY")]
    pub perplexity_api_key: String,
    #[serde(alias = "PERPLEXITY_MODEL")]
    pub perplexity_model: String,
    #[serde(alias = "PERPLEXITY_API_URL")]
    pub perplexity_api_base_url: String,
    #[serde(alias = "PERPLEXITY_MAX_TOKENS")]
    pub perplexity_max_tokens: u32,
    #[serde(alias = "PERPLEXITY_TEMPERATURE")]
    pub perplexity_temperature: f32,
    #[serde(alias = "PERPLEXITY_TOP_P")]
    pub perplexity_top_p: f32,
    #[serde(alias = "PERPLEXITY_PRESENCE_PENALTY")]
    pub perplexity_presence_penalty: f32,
    #[serde(alias = "PERPLEXITY_FREQUENCY_PENALTY")]
    pub perplexity_frequency_penalty: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DefaultSettings {
    #[serde(alias = "MAX_CONCURRENT_REQUESTS")]
    pub max_concurrent_requests: u32,
    #[serde(alias = "MAX_RETRIES")]
    pub max_retries: u32,
    #[serde(alias = "RETRY_DELAY")]
    pub retry_delay: u32,
    #[serde(alias = "API_CLIENT_TIMEOUT")]
    pub api_client_timeout: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VisualizationSettings {
    pub node_color: String,
    pub edge_color: String,
    pub hologram_color: String,
    pub node_size_scaling_factor: u32,
    pub hologram_scale: u32,
    pub hologram_opacity: f32,
    pub edge_opacity: f32,
    pub label_font_size: u32,
    pub fog_density: f32,
    pub force_directed_iterations: u32,
    pub force_directed_repulsion: f32,
    pub force_directed_attraction: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BloomSettings {
    pub node_bloom_strength: f32,
    pub node_bloom_radius: f32,
    pub node_bloom_threshold: f32,
    pub edge_bloom_strength: f32,
    pub edge_bloom_radius: f32,
    pub edge_bloom_threshold: f32,
    pub environment_bloom_strength: f32,
    pub environment_bloom_radius: f32,
    pub environment_bloom_threshold: f32,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
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

        // Load settings.toml first as defaults
        log::debug!("Loading settings.toml as defaults...");
        let toml_settings = Config::builder()
            .add_source(File::with_name("settings.toml"))
            .build()?;

        // Create a new builder for environment variables
        log::debug!("Loading environment variables...");
        let env_settings = Config::builder()
            .add_source(
                Environment::default()
                    .separator("_")
                    .try_parsing(true)
                    .list_separator(",")
            )
            .build()?;

        // Merge settings with environment variables taking precedence
        let config = Config::builder()
            .add_source(toml_settings)
            .add_source(env_settings)
            .build()?;

        // Try to deserialize and log the results
        match config.try_deserialize::<Settings>() {
            Ok(s) => {
                log::debug!("Successfully loaded settings");
                
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
                    log::debug!("GitHub access token is present: {}", s.github.github_access_token.chars().take(10).collect::<String>());
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
            topics = ["Topic1", "Topic2"]

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
