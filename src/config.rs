use config::{Config, ConfigError, Environment, File};
use log::{debug, error};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub server: ServerSettings,
    pub github: GithubSettings,
    pub ragflow: RagFlowSettings,
    pub openai: OpenAISettings,
    pub perplexity: PerplexitySettings,
    pub visualization: VisualizationSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub ssl_cert_path: Option<PathBuf>,
    pub ssl_key_path: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GithubSettings {
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub directory: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagFlowSettings {
    pub api_key: String,
    pub api_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    pub openai_api_key: String,
    pub openai_base_url: String,
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerplexitySettings {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VisualizationSettings {
    pub force_directed_iterations: u32,
    pub force_directed_repulsion: f32,
    pub force_directed_attraction: f32,
    pub force_directed_damping: f32,
}

impl Default for VisualizationSettings {
    fn default() -> Self {
        Self {
            force_directed_iterations: 100,
            force_directed_repulsion: 1.0,
            force_directed_attraction: 0.5,
            force_directed_damping: 0.9,
        }
    }
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let run_mode = std::env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        debug!("Loading configuration for mode: {}", run_mode);

        let config_dir = std::env::var("CONFIG_DIR").unwrap_or_else(|_| "./config".into());
        let settings = Config::builder()
            .add_source(File::with_name(&format!("{}/default", config_dir)))
            .add_source(File::with_name(&format!("{}/{}", config_dir, run_mode)).required(false))
            .add_source(Environment::with_prefix("app").separator("__"))
            .build()?;

        settings.try_deserialize()
    }
}

impl fmt::Display for ServerSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ServerSettings {{ host: {}, port: {} }}", self.host, self.port)
    }
}

impl fmt::Display for GithubSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GithubSettings {{ token: [REDACTED], owner: {}, repo: {} }}", 
               self.owner, self.repo)
    }
}

impl fmt::Display for RagFlowSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RagFlowSettings {{ api_url: {}, api_key: [REDACTED] }}", 
               self.api_url)
    }
}

impl fmt::Display for OpenAISettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenAISettings {{ base_url: {}, api_key: [REDACTED], model: {} }}", 
               self.openai_base_url, self.model)
    }
}

impl fmt::Display for PerplexitySettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PerplexitySettings {{ base_url: {}, api_key: [REDACTED], model: {} }}", 
               self.base_url, self.model)
    }
}

impl fmt::Display for VisualizationSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VisualizationSettings {{ iterations: {}, repulsion: {}, attraction: {}, damping: {} }}", 
               self.force_directed_iterations,
               self.force_directed_repulsion,
               self.force_directed_attraction,
               self.force_directed_damping)
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
            [server]
            host = "127.0.0.1"
            port = 8080

            [github]
            token = "default_token"
            owner = "default_owner"
            repo = "default_repo"

            [ragflow]
            api_url = "http://default.example.com"
            api_key = "default_key"

            [openai]
            openai_api_key = "default_openai_key"
            openai_base_url = "https://api.openai.com"
            model = "gpt-4"

            [perplexity]
            api_key = "default_perplexity_key"
            base_url = "https://api.perplexity.ai"
            model = "mixtral-8x7b"
            max_tokens = 2048
            temperature = 0.7
            top_p = 0.9
            presence_penalty = 0.0
            frequency_penalty = 0.0

            [visualization]
            force_directed_iterations = 100
            force_directed_repulsion = 1.0
            force_directed_attraction = 0.5
            force_directed_damping = 0.9
        "#;

        let default_path = config_path.join("default.toml");
        let mut file = fs::File::create(&default_path)?;
        file.write_all(default_config.as_bytes())?;

        std::env::set_var("CONFIG_DIR", config_path.to_str().unwrap());

        let settings = Settings::new()?;

        assert_eq!(settings.server.host, "127.0.0.1");
        assert_eq!(settings.server.port, 8080);
        assert_eq!(settings.github.owner, "default_owner");
        assert_eq!(settings.ragflow.api_url, "http://default.example.com");
        assert_eq!(settings.openai.model, "gpt-4");
        assert_eq!(settings.perplexity.model, "mixtral-8x7b");
        assert_eq!(settings.visualization.force_directed_iterations, 100);

        Ok(())
    }
}
