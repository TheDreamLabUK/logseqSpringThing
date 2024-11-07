use config::{Config, ConfigError, File};
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;
use std::env;

/// Converts a color value to proper CSS hex format
fn normalize_color(value: String) -> String {
    if value.starts_with('#') {
        value
    } else if value.starts_with("0x") {
        format!("#{}", &value[2..])
    } else {
        format!("#{}", value)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    #[serde(default = "default_prompt")]
    pub prompt: String,
    #[serde(skip_deserializing)]
    pub topics: Vec<String>,
    pub github: GithubSettings,
    pub ragflow: RagFlowSettings,
    pub perplexity: PerplexitySettings,
    pub openai: OpenAISettings,
    #[serde(default = "default_settings")]
    pub default: DefaultSettings,
    #[serde(default = "default_visualization")]
    pub visualization: VisualizationSettings,
    #[serde(default = "default_bloom")]
    pub bloom: BloomSettings,
    #[serde(default = "default_fisheye")]
    pub fisheye: FisheyeSettings,
}

fn default_prompt() -> String {
    "Your default prompt here".to_string()
}

fn default_settings() -> DefaultSettings {
    DefaultSettings {
        max_concurrent_requests: 5,
        max_retries: 3,
        retry_delay: 5,
        api_client_timeout: 30,
    }
}

fn default_visualization() -> VisualizationSettings {
    VisualizationSettings {
        node_color: "#1A0B31".to_string(),
        edge_color: "#FF0000".to_string(),
        hologram_color: "#FFD700".to_string(),
        node_size_scaling_factor: 5,
        hologram_scale: 5,
        hologram_opacity: 0.1,
        edge_opacity: 0.3,
        label_font_size: 36,
        fog_density: 0.002,
        force_directed_iterations: 100,
        force_directed_spring: 0.1,
        force_directed_repulsion: 1000.0,
        force_directed_attraction: 0.01,
        force_directed_damping: 0.8,
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

fn default_fisheye() -> FisheyeSettings {
    FisheyeSettings {
        enabled: false,
        strength: 0.5,
        focus_point: [0.0, 0.0, 0.0],
        radius: 100.0,
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GithubSettings {
    pub github_access_token: String,
    pub github_owner: String,
    pub github_repo: String,
    pub github_directory: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagFlowSettings {
    pub ragflow_api_key: String,
    pub ragflow_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    pub openai_api_key: String,
    pub openai_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerplexitySettings {
    pub perplexity_api_key: String,
    pub perplexity_model: String,
    pub perplexity_api_url: String,
    pub perplexity_max_tokens: u32,
    pub perplexity_temperature: f32,
    pub perplexity_top_p: f32,
    pub perplexity_presence_penalty: f32,
    pub perplexity_frequency_penalty: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DefaultSettings {
    pub max_concurrent_requests: u32,
    pub max_retries: u32,
    pub retry_delay: u32,
    pub api_client_timeout: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
    pub force_directed_spring: f32,
    pub force_directed_repulsion: f32,
    pub force_directed_attraction: f32,
    pub force_directed_damping: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct FisheyeSettings {
    pub enabled: bool,
    pub strength: f32,
    pub focus_point: [f32; 3],
    pub radius: f32,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        if !std::env::var("DOCKER").is_ok() {
            match dotenv() {
                Ok(_) => log::debug!("Successfully loaded .env file"),
                Err(e) => log::warn!("Failed to load .env file: {}", e),
            }
        }

        let run_mode = std::env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        log::debug!("Loading configuration for mode: {}", run_mode);

        let mut builder = Config::builder();

        // Add settings.toml
        builder = builder.add_source(File::with_name("settings.toml").required(false));

        // Environment variable overrides
        if let Ok(token) = env::var("GITHUB_ACCESS_TOKEN") {
            builder = builder.set_override("github.github_access_token", token)?;
        }
        if let Ok(owner) = env::var("GITHUB_OWNER") {
            builder = builder.set_override("github.github_owner", owner)?;
        }
        if let Ok(repo) = env::var("GITHUB_REPO") {
            builder = builder.set_override("github.github_repo", repo)?;
        }
        if let Ok(directory) = env::var("GITHUB_DIRECTORY") {
            builder = builder.set_override("github.github_directory", directory)?;
        }

        if let Ok(api_key) = env::var("RAGFLOW_API_KEY") {
            builder = builder.set_override("ragflow.ragflow_api_key", api_key)?;
        }
        if let Ok(base_url) = env::var("RAGFLOW_BASE_URL") {
            builder = builder.set_override("ragflow.ragflow_base_url", base_url)?;
        }

        if let Ok(api_key) = env::var("PERPLEXITY_API_KEY") {
            builder = builder.set_override("perplexity.perplexity_api_key", api_key)?;
        }
        if let Ok(model) = env::var("PERPLEXITY_MODEL") {
            builder = builder.set_override("perplexity.perplexity_model", model)?;
        }
        if let Ok(api_url) = env::var("PERPLEXITY_API_URL") {
            builder = builder.set_override("perplexity.perplexity_api_url", api_url)?;
        }
        if let Ok(max_tokens) = env::var("PERPLEXITY_MAX_TOKENS") {
            builder = builder.set_override("perplexity.perplexity_max_tokens", max_tokens)?;
        }
        if let Ok(temperature) = env::var("PERPLEXITY_TEMPERATURE") {
            builder = builder.set_override("perplexity.perplexity_temperature", temperature)?;
        }
        if let Ok(top_p) = env::var("PERPLEXITY_TOP_P") {
            builder = builder.set_override("perplexity.perplexity_top_p", top_p)?;
        }
        if let Ok(presence_penalty) = env::var("PERPLEXITY_PRESENCE_PENALTY") {
            builder = builder.set_override("perplexity.perplexity_presence_penalty", presence_penalty)?;
        }
        if let Ok(frequency_penalty) = env::var("PERPLEXITY_FREQUENCY_PENALTY") {
            builder = builder.set_override("perplexity.perplexity_frequency_penalty", frequency_penalty)?;
        }

        if let Ok(api_key) = env::var("OPENAI_API_KEY") {
            builder = builder.set_override("openai.openai_api_key", api_key)?;
        }
        if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
            builder = builder.set_override("openai.openai_base_url", base_url)?;
        }

        // Default settings
        if let Ok(max_requests) = env::var("MAX_CONCURRENT_REQUESTS") {
            builder = builder.set_override("default.max_concurrent_requests", max_requests)?;
        }
        if let Ok(max_retries) = env::var("MAX_RETRIES") {
            builder = builder.set_override("default.max_retries", max_retries)?;
        }
        if let Ok(retry_delay) = env::var("RETRY_DELAY") {
            builder = builder.set_override("default.retry_delay", retry_delay)?;
        }
        if let Ok(timeout) = env::var("API_CLIENT_TIMEOUT") {
            builder = builder.set_override("default.api_client_timeout", timeout)?;
        }

        // Visualization settings
        if let Ok(value) = env::var("NODE_COLOR") {
            builder = builder.set_override("visualization.node_color", normalize_color(value))?;
        }
        if let Ok(value) = env::var("EDGE_COLOR") {
            builder = builder.set_override("visualization.edge_color", normalize_color(value))?;
        }
        if let Ok(value) = env::var("HOLOGRAM_COLOR") {
            builder = builder.set_override("visualization.hologram_color", normalize_color(value))?;
        }
        if let Ok(value) = env::var("NODE_SIZE_SCALING_FACTOR") {
            builder = builder.set_override("visualization.node_size_scaling_factor", value)?;
        }
        if let Ok(value) = env::var("HOLOGRAM_SCALE") {
            builder = builder.set_override("visualization.hologram_scale", value)?;
        }
        if let Ok(value) = env::var("HOLOGRAM_OPACITY") {
            builder = builder.set_override("visualization.hologram_opacity", value)?;
        }
        if let Ok(value) = env::var("EDGE_OPACITY") {
            builder = builder.set_override("visualization.edge_opacity", value)?;
        }
        if let Ok(value) = env::var("LABEL_FONT_SIZE") {
            builder = builder.set_override("visualization.label_font_size", value)?;
        }
        if let Ok(value) = env::var("FOG_DENSITY") {
            builder = builder.set_override("visualization.fog_density", value)?;
        }
        if let Ok(value) = env::var("FORCE_DIRECTED_ITERATIONS") {
            builder = builder.set_override("visualization.force_directed_iterations", value)?;
        }
        if let Ok(value) = env::var("FORCE_DIRECTED_SPRING") {
            builder = builder.set_override("visualization.force_directed_spring", value)?;
        }
        if let Ok(value) = env::var("FORCE_DIRECTED_REPULSION") {
            builder = builder.set_override("visualization.force_directed_repulsion", value)?;
        }
        if let Ok(value) = env::var("FORCE_DIRECTED_ATTRACTION") {
            builder = builder.set_override("visualization.force_directed_attraction", value)?;
        }
        if let Ok(value) = env::var("FORCE_DIRECTED_DAMPING") {
            builder = builder.set_override("visualization.force_directed_damping", value)?;
        }

        // Bloom settings
        if let Ok(value) = env::var("NODE_BLOOM_STRENGTH") {
            builder = builder.set_override("bloom.node_bloom_strength", value)?;
        }
        if let Ok(value) = env::var("NODE_BLOOM_RADIUS") {
            builder = builder.set_override("bloom.node_bloom_radius", value)?;
        }
        if let Ok(value) = env::var("NODE_BLOOM_THRESHOLD") {
            builder = builder.set_override("bloom.node_bloom_threshold", value)?;
        }
        if let Ok(value) = env::var("EDGE_BLOOM_STRENGTH") {
            builder = builder.set_override("bloom.edge_bloom_strength", value)?;
        }
        if let Ok(value) = env::var("EDGE_BLOOM_RADIUS") {
            builder = builder.set_override("bloom.edge_bloom_radius", value)?;
        }
        if let Ok(value) = env::var("EDGE_BLOOM_THRESHOLD") {
            builder = builder.set_override("bloom.edge_bloom_threshold", value)?;
        }
        if let Ok(value) = env::var("ENVIRONMENT_BLOOM_STRENGTH") {
            builder = builder.set_override("bloom.environment_bloom_strength", value)?;
        }
        if let Ok(value) = env::var("ENVIRONMENT_BLOOM_RADIUS") {
            builder = builder.set_override("bloom.environment_bloom_radius", value)?;
        }
        if let Ok(value) = env::var("ENVIRONMENT_BLOOM_THRESHOLD") {
            builder = builder.set_override("bloom.environment_bloom_threshold", value)?;
        }

        // Fisheye settings
        if let Ok(value) = env::var("FISHEYE_ENABLED") {
            builder = builder.set_override("fisheye.enabled", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_STRENGTH") {
            builder = builder.set_override("fisheye.strength", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_RADIUS") {
            builder = builder.set_override("fisheye.radius", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_FOCUS_X") {
            builder = builder.set_override("fisheye.focus_point[0]", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_FOCUS_Y") {
            builder = builder.set_override("fisheye.focus_point[1]", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_FOCUS_Z") {
            builder = builder.set_override("fisheye.focus_point[2]", value)?;
        }

        let config = builder.build()?;

        match config.try_deserialize::<Settings>() {
            Ok(mut settings) => {
                settings.topics = load_topics_from_markdown();
                Ok(settings)
            }
            Err(e) => {
                log::error!("Failed to deserialize settings: {}", e);
                Err(e)
            }
        }
    }
}

fn load_topics_from_markdown() -> Vec<String> {
    let markdown_dir = Path::new("/app/data/markdown");
    if !markdown_dir.exists() {
        return vec!["default_topic".to_string()];
    }

    match fs::read_dir(markdown_dir) {
        Ok(entries) => {
            let mut topics: Vec<String> = entries
                .filter_map(|entry| {
                    entry.ok().and_then(|e| {
                        let path = e.path();
                        if let Some(ext) = path.extension() {
                            if ext == "md" {
                                path.file_stem()
                                    .and_then(|s| s.to_str())
                                    .map(|s| s.to_string())
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
                vec!["default_topic".to_string()]
            } else {
                topics.sort();
                topics
            }
        }
        Err(_) => vec!["default_topic".to_string()],
    }
}

impl fmt::Display for GithubSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GithubSettings {{ access_token: [REDACTED], owner: {}, repo: {}, directory: {} }}", 
               self.github_owner, self.github_repo, self.github_directory)
    }
}

impl fmt::Display for RagFlowSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RagFlowSettings {{ base_url: {}, api_key: [REDACTED] }}", 
               self.ragflow_base_url)
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
        write!(f, "PerplexitySettings {{ api_url: {}, api_key: [REDACTED], model: {} }}", 
               self.perplexity_api_url, self.perplexity_model)
    }
}

impl fmt::Display for DefaultSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DefaultSettings {{ max_concurrent_requests: {}, max_retries: {}, retry_delay: {}, api_client_timeout: {} }}", 
               self.max_concurrent_requests, self.max_retries, self.retry_delay, self.api_client_timeout)
    }
}

impl fmt::Display for VisualizationSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VisualizationSettings {{ node_color: {}, edge_color: {}, iterations: {}, repulsion: {}, attraction: {} }}", 
               self.node_color, self.edge_color, self.force_directed_iterations, 
               self.force_directed_repulsion, self.force_directed_attraction)
    }
}

impl fmt::Display for BloomSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomSettings {{ node_strength: {}, edge_strength: {}, environment_strength: {} }}", 
               self.node_bloom_strength, self.edge_bloom_strength, self.environment_bloom_strength)
    }
}

impl fmt::Display for FisheyeSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FisheyeSettings {{ enabled: {}, strength: {}, radius: {} }}", 
               self.enabled, self.strength, self.radius)
    }
}
