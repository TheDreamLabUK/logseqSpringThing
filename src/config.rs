use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub prompt: String,
    pub github: GithubSettings,
    pub ragflow: RagflowSettings,
    pub perplexity: PerplexitySettings,
    pub openai: OpenAISettings,
    pub default: DefaultSettings,
    pub visualization: VisualizationSettings,
    pub bloom: BloomSettings,
    pub fisheye: FisheyeSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GithubSettings {
    pub github_access_token: String,
    pub github_owner: String,
    pub github_repo: String,
    pub github_directory: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagflowSettings {
    pub ragflow_api_key: String,
    pub ragflow_api_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerplexitySettings {
    pub perplexity_api_key: String,
    pub perplexity_model: String,
    pub perplexity_api_url: String,
    pub perplexity_max_tokens: usize,
    pub perplexity_temperature: f32,
    pub perplexity_top_p: f32,
    pub perplexity_presence_penalty: f32,
    pub perplexity_frequency_penalty: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    pub openai_api_key: String,
    pub openai_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DefaultSettings {
    pub max_concurrent_requests: usize,
    pub max_retries: usize,
    pub retry_delay: u64,
    pub api_client_timeout: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VisualizationSettings {
    pub node_color: String,
    pub edge_color: String,
    pub hologram_color: String,
    pub node_size_scaling_factor: f32,
    pub hologram_scale: f32,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FisheyeSettings {
    pub enabled: bool,
    pub strength: f32,
    pub focus_point: [f32; 3],
    pub radius: f32,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let mut builder = Config::builder()
            .add_source(File::with_name("settings"))
            .add_source(Environment::with_prefix("APP"));

        // Override settings with environment variables if they exist
        if let Ok(value) = env::var("GITHUB_ACCESS_TOKEN") {
            builder = builder.set_override("github.github_access_token", value)?;
        }
        if let Ok(value) = env::var("GITHUB_OWNER") {
            builder = builder.set_override("github.github_owner", value)?;
        }
        if let Ok(value) = env::var("GITHUB_REPO") {
            builder = builder.set_override("github.github_repo", value)?;
        }
        if let Ok(value) = env::var("GITHUB_DIRECTORY") {
            builder = builder.set_override("github.github_directory", value)?;
        }
        if let Ok(value) = env::var("RAGFLOW_API_KEY") {
            builder = builder.set_override("ragflow.ragflow_api_key", value)?;
        }
        if let Ok(value) = env::var("RAGFLOW_BASE_URL") {
            builder = builder.set_override("ragflow.ragflow_api_base_url", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_API_KEY") {
            builder = builder.set_override("perplexity.perplexity_api_key", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_MODEL") {
            builder = builder.set_override("perplexity.perplexity_model", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_API_URL") {
            builder = builder.set_override("perplexity.perplexity_api_url", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_MAX_TOKENS") {
            builder = builder.set_override("perplexity.perplexity_max_tokens", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_TEMPERATURE") {
            builder = builder.set_override("perplexity.perplexity_temperature", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_TOP_P") {
            builder = builder.set_override("perplexity.perplexity_top_p", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_PRESENCE_PENALTY") {
            builder = builder.set_override("perplexity.perplexity_presence_penalty", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_FREQUENCY_PENALTY") {
            builder = builder.set_override("perplexity.perplexity_frequency_penalty", value)?;
        }
        if let Ok(value) = env::var("OPENAI_API_KEY") {
            builder = builder.set_override("openai.openai_api_key", value)?;
        }
        if let Ok(value) = env::var("OPENAI_BASE_URL") {
            builder = builder.set_override("openai.openai_base_url", value)?;
        }
        if let Ok(value) = env::var("MAX_CONCURRENT_REQUESTS") {
            builder = builder.set_override("default.max_concurrent_requests", value)?;
        }
        if let Ok(value) = env::var("MAX_RETRIES") {
            builder = builder.set_override("default.max_retries", value)?;
        }
        if let Ok(value) = env::var("RETRY_DELAY") {
            builder = builder.set_override("default.retry_delay", value)?;
        }
        if let Ok(value) = env::var("API_CLIENT_TIMEOUT") {
            builder = builder.set_override("default.api_client_timeout", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR") {
            builder = builder.set_override("visualization.node_color", value)?;
        }
        if let Ok(value) = env::var("EDGE_COLOR") {
            builder = builder.set_override("visualization.edge_color", value)?;
        }
        if let Ok(value) = env::var("HOLOGRAM_COLOR") {
            builder = builder.set_override("visualization.hologram_color", value)?;
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
        if let Ok(value) = env::var("FISHEYE_ENABLED") {
            builder = builder.set_override("fisheye.enabled", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_STRENGTH") {
            builder = builder.set_override("fisheye.strength", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_RADIUS") {
            builder = builder.set_override("fisheye.radius", value)?;
        }

        builder.build()?.try_deserialize()
    }
}

impl fmt::Display for VisualizationSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VisualizationSettings {{ node_color: {}, edge_color: {}, iterations: {}, repulsion: {}, attraction: {} }}",
               self.node_color, self.edge_color, self.force_directed_iterations,
               self.force_directed_repulsion, self.force_directed_attraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_params_from_config() {
        let config = VisualizationSettings {
            node_color: "0x1A0B31".to_string(),
            edge_color: "0xff0000".to_string(),
            hologram_color: "0xFFD700".to_string(),
            node_size_scaling_factor: 1.0,
            hologram_scale: 5.0,
            hologram_opacity: 0.1,
            edge_opacity: 0.3,
            label_font_size: 16,
            fog_density: 0.002,
            force_directed_iterations: 100,
            force_directed_spring: 0.1,
            force_directed_repulsion: 1000.0,
            force_directed_attraction: 0.01,
            force_directed_damping: 0.8,
        };

        let params = crate::models::simulation_params::SimulationParams::from_config(&config);
        assert_eq!(params.iterations, 100);
        assert_eq!(params.repulsion_strength, 1000.0);
        assert_eq!(params.attraction_strength, 0.01);
        assert_eq!(params.damping, 0.8);
    }

    #[test]
    fn test_simulation_params_clamping() {
        let params = crate::models::simulation_params::SimulationParams::new(5, 50.0, 0.001, 0.05);
        assert_eq!(params.iterations, 10); // Clamped to min
        assert_eq!(params.repulsion_strength, 100.0); // Clamped to min
        assert_eq!(params.attraction_strength, 0.01); // Clamped to min
        assert_eq!(params.damping, 0.1); // Clamped to min

        let params = crate::models::simulation_params::SimulationParams::new(1000, 10000.0, 2.0, 1.0);
        assert_eq!(params.iterations, 500); // Clamped to max
        assert_eq!(params.repulsion_strength, 5000.0); // Clamped to max
        assert_eq!(params.attraction_strength, 1.0); // Clamped to max
        assert_eq!(params.damping, 0.9); // Clamped to max
    }

    #[test]
    fn test_simulation_params_builder() {
        let params = crate::models::simulation_params::SimulationParams::default()
            .with_iterations(200)
            .with_repulsion(2000.0)
            .with_attraction(0.05)
            .with_damping(0.7);

        assert_eq!(params.iterations, 200);
        assert_eq!(params.repulsion_strength, 2000.0);
        assert_eq!(params.attraction_strength, 0.05);
        assert_eq!(params.damping, 0.7);
    }
}
