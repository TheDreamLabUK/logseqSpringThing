use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::{env, fmt};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub debug_mode: bool,
    pub network: NetworkSettings,
    pub security: SecuritySettings,
    pub github: GitHubSettings,
    pub ragflow: RagFlowSettings,
    pub perplexity: PerplexitySettings,
    pub openai: OpenAISettings,
    pub default: DefaultSettings,
    pub visualization: VisualizationSettings,
    pub bloom: BloomSettings,
    pub fisheye: FisheyeSettings,
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkSettings {
    pub domain: String,
    pub port: u16,
    pub bind_address: String,
    pub enable_tls: bool,
    pub min_tls_version: String,
    pub enable_http2: bool,
    pub max_request_size: usize,
    pub enable_rate_limiting: bool,
    pub rate_limit_requests: u32,
    pub rate_limit_window: u32,
    pub tunnel_id: String,  // Added tunnel_id field
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SecuritySettings {
    pub enable_cors: bool,
    pub allowed_origins: Vec<String>,
    pub enable_csrf: bool,
    pub csrf_token_timeout: u32,
    pub session_timeout: u32,
    pub cookie_secure: bool,
    pub cookie_httponly: bool,
    pub cookie_samesite: String,
    pub enable_security_headers: bool,
    pub enable_request_validation: bool,
    pub enable_audit_logging: bool,
    pub audit_log_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GitHubSettings {
    pub access_token: String,
    pub owner: String,
    pub repo: String,
    pub directory: String,
    pub api_version: String,
    pub rate_limit_enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagFlowSettings {
    pub api_key: String,
    pub base_url: String,
    pub timeout: u32,
    pub max_retries: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerplexitySettings {
    pub api_key: String,
    pub model: String,
    pub api_url: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub timeout: u32,
    pub rate_limit: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    pub api_key: String,
    pub base_url: String,
    pub timeout: u32,
    pub rate_limit: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DefaultSettings {
    pub max_concurrent_requests: usize,
    pub max_retries: u32,
    pub retry_delay: u64,
    pub api_client_timeout: u64,
    pub max_payload_size: usize,
    pub enable_request_logging: bool,
    pub enable_metrics: bool,
    pub metrics_port: u16,
    pub log_level: String,
    pub log_format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VisualizationSettings {
    // Colors
    pub node_color: String,
    pub edge_color: String,
    pub hologram_color: String,
    pub node_color_new: String,
    pub node_color_recent: String,
    pub node_color_medium: String,
    pub node_color_old: String,
    pub node_color_core: String,
    pub node_color_secondary: String,
    pub node_color_default: String,

    // Physical dimensions
    pub min_node_size: f32,    // In meters (0.1m = 10cm)
    pub max_node_size: f32,    // In meters (0.3m = 30cm)
    pub hologram_scale: f32,
    pub hologram_opacity: f32,
    pub edge_opacity: f32,

    // Label settings
    pub label_font_size: u32,
    pub label_font_family: String,
    pub label_padding: u32,
    pub label_vertical_offset: f32,
    pub label_close_offset: f32,
    pub label_background_color: String,
    pub label_text_color: String,
    pub label_info_text_color: String,
    pub label_xr_font_size: u32,

    // Geometry settings
    pub geometry_min_segments: u32,
    pub geometry_max_segments: u32,
    pub geometry_segment_per_hyperlink: f32,

    // Material settings
    pub node_material_metalness: f32,
    pub node_material_roughness: f32,
    pub node_material_clearcoat: f32,
    pub node_material_clearcoat_roughness: f32,
    pub node_material_opacity: f32,
    pub node_emissive_min_intensity: f32,
    pub node_emissive_max_intensity: f32,

    // Interaction settings
    pub click_emissive_boost: f32,
    pub click_feedback_duration: u32,

    // Environment settings
    pub fog_density: f32,

    // Physics simulation parameters
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
    pub radius: f32,
    pub focus_x: f32,
    pub focus_y: f32,
    pub focus_z: f32,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let mut builder = Config::builder()
            .add_source(File::with_name("settings"))
            .add_source(Environment::with_prefix("APP"));

        // Network settings
        if let Ok(value) = env::var("DOMAIN") {
            builder = builder.set_override("network.domain", value)?;
        }
        if let Ok(value) = env::var("PORT") {
            builder = builder.set_override("network.port", value)?;
        }
        if let Ok(value) = env::var("BIND_ADDRESS") {
            builder = builder.set_override("network.bind_address", value)?;
        }
        if let Ok(value) = env::var("ENABLE_TLS") {
            builder = builder.set_override("network.enable_tls", value)?;
        }
        if let Ok(value) = env::var("MIN_TLS_VERSION") {
            builder = builder.set_override("network.min_tls_version", value)?;
        }
        if let Ok(value) = env::var("ENABLE_HTTP2") {
            builder = builder.set_override("network.enable_http2", value)?;
        }
        if let Ok(value) = env::var("MAX_REQUEST_SIZE") {
            builder = builder.set_override("network.max_request_size", value)?;
        }
        if let Ok(value) = env::var("ENABLE_RATE_LIMITING") {
            builder = builder.set_override("network.enable_rate_limiting", value)?;
        }
        if let Ok(value) = env::var("RATE_LIMIT_REQUESTS") {
            builder = builder.set_override("network.rate_limit_requests", value)?;
        }
        if let Ok(value) = env::var("RATE_LIMIT_WINDOW") {
            builder = builder.set_override("network.rate_limit_window", value)?;
        }
        if let Ok(value) = env::var("TUNNEL_ID") {
            builder = builder.set_override("network.tunnel_id", value)?;
        }

        // Security settings
        if let Ok(value) = env::var("ENABLE_CORS") {
            builder = builder.set_override("security.enable_cors", value)?;
        }
        if let Ok(value) = env::var("ENABLE_CSRF") {
            builder = builder.set_override("security.enable_csrf", value)?;
        }
        if let Ok(value) = env::var("CSRF_TOKEN_TIMEOUT") {
            builder = builder.set_override("security.csrf_token_timeout", value)?;
        }
        if let Ok(value) = env::var("SESSION_TIMEOUT") {
            builder = builder.set_override("security.session_timeout", value)?;
        }
        if let Ok(value) = env::var("COOKIE_SECURE") {
            builder = builder.set_override("security.cookie_secure", value)?;
        }
        if let Ok(value) = env::var("COOKIE_HTTPONLY") {
            builder = builder.set_override("security.cookie_httponly", value)?;
        }
        if let Ok(value) = env::var("COOKIE_SAMESITE") {
            builder = builder.set_override("security.cookie_samesite", value)?;
        }
        if let Ok(value) = env::var("ENABLE_SECURITY_HEADERS") {
            builder = builder.set_override("security.enable_security_headers", value)?;
        }
        if let Ok(value) = env::var("ENABLE_REQUEST_VALIDATION") {
            builder = builder.set_override("security.enable_request_validation", value)?;
        }
        if let Ok(value) = env::var("ENABLE_AUDIT_LOGGING") {
            builder = builder.set_override("security.enable_audit_logging", value)?;
        }
        if let Ok(value) = env::var("AUDIT_LOG_PATH") {
            builder = builder.set_override("security.audit_log_path", value)?;
        }

        // GitHub settings
        if let Ok(value) = env::var("GITHUB_ACCESS_TOKEN") {
            builder = builder.set_override("github.access_token", value)?;
        }
        if let Ok(value) = env::var("GITHUB_OWNER") {
            builder = builder.set_override("github.owner", value)?;
        }
        if let Ok(value) = env::var("GITHUB_REPO") {
            builder = builder.set_override("github.repo", value)?;
        }
        if let Ok(value) = env::var("GITHUB_DIRECTORY") {
            builder = builder.set_override("github.directory", value)?;
        }
        if let Ok(value) = env::var("GITHUB_API_VERSION") {
            builder = builder.set_override("github.api_version", value)?;
        }
        if let Ok(value) = env::var("GITHUB_RATE_LIMIT_ENABLED") {
            builder = builder.set_override("github.rate_limit_enabled", value)?;
        }

        // RAGFlow settings
        if let Ok(value) = env::var("RAGFLOW_API_KEY") {
            builder = builder.set_override("ragflow.api_key", value)?;
        }
        if let Ok(value) = env::var("RAGFLOW_BASE_URL") {
            builder = builder.set_override("ragflow.base_url", value)?;
        }
        if let Ok(value) = env::var("RAGFLOW_TIMEOUT") {
            builder = builder.set_override("ragflow.timeout", value)?;
        }
        if let Ok(value) = env::var("RAGFLOW_MAX_RETRIES") {
            builder = builder.set_override("ragflow.max_retries", value)?;
        }

        // Perplexity settings
        if let Ok(value) = env::var("PERPLEXITY_API_KEY") {
            builder = builder.set_override("perplexity.api_key", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_MODEL") {
            builder = builder.set_override("perplexity.model", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_API_URL") {
            builder = builder.set_override("perplexity.api_url", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_MAX_TOKENS") {
            builder = builder.set_override("perplexity.max_tokens", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_TEMPERATURE") {
            builder = builder.set_override("perplexity.temperature", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_TOP_P") {
            builder = builder.set_override("perplexity.top_p", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_PRESENCE_PENALTY") {
            builder = builder.set_override("perplexity.presence_penalty", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_FREQUENCY_PENALTY") {
            builder = builder.set_override("perplexity.frequency_penalty", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_TIMEOUT") {
            builder = builder.set_override("perplexity.timeout", value)?;
        }
        if let Ok(value) = env::var("PERPLEXITY_RATE_LIMIT") {
            builder = builder.set_override("perplexity.rate_limit", value)?;
        }

        // OpenAI settings
        if let Ok(value) = env::var("OPENAI_API_KEY") {
            builder = builder.set_override("openai.api_key", value)?;
        }
        if let Ok(value) = env::var("OPENAI_BASE_URL") {
            builder = builder.set_override("openai.base_url", value)?;
        }
        if let Ok(value) = env::var("OPENAI_TIMEOUT") {
            builder = builder.set_override("openai.timeout", value)?;
        }
        if let Ok(value) = env::var("OPENAI_RATE_LIMIT") {
            builder = builder.set_override("openai.rate_limit", value)?;
        }

        // Default settings
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
        if let Ok(value) = env::var("MAX_PAYLOAD_SIZE") {
            builder = builder.set_override("default.max_payload_size", value)?;
        }
        if let Ok(value) = env::var("ENABLE_REQUEST_LOGGING") {
            builder = builder.set_override("default.enable_request_logging", value)?;
        }
        if let Ok(value) = env::var("ENABLE_METRICS") {
            builder = builder.set_override("default.enable_metrics", value)?;
        }
        if let Ok(value) = env::var("METRICS_PORT") {
            builder = builder.set_override("default.metrics_port", value)?;
        }
        if let Ok(value) = env::var("LOG_LEVEL") {
            builder = builder.set_override("default.log_level", value)?;
        }
        if let Ok(value) = env::var("LOG_FORMAT") {
            builder = builder.set_override("default.log_format", value)?;
        }

        // Visualization settings
        if let Ok(value) = env::var("NODE_COLOR") {
            builder = builder.set_override("visualization.node_color", value)?;
        }
        if let Ok(value) = env::var("EDGE_COLOR") {
            builder = builder.set_override("visualization.edge_color", value)?;
        }
        if let Ok(value) = env::var("HOLOGRAM_COLOR") {
            builder = builder.set_override("visualization.hologram_color", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_NEW") {
            builder = builder.set_override("visualization.node_color_new", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_RECENT") {
            builder = builder.set_override("visualization.node_color_recent", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_MEDIUM") {
            builder = builder.set_override("visualization.node_color_medium", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_OLD") {
            builder = builder.set_override("visualization.node_color_old", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_CORE") {
            builder = builder.set_override("visualization.node_color_core", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_SECONDARY") {
            builder = builder.set_override("visualization.node_color_secondary", value)?;
        }
        if let Ok(value) = env::var("NODE_COLOR_DEFAULT") {
            builder = builder.set_override("visualization.node_color_default", value)?;
        }

        // Visualization settings - Dimensions
        if let Ok(value) = env::var("MIN_NODE_SIZE") {
            builder = builder.set_override("visualization.min_node_size", value)?;
        }
        if let Ok(value) = env::var("MAX_NODE_SIZE") {
            builder = builder.set_override("visualization.max_node_size", value)?;
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

        // Visualization settings - Labels
        if let Ok(value) = env::var("LABEL_FONT_SIZE") {
            builder = builder.set_override("visualization.label_font_size", value)?;
        }
        if let Ok(value) = env::var("LABEL_FONT_FAMILY") {
            builder = builder.set_override("visualization.label_font_family", value)?;
        }
        if let Ok(value) = env::var("LABEL_PADDING") {
            builder = builder.set_override("visualization.label_padding", value)?;
        }
        if let Ok(value) = env::var("LABEL_VERTICAL_OFFSET") {
            builder = builder.set_override("visualization.label_vertical_offset", value)?;
        }
        if let Ok(value) = env::var("LABEL_CLOSE_OFFSET") {
            builder = builder.set_override("visualization.label_close_offset", value)?;
        }
        if let Ok(value) = env::var("LABEL_BACKGROUND_COLOR") {
            builder = builder.set_override("visualization.label_background_color", value)?;
        }
        if let Ok(value) = env::var("LABEL_TEXT_COLOR") {
            builder = builder.set_override("visualization.label_text_color", value)?;
        }
        if let Ok(value) = env::var("LABEL_INFO_TEXT_COLOR") {
            builder = builder.set_override("visualization.label_info_text_color", value)?;
        }
        if let Ok(value) = env::var("LABEL_XR_FONT_SIZE") {
            builder = builder.set_override("visualization.label_xr_font_size", value)?;
        }

        // Visualization settings - Geometry
        if let Ok(value) = env::var("GEOMETRY_MIN_SEGMENTS") {
            builder = builder.set_override("visualization.geometry_min_segments", value)?;
        }
        if let Ok(value) = env::var("GEOMETRY_MAX_SEGMENTS") {
            builder = builder.set_override("visualization.geometry_max_segments", value)?;
        }
        if let Ok(value) = env::var("GEOMETRY_SEGMENT_PER_HYPERLINK") {
            builder = builder.set_override("visualization.geometry_segment_per_hyperlink", value)?;
        }

        // Visualization settings - Material
        if let Ok(value) = env::var("NODE_MATERIAL_METALNESS") {
            builder = builder.set_override("visualization.node_material_metalness", value)?;
        }
        if let Ok(value) = env::var("NODE_MATERIAL_ROUGHNESS") {
            builder = builder.set_override("visualization.node_material_roughness", value)?;
        }
        if let Ok(value) = env::var("NODE_MATERIAL_CLEARCOAT") {
            builder = builder.set_override("visualization.node_material_clearcoat", value)?;
        }
        if let Ok(value) = env::var("NODE_MATERIAL_CLEARCOAT_ROUGHNESS") {
            builder = builder.set_override("visualization.node_material_clearcoat_roughness", value)?;
        }
        if let Ok(value) = env::var("NODE_MATERIAL_OPACITY") {
            builder = builder.set_override("visualization.node_material_opacity", value)?;
        }
        if let Ok(value) = env::var("NODE_EMISSIVE_MIN_INTENSITY") {
            builder = builder.set_override("visualization.node_emissive_min_intensity", value)?;
        }
        if let Ok(value) = env::var("NODE_EMISSIVE_MAX_INTENSITY") {
            builder = builder.set_override("visualization.node_emissive_max_intensity", value)?;
        }

        // Visualization settings - Interaction
        if let Ok(value) = env::var("CLICK_EMISSIVE_BOOST") {
            builder = builder.set_override("visualization.click_emissive_boost", value)?;
        }
        if let Ok(value) = env::var("CLICK_FEEDBACK_DURATION") {
            builder = builder.set_override("visualization.click_feedback_duration", value)?;
        }
        if let Ok(value) = env::var("FISHEYE_FOCUS_Z") {
            builder = builder.set_override("fisheye.focus_z", value)?;
        }

        builder.build()?.try_deserialize()
    }
}

impl fmt::Display for VisualizationSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VisualizationSettings {{ node_color: {}, edge_color: {}, min_node_size: {}m, max_node_size: {}m, iterations: {}, repulsion: {}, attraction: {} }}",
            self.node_color,
            self.edge_color,
            self.min_node_size,
            self.max_node_size,
            self.force_directed_iterations,
            self.force_directed_repulsion,
            self.force_directed_attraction
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::simulation_params::{SimulationParams, SimulationPhase};

    #[test]
    fn test_simulation_params_from_config() {
        let config = VisualizationSettings {
            // Colors
            node_color: "0x1A0B31".to_string(),
            edge_color: "0xff0000".to_string(),
            hologram_color: "0xFFD700".to_string(),
            node_color_new: "0x00ff88".to_string(),
            node_color_recent: "0x4444ff".to_string(),
            node_color_medium: "0xffaa00".to_string(),
            node_color_old: "0xff4444".to_string(),
            node_color_core: "0xffa500".to_string(),
            node_color_secondary: "0x00ffff".to_string(),
            node_color_default: "0x00ff00".to_string(),

            // Physical dimensions
            min_node_size: 0.1,
            max_node_size: 0.3,
            hologram_scale: 5.0,
            hologram_opacity: 0.1,
            edge_opacity: 0.3,

            // Label settings
            label_font_size: 36,
            label_font_family: "Arial".to_string(),
            label_padding: 20,
            label_vertical_offset: 2.0,
            label_close_offset: 0.2,
            label_background_color: "rgba(0, 0, 0, 0.8)".to_string(),
            label_text_color: "white".to_string(),
            label_info_text_color: "lightgray".to_string(),
            label_xr_font_size: 24,

            // Geometry settings
            geometry_min_segments: 16,
            geometry_max_segments: 32,
            geometry_segment_per_hyperlink: 0.5,

            // Material settings
            node_material_metalness: 0.2,
            node_material_roughness: 0.2,
            node_material_clearcoat: 0.3,
            node_material_clearcoat_roughness: 0.2,
            node_material_opacity: 0.9,
            node_emissive_min_intensity: 0.3,
            node_emissive_max_intensity: 1.0,

            // Interaction settings
            click_emissive_boost: 2.0,
            click_feedback_duration: 200,

            // Environment settings
            fog_density: 0.002,

            // Physics simulation parameters
            force_directed_iterations: 100,
            force_directed_spring: 0.1,
            force_directed_repulsion: 1000.0,
            force_directed_attraction: 0.01,
            force_directed_damping: 0.8,
        };

        let params = SimulationParams::from_config(&config, SimulationPhase::Initial);
        assert_eq!(params.iterations, 100);
        assert_eq!(params.spring_strength, 0.1);
        assert_eq!(params.repulsion_strength, 1000.0);
        assert_eq!(params.attraction_strength, 0.01);
        assert_eq!(params.damping, 0.8);
        assert!(params.is_initial_layout);
    }

    #[test]
    fn test_simulation_params_clamping() {
        let params = SimulationParams::new(
            5,
            0.0001, // spring_strength
            0.5,    // repulsion_strength
            0.0001, // attraction_strength
            0.3,    // damping
            false   // is_initial
        );
        assert_eq!(params.iterations, 5);
        assert_eq!(params.spring_strength, 0.001); // Clamped to min
        assert_eq!(params.repulsion_strength, 1.0); // Clamped to min
        assert_eq!(params.attraction_strength, 0.001); // Clamped to min
        assert_eq!(params.damping, 0.5); // Clamped to min
        assert!(!params.is_initial_layout);

        let params = SimulationParams::new(
            1000,
            2.0,     // spring_strength
            20000.0, // repulsion_strength
            2.0,     // attraction_strength
            1.0,     // damping
            true     // is_initial
        );
        assert_eq!(params.iterations, 500); // Clamped to max
        assert_eq!(params.spring_strength, 1.0); // Clamped to max
        assert_eq!(params.repulsion_strength, 10000.0); // Clamped to max
        assert_eq!(params.attraction_strength, 1.0); // Clamped to max
        assert_eq!(params.damping, 0.95); // Clamped to max
        assert!(params.is_initial_layout);
    }

    #[test]
    fn test_simulation_params_builder() {
        let params = SimulationParams::default()
            .with_iterations(200)
            .with_spring_strength(0.5)
            .with_repulsion_strength(5000.0)
            .with_attraction_strength(0.05)
            .with_damping(0.7)
            .with_time_step(0.8);

        assert_eq!(params.iterations, 10); // Clamped to interactive max since not initial
        assert_eq!(params.spring_strength, 0.5);
        assert_eq!(params.repulsion_strength, 5000.0);
        assert_eq!(params.attraction_strength, 0.05);
        assert_eq!(params.damping, 0.7);
        assert_eq!(params.time_step, 0.8);
    }
}
