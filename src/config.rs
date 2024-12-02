use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment, File};
use log::debug;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub debug_mode: bool,
    pub debug: DebugSettings,
    pub prompt: String,
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
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    pub enable_websocket_debug: bool,
    pub enable_data_debug: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GitHubSettings {
    #[serde(default = "default_token")]
    #[serde(rename(deserialize = "GITHUB_TOKEN"))]
    pub token: String,
    
    #[serde(default = "default_owner")]
    #[serde(rename(deserialize = "GITHUB_OWNER"))]
    pub owner: String,
    
    #[serde(default = "default_repo")]
    #[serde(rename(deserialize = "GITHUB_REPO"))]
    pub repo: String,
    
    #[serde(default = "default_path")]
    #[serde(rename(deserialize = "GITHUB_BASE_PATH"))]
    pub base_path: String,
    
    #[serde(default = "default_version")]
    #[serde(rename(deserialize = "GITHUB_VERSION"))]
    pub version: String,
    
    #[serde(default = "default_rate_limit")]
    #[serde(rename(deserialize = "GITHUB_RATE_LIMIT"))]
    pub rate_limit: bool,
}

fn default_token() -> String { "".to_string() }
fn default_owner() -> String { "".to_string() }
fn default_repo() -> String { "".to_string() }
fn default_path() -> String { "".to_string() }
fn default_version() -> String { "2022-11-28".to_string() }
fn default_rate_limit() -> bool { true }

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Loading settings from settings.toml");
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            // First load defaults from settings.toml
            .add_source(File::with_name("settings.toml"))
            // Then override with environment variables
            .add_source(
                Environment::default()
                    .separator("_")
                    .try_parsing(true)
            )
            .build()?;

        // Log GitHub settings for debugging
        debug!("Environment variables:");
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            debug!("GITHUB_TOKEN from env is set");
        }
        if let Ok(owner) = std::env::var("GITHUB_OWNER") {
            debug!("GITHUB_OWNER from env: {}", owner);
        }
        if let Ok(repo) = std::env::var("GITHUB_REPO") {
            debug!("GITHUB_REPO from env: {}", repo);
        }
        if let Ok(dir) = std::env::var("GITHUB_BASE_PATH") {
            debug!("GITHUB_BASE_PATH from env: {}", dir);
        }

        // Log config values
        debug!("Config values:");
        if let Ok(owner) = config.get_string("github.owner") {
            debug!("github.owner from config: {}", owner);
        }
        if let Ok(repo) = config.get_string("github.repo") {
            debug!("github.repo from config: {}", repo);
        }
        if let Ok(dir) = config.get_string("github.base_path") {
            debug!("github.base_path from config: {}", dir);
        }

        // Try to convert it into our Settings type
        let settings: Settings = config.try_deserialize()?;
        
        // Log final non-sensitive settings
        let settings_clone = settings.clone();
        debug!("GitHub settings loaded: owner={}, repo={}, base_path={}", 
            settings_clone.github.owner,
            settings_clone.github.repo,
            settings_clone.github.base_path
        );

        Ok(settings)
    }

    pub fn from_env() -> Result<Self, ConfigError> {
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            .add_source(
                Environment::default()
                    .separator("_")
                    .try_parsing(true)
            )
            .build()?;

        config.try_deserialize::<Settings>()
    }
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
    pub tunnel_id: String,
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
pub struct RagFlowSettings {
    pub api_key: String,
    pub base_url: String,
    pub timeout: u64,
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
    pub timeout: u64,
    pub rate_limit: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    pub api_key: String,
    pub base_url: String,
    pub timeout: u64,
    pub rate_limit: u32,
    pub model: String,
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

    // Node age-based colors
    pub node_color_new: String,
    pub node_color_recent: String,
    pub node_color_medium: String,
    pub node_color_old: String,
    pub node_age_max_days: u32,

    // Node type colors
    pub node_color_core: String,
    pub node_color_secondary: String,
    pub node_color_default: String,

    // Sizes and scales
    pub min_node_size: f32,
    pub max_node_size: f32,
    pub hologram_scale: f32,

    // Opacity settings
    pub hologram_opacity: f32,
    pub edge_opacity: f32,

    // Environment settings
    pub fog_density: f32,

    // Node material properties
    pub node_material_metalness: f32,
    pub node_material_roughness: f32,
    pub node_material_clearcoat: f32,
    pub node_material_clearcoat_roughness: f32,
    pub node_material_opacity: f32,
    pub node_emissive_min_intensity: f32,
    pub node_emissive_max_intensity: f32,

    // Label properties
    pub label_font_size: u32,
    pub label_font_family: String,
    pub label_padding: u32,
    pub label_vertical_offset: f32,
    pub label_close_offset: f32,
    pub label_background_color: String,
    pub label_text_color: String,
    pub label_info_text_color: String,
    pub label_xr_font_size: u32,

    // Edge properties
    pub edge_weight_normalization: f32,
    pub edge_min_width: f32,
    pub edge_max_width: f32,

    // Geometry properties
    pub geometry_min_segments: u32,
    pub geometry_max_segments: u32,
    pub geometry_segment_per_hyperlink: f32,

    // Interaction properties
    pub click_emissive_boost: f32,
    pub click_feedback_duration: u32,

    // Force-directed layout parameters
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
