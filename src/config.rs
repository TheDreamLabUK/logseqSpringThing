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
    pub token: String,
    
    #[serde(default = "default_owner")]
    pub owner: String,
    
    #[serde(default = "default_repo")]
    pub repo: String,
    
    #[serde(default = "default_path")]
    pub base_path: String,
    
    #[serde(default = "default_version")]
    pub version: String,
    
    #[serde(default = "default_rate_limit")]
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

        // Try to convert it into our Settings type
        let mut settings: Settings = config.try_deserialize()?;
        
        // Override GitHub settings with environment variables if they exist
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            debug!("Found GITHUB_TOKEN in environment");
            settings.github.token = token;
        }
        if let Ok(owner) = std::env::var("GITHUB_OWNER") {
            debug!("Found GITHUB_OWNER in environment: {}", owner);
            settings.github.owner = owner;
        }
        if let Ok(repo) = std::env::var("GITHUB_REPO") {
            debug!("Found GITHUB_REPO in environment: {}", repo);
            settings.github.repo = repo;
        }
        if let Ok(path) = std::env::var("GITHUB_PATH") {
            debug!("Found GITHUB_PATH in environment: {}", path);
            settings.github.base_path = path;
        }
        if let Ok(version) = std::env::var("GITHUB_VERSION") {
            debug!("Found GITHUB_VERSION in environment: {}", version);
            settings.github.version = version;
        }
        
        // Log final non-sensitive settings
        debug!("GitHub settings loaded: owner={}, repo={}, base_path={}, version={}", 
            settings.github.owner,
            settings.github.repo,
            settings.github.base_path,
            settings.github.version
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

        let mut settings: Settings = config.try_deserialize()?;
        
        // Override GitHub settings with environment variables if they exist
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            settings.github.token = token;
        }
        if let Ok(owner) = std::env::var("GITHUB_OWNER") {
            settings.github.owner = owner;
        }
        if let Ok(repo) = std::env::var("GITHUB_REPO") {
            settings.github.repo = repo;
        }
        if let Ok(path) = std::env::var("GITHUB_PATH") {
            settings.github.base_path = path;
        }
        if let Ok(version) = std::env::var("GITHUB_VERSION") {
            settings.github.version = version;
        }

        Ok(settings)
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
