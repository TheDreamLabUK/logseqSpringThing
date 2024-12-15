use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment, File};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub default: DefaultSettings,
    pub network: NetworkSettings,
    pub security: SecuritySettings,
    pub rendering: RenderingSettings,
    pub ar: ARSettings,
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub physics: PhysicsSettings,
    pub bloom: BloomSettings,
    pub labels: LabelSettings,
    pub websocket: WebSocketSettings,
    pub server_debug: DebugSettings,
    pub client_debug: DebugSettings,
    pub github: GitHubSettings,
    pub openai: OpenAISettings,
    pub perplexity: PerplexitySettings,
    pub ragflow: RagFlowSettings,
    pub animations: AnimationSettings,
    pub audio: AudioSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DefaultSettings {
    pub api_client_timeout: u64,
    pub enable_metrics: bool,
    pub enable_request_logging: bool,
    pub log_format: String,
    pub log_level: String,
    pub max_concurrent_requests: usize,
    pub max_payload_size: usize,
    pub max_retries: u32,
    pub metrics_port: u16,
    pub retry_delay: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkSettings {
    pub bind_address: String,
    pub domain: String,
    pub enable_http2: bool,
    pub enable_rate_limiting: bool,
    pub enable_tls: bool,
    pub max_request_size: usize,
    pub min_tls_version: String,
    pub port: u16,
    pub rate_limit_requests: u32,
    pub rate_limit_window: u32,
    pub tunnel_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SecuritySettings {
    pub allowed_origins: Vec<String>,
    pub audit_log_path: String,
    pub cookie_httponly: bool,
    pub cookie_samesite: String,
    pub cookie_secure: bool,
    pub csrf_token_timeout: u32,
    pub enable_audit_logging: bool,
    pub enable_request_validation: bool,
    pub session_timeout: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ARSettings {
    pub enable_hand_tracking: bool,
    pub enable_haptics: bool,
    pub enable_plane_detection: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeSettings {
    pub base_color: String,
    pub base_size: f32,
    pub size_range: Vec<f32>,
    pub size_by_connections: bool,
    pub enable_instancing: bool,
    pub material_type: String,
    pub opacity: f32,
    pub roughness: f32,
    pub metalness: f32,
    pub clearcoat: f32,
    pub highlight_color: String,
    pub highlight_duration: u32,
    pub enable_hover_effect: bool,
    pub hover_scale: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EdgeSettings {
    pub base_width: f32,
    pub width_range: Vec<f32>,
    pub color: String,
    pub opacity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    pub attraction_strength: f32,
    pub bounds_size: f32,
    pub collision_radius: f32,
    pub damping: f32,
    pub enable_bounds: bool,
    pub enabled: bool,
    pub iterations: u32,
    pub max_velocity: f32,
    pub repulsion_strength: f32,
    pub spring_strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BloomSettings {
    pub enabled: bool,
    pub node_bloom_strength: f32,
    pub edge_bloom_strength: f32,
    pub environment_bloom_strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LabelSettings {
    pub enable_labels: bool,
    pub text_color: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSocketSettings {
    pub binary_chunk_size: usize,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub heartbeat_interval: u64,
    pub heartbeat_timeout: u64,
    pub max_connections: usize,
    pub max_message_size: usize,
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub update_rate: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    pub enable_data_debug: bool,
    pub enable_websocket_debug: bool,
    pub enabled: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GitHubSettings {
    pub base_path: String,
    pub owner: String,
    pub rate_limit: bool,
    pub repo: String,
    pub token: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISettings {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub rate_limit: u32,
    pub timeout: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerplexitySettings {
    pub api_key: String,
    pub api_url: String,
    pub frequency_penalty: f32,
    pub max_tokens: u32,
    pub model: String,
    pub prompt: String,
    pub rate_limit: u32,
    pub temperature: f32,
    pub timeout: u64,
    pub top_p: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagFlowSettings {
    pub api_key: String,
    pub base_url: String,
    pub max_retries: u32,
    pub timeout: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnimationSettings {
    #[serde(default)]
    pub enable_node_animations: bool,
    #[serde(default)]
    pub selection_wave_enabled: bool,
    #[serde(default)]
    pub pulse_enabled: bool,
    #[serde(default)]
    pub ripple_enabled: bool,
    #[serde(default)]
    pub edge_animation_enabled: bool,
    #[serde(default)]
    pub flow_particles_enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AudioSettings {
    #[serde(default)]
    pub enable_spatial_audio: bool,
    #[serde(default)]
    pub enable_interaction_sounds: bool,
    #[serde(default)]
    pub enable_ambient_sounds: bool,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            .add_source(File::with_name("settings.toml"))
            .add_source(
                Environment::default()
                    .separator("_")
                    .try_parsing(true)
            )
            .build()?;

        let mut settings: Settings = config.try_deserialize()?;
        
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

        Ok(settings)
    }
}
