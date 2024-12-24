use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment, File};
use log::{debug, error};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Settings {
    // UI/Rendering settings from settings.toml
    #[serde(default)]
    pub animations: AnimationSettings,
    #[serde(default)]
    pub ar: ARSettings,
    #[serde(default)]
    pub audio: AudioSettings,
    #[serde(default)]
    pub bloom: BloomSettings,
    #[serde(default)]
    pub client_debug: DebugSettings,
    #[serde(default)]
    pub default: DefaultSettings,
    #[serde(default)]
    pub edges: EdgeSettings,
    #[serde(default)]
    pub labels: LabelSettings,
    #[serde(default)]
    pub network: NetworkSettings,
    #[serde(default)]
    pub nodes: NodeSettings,
    #[serde(default)]
    pub physics: PhysicsSettings,
    #[serde(default)]
    pub rendering: RenderingSettings,
    #[serde(default)]
    pub security: SecuritySettings,
    #[serde(default)]
    pub server_debug: DebugSettings,
    #[serde(default)]
    pub websocket: WebSocketSettings,
    
    // Service settings from .env (server-side only)
    #[serde(default)]
    pub github: GitHubSettings,
    #[serde(default)]
    pub ragflow: RagFlowSettings,
    #[serde(default)]
    pub perplexity: PerplexitySettings,
    #[serde(default)]
    pub openai: OpenAISettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct DebugSettings {
    pub enable_data_debug: bool,
    pub enable_websocket_debug: bool,
    pub enabled: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            enable_data_debug: false,
            enable_websocket_debug: false,
            enabled: false,
            log_binary_headers: false,
            log_full_json: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct DefaultSettings {
    pub api_client_timeout: u32,
    pub enable_metrics: bool,
    pub enable_request_logging: bool,
    pub log_format: String,
    pub log_level: String,
    pub max_concurrent_requests: u32,
    pub max_payload_size: usize,
    pub max_retries: u32,
    pub metrics_port: u16,
    pub retry_delay: u32,
}

impl Default for DefaultSettings {
    fn default() -> Self {
        Self {
            api_client_timeout: 30,
            enable_metrics: true,
            enable_request_logging: true,
            log_format: "json".to_string(),
            log_level: "debug".to_string(),
            max_concurrent_requests: 5,
            max_payload_size: 5242880,
            max_retries: 3,
            metrics_port: 9090,
            retry_delay: 5,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
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

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            domain: "localhost".to_string(),
            enable_http2: false,
            enable_rate_limiting: true,
            enable_tls: false,
            max_request_size: 10485760,
            min_tls_version: String::new(),
            port: 3001,
            rate_limit_requests: 100,
            rate_limit_window: 60,
            tunnel_id: "dummy".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GitHubSettings {
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub base_path: String,
    #[serde(default)]
    pub version: String,
    #[serde(default = "default_true")]
    pub rate_limit: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RagFlowSettings {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub api_base_url: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub chat_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PerplexitySettings {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub api_url: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_rate_limit")]
    pub rate_limit: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct OpenAISettings {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_rate_limit")]
    pub rate_limit: u32,
}

// Default value functions
fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    30
}

fn default_max_retries() -> u32 {
    3
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f32 {
    0.5
}

fn default_top_p() -> f32 {
    0.9
}

fn default_rate_limit() -> u32 {
    100
}

// UI/Rendering settings from settings.toml (using snake_case as they're shared with client)

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct AnimationSettings {
    pub enable_motion_blur: bool,
    pub enable_node_animations: bool,
    pub motion_blur_strength: f32,
    pub pulse_enabled: bool,
    pub ripple_enabled: bool,
    pub edge_animation_enabled: bool,
    pub flow_particles_enabled: bool,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            enable_motion_blur: false,
            enable_node_animations: false,
            motion_blur_strength: 0.4,
            pulse_enabled: false,
            ripple_enabled: false,
            edge_animation_enabled: false,
            flow_particles_enabled: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct ARSettings {
    pub drag_threshold: f32,
    pub enable_hand_tracking: bool,
    pub enable_haptics: bool,
    pub enable_light_estimation: bool,
    pub enable_passthrough_portal: bool,
    pub enable_plane_detection: bool,
    pub enable_scene_understanding: bool,
    pub gesture_smoothing: f32,
    pub hand_mesh_color: String,
    pub hand_mesh_enabled: bool,
    pub hand_mesh_opacity: f32,
    pub hand_point_size: f32,
    pub hand_ray_color: String,
    pub hand_ray_enabled: bool,
    pub hand_ray_width: f32,
    pub haptic_intensity: f32,
    pub passthrough_brightness: f32,
    pub passthrough_contrast: f32,
    pub passthrough_opacity: f32,
    pub pinch_threshold: f32,
    pub plane_color: String,
    pub plane_opacity: f32,
    pub portal_edge_color: String,
    pub portal_edge_width: f32,
    pub portal_size: f32,
    pub room_scale: bool,
    pub rotation_threshold: f32,
    pub show_plane_overlay: bool,
    pub snap_to_floor: bool,
}

impl Default for ARSettings {
    fn default() -> Self {
        Self {
            drag_threshold: 0.04,
            enable_hand_tracking: true,
            enable_haptics: true,
            enable_light_estimation: true,
            enable_passthrough_portal: false,
            enable_plane_detection: true,
            enable_scene_understanding: true,
            gesture_smoothing: 0.5,
            hand_mesh_color: "#FFD700".to_string(),
            hand_mesh_enabled: true,
            hand_mesh_opacity: 0.3,
            hand_point_size: 0.01,
            hand_ray_color: "#FFD700".to_string(),
            hand_ray_enabled: true,
            hand_ray_width: 0.002,
            haptic_intensity: 0.7,
            passthrough_brightness: 1.0,
            passthrough_contrast: 1.0,
            passthrough_opacity: 0.8,
            pinch_threshold: 0.015,
            plane_color: "#808080".to_string(),
            plane_opacity: 0.5,
            portal_edge_color: "#00FF00".to_string(),
            portal_edge_width: 0.02,
            portal_size: 2.0,
            room_scale: true,
            rotation_threshold: 0.08,
            show_plane_overlay: true,
            snap_to_floor: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct AudioSettings {
    pub enable_spatial_audio: bool,
    pub enable_interaction_sounds: bool,
    pub enable_ambient_sounds: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            enable_spatial_audio: false,
            enable_interaction_sounds: false,
            enable_ambient_sounds: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct BloomSettings {
    pub edge_bloom_strength: f32,
    pub enabled: bool,
    pub environment_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub radius: f32,
    pub strength: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            edge_bloom_strength: 0.3,
            enabled: false,
            environment_bloom_strength: 0.5,
            node_bloom_strength: 0.2,
            radius: 0.5,
            strength: 1.8,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct EdgeSettings {
    pub arrow_size: f32,
    pub base_width: f32,
    pub color: String,
    pub enable_arrows: bool,
    pub opacity: f32,
    pub width_range: Vec<f32>,
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            arrow_size: 0.2,
            base_width: 2.0,
            color: "#917f18".to_string(),
            enable_arrows: false,
            opacity: 0.6,
            width_range: vec![1.0, 3.0],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct LabelSettings {
    pub desktop_font_size: u32,
    pub enable_labels: bool,
    pub text_color: String,
}

impl Default for LabelSettings {
    fn default() -> Self {
        Self {
            desktop_font_size: 48,
            enable_labels: true,
            text_color: "#FFFFFF".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct NodeSettings {
    pub base_color: String,
    pub base_size: f32,
    pub clearcoat: f32,
    pub enable_hover_effect: bool,
    pub enable_instancing: bool,
    pub highlight_color: String,
    pub highlight_duration: u32,
    pub hover_scale: f32,
    pub material_type: String,
    pub metalness: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub size_by_connections: bool,
    pub size_range: Vec<f32>,
}

impl Default for NodeSettings {
    fn default() -> Self {
        Self {
            base_color: "#c3ab6f".to_string(),
            base_size: 1.0,
            clearcoat: 0.5,
            enable_hover_effect: false,
            enable_instancing: false,
            highlight_color: "#822626".to_string(),
            highlight_duration: 300,
            hover_scale: 1.2,
            material_type: "basic".to_string(),
            metalness: 0.3,
            opacity: 0.4,
            roughness: 0.35,
            size_by_connections: true,
            size_range: vec![1.0, 10.0],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
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

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            attraction_strength: 0.015,
            bounds_size: 12.0,
            collision_radius: 0.25,
            damping: 0.88,
            enable_bounds: true,
            enabled: false,
            iterations: 500,
            max_velocity: 2.5,
            repulsion_strength: 1500.0,
            spring_strength: 0.018,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            ambient_light_intensity: 0.7,
            background_color: "#000000".to_string(),
            directional_light_intensity: 1.0,
            enable_ambient_occlusion: false,
            enable_antialiasing: true,
            enable_shadows: false,
            environment_intensity: 1.2,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
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

impl Default for SecuritySettings {
    fn default() -> Self {
        Self {
            allowed_origins: Vec::new(),
            audit_log_path: "/app/logs/audit.log".to_string(),
            cookie_httponly: true,
            cookie_samesite: "Strict".to_string(),
            cookie_secure: true,
            csrf_token_timeout: 3600,
            enable_audit_logging: true,
            enable_request_validation: true,
            session_timeout: 3600,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
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

impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            binary_chunk_size: 65536,
            compression_enabled: true,
            compression_threshold: 1024,
            heartbeat_interval: 15000,
            heartbeat_timeout: 60000,
            max_connections: 1000,
            max_message_size: 100485760,
            reconnect_attempts: 3,
            reconnect_delay: 5000,
            update_rate: 90,
        }
    }
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing settings");
        
        // Load .env file first
        dotenvy::dotenv().ok();
        
        // Use environment variable or default to /app/settings.toml
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.toml"));
        
        debug!("Loading settings from: {:?}", settings_path);
        
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            .add_source(File::from(settings_path))
            .add_source(
                Environment::default()
                    .separator("_")
                    .try_parsing(true)
            )
            .build()?;

        debug!("Deserializing settings");
        let mut settings: Settings = match config.try_deserialize() {
            Ok(s) => {
                debug!("Successfully deserialized settings");
                s
            },
            Err(e) => {
                error!("Failed to deserialize settings: {}", e);
                return Err(e);
            }
        };
        
        debug!("Checking for environment variables");
        
        // Network settings from environment variables
        if let Ok(domain) = std::env::var("DOMAIN") {
            settings.network.domain = domain;
        }
        if let Ok(port) = std::env::var("PORT") {
            settings.network.port = port.parse().unwrap_or(4000);
        }
        if let Ok(bind_address) = std::env::var("BIND_ADDRESS") {
            settings.network.bind_address = bind_address;
        }
        if let Ok(tunnel_id) = std::env::var("TUNNEL_ID") {
            settings.network.tunnel_id = tunnel_id;
        }
        if let Ok(enable_http2) = std::env::var("HTTP2_ENABLED") {
            settings.network.enable_http2 = enable_http2.parse().unwrap_or(true);
        }

        // GitHub settings from environment variables
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
        if let Ok(rate_limit) = std::env::var("GITHUB_RATE_LIMIT") {
            settings.github.rate_limit = rate_limit.parse().unwrap_or(true);
        }

        // RAGFlow settings from environment variables
        if let Ok(api_key) = std::env::var("RAGFLOW_API_KEY") {
            settings.ragflow.api_key = api_key;
        }
        if let Ok(base_url) = std::env::var("RAGFLOW_API_BASE_URL") {
            settings.ragflow.api_base_url = base_url;
        }
        if let Ok(timeout) = std::env::var("RAGFLOW_TIMEOUT") {
            settings.ragflow.timeout = timeout.parse().unwrap_or(30);
        }
        if let Ok(max_retries) = std::env::var("RAGFLOW_MAX_RETRIES") {
            settings.ragflow.max_retries = max_retries.parse().unwrap_or(3);
        }
        if let Ok(chat_id) = std::env::var("RAGFLOW_CHAT_ID") {
            settings.ragflow.chat_id = chat_id;
        }

        // Perplexity settings from environment variables
        if let Ok(api_key) = std::env::var("PERPLEXITY_API_KEY") {
            settings.perplexity.api_key = api_key;
        }
        if let Ok(api_url) = std::env::var("PERPLEXITY_API_URL") {
            settings.perplexity.api_url = api_url;
        }
        if let Ok(model) = std::env::var("PERPLEXITY_MODEL") {
            settings.perplexity.model = model;
        }
        if let Ok(max_tokens) = std::env::var("PERPLEXITY_MAX_TOKENS") {
            settings.perplexity.max_tokens = max_tokens.parse().unwrap_or(4096);
        }
        if let Ok(temperature) = std::env::var("PERPLEXITY_TEMPERATURE") {
            settings.perplexity.temperature = temperature.parse().unwrap_or(0.5);
        }
        if let Ok(top_p) = std::env::var("PERPLEXITY_TOP_P") {
            settings.perplexity.top_p = top_p.parse().unwrap_or(0.9);
        }
        if let Ok(presence_penalty) = std::env::var("PERPLEXITY_PRESENCE_PENALTY") {
            settings.perplexity.presence_penalty = presence_penalty.parse().unwrap_or(0.0);
        }
        if let Ok(frequency_penalty) = std::env::var("PERPLEXITY_FREQUENCY_PENALTY") {
            settings.perplexity.frequency_penalty = frequency_penalty.parse().unwrap_or(1.0);
        }
        if let Ok(timeout) = std::env::var("PERPLEXITY_TIMEOUT") {
            settings.perplexity.timeout = timeout.parse().unwrap_or(30);
        }
        if let Ok(rate_limit) = std::env::var("PERPLEXITY_RATE_LIMIT") {
            settings.perplexity.rate_limit = rate_limit.parse().unwrap_or(100);
        }

        // OpenAI settings from environment variables
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            settings.openai.api_key = api_key;
        }
        if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            settings.openai.base_url = base_url;
        }
        if let Ok(timeout) = std::env::var("OPENAI_TIMEOUT") {
            settings.openai.timeout = timeout.parse().unwrap_or(30);
        }
        if let Ok(rate_limit) = std::env::var("OPENAI_RATE_LIMIT") {
            settings.openai.rate_limit = rate_limit.parse().unwrap_or(100);
        }

        debug!("Successfully loaded settings");
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

        config.try_deserialize()
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            animations: AnimationSettings::default(),
            ar: ARSettings::default(),
            audio: AudioSettings::default(),
            bloom: BloomSettings::default(),
            client_debug: DebugSettings::default(),
            default: DefaultSettings::default(),
            edges: EdgeSettings::default(),
            labels: LabelSettings::default(),
            nodes: NodeSettings::default(),
            physics: PhysicsSettings::default(),
            rendering: RenderingSettings::default(),
            security: SecuritySettings::default(),
            server_debug: DebugSettings::default(),
            websocket: WebSocketSettings::default(),
            network: NetworkSettings::default(),
            github: GitHubSettings::default(),
            ragflow: RagFlowSettings::default(),
            perplexity: PerplexitySettings::default(),
            openai: OpenAISettings::default(),
        }
    }
}
