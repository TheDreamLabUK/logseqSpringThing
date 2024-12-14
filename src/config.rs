use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment, File};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub server_debug: DebugSettings,
    pub client_debug: DebugSettings,
    pub network: NetworkSettings,
    pub security: SecuritySettings,
    pub github: GitHubSettings,
    pub ragflow: RagFlowSettings,
    pub perplexity: PerplexitySettings,
    pub openai: OpenAISettings,
    pub default: DefaultSettings,
    pub rendering: RenderingSettings,
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub labels: LabelSettings,
    pub bloom: BloomSettings,
    pub ar: ARSettings,
    pub physics: PhysicsSettings,
    pub animations: AnimationSettings,
    pub audio: AudioSettings,
    pub websocket: WebSocketSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSocketSettings {
    #[serde(default = "default_compression_enabled")]
    pub compression_enabled: bool,
    #[serde(default = "default_compression_threshold")]
    pub compression_threshold: usize,
    #[serde(default = "default_max_message_size")]
    pub max_message_size: usize,
    #[serde(default = "default_update_rate")]
    pub update_rate: u32,
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval: u64,
    #[serde(default = "default_heartbeat_timeout")]
    pub heartbeat_timeout: u64,
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_reconnect_attempts")]
    pub reconnect_attempts: u32,
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay: u64,
    #[serde(default = "default_binary_chunk_size")]
    pub binary_chunk_size: usize,
}

// Default functions for WebSocket settings
fn default_compression_enabled() -> bool { true }
fn default_compression_threshold() -> usize { 1024 }  // 1KB
fn default_max_message_size() -> usize { 100 * 1024 * 1024 }  // 100MB
fn default_update_rate() -> u32 { 5 }  // 5fps
fn default_heartbeat_interval() -> u64 { 15000 }  // 15 seconds
fn default_heartbeat_timeout() -> u64 { 60000 }  // 60 seconds
fn default_max_connections() -> usize { 1000 }
fn default_reconnect_attempts() -> u32 { 3 }
fn default_reconnect_delay() -> u64 { 5000 }  // 5 seconds
fn default_binary_chunk_size() -> usize { 64 * 1024 }  // 64KB

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    pub enabled: bool,
    pub enable_websocket_debug: bool,
    pub enable_data_debug: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            enable_websocket_debug: false,
            enable_data_debug: false,
            log_binary_headers: false,
            log_full_json: false,
        }
    }
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
    
    #[serde(default = "default_rate_limit")]
    pub rate_limit: bool,
}

fn default_token() -> String { "".to_string() }
fn default_owner() -> String { "".to_string() }
fn default_repo() -> String { "".to_string() }
fn default_path() -> String { "".to_string() }
fn default_rate_limit() -> bool { true }

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
    pub prompt: String,
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
    pub log_format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RenderingSettings {
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub enable_ambient_occlusion: bool,
    pub shadow_map_size: u32,
    pub pixel_ratio: f32,
    pub enable_gpu_acceleration: bool,
    pub background_color: String,
    pub environment_intensity: f32,
    pub ambient_light_intensity: f32,
    pub directional_light_intensity: f32,
    pub enable_hemisphere_light: bool,
    pub fog_enabled: bool,
    pub fog_color: String,
    pub fog_density: f32,
    pub enable_grid: bool,
    pub grid_size: u32,
    pub grid_divisions: u32,
    pub grid_color: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeSettings {
    pub base_size: f32,
    pub size_range: Vec<f32>,
    pub size_by_connections: bool,
    pub geometry_segments: u32,
    pub enable_instancing: bool,
    pub material_type: String,
    pub metalness: f32,
    pub roughness: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub opacity: f32,
    pub enable_transparency: bool,
    pub base_color: String,
    pub color_scheme: String,
    pub new_node_color: String,
    pub old_node_color: String,
    pub core_node_color: String,
    pub secondary_node_color: String,
    pub age_max_days: u32,
    pub highlight_color: String,
    pub highlight_intensity: f32,
    pub highlight_duration: u32,
    pub enable_hover_effect: bool,
    pub hover_scale: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EdgeSettings {
    pub base_width: f32,
    pub width_range: Vec<f32>,
    pub width_by_strength: bool,
    pub curve_segments: u32,
    pub enable_arrows: bool,
    pub arrow_size: f32,
    pub opacity: f32,
    pub color: String,
    pub highlight_color: String,
    pub enable_glow: bool,
    pub glow_intensity: f32,
    pub glow_color: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LabelSettings {
    pub enable_labels: bool,
    pub font_family: String,
    pub desktop_font_size: u32,
    pub ar_font_size: u32,
    pub padding: u32,
    pub background_opacity: f32,
    pub max_visible_labels: u32,
    pub vertical_offset: f32,
    pub close_offset: f32,
    pub view_angle_fade: f32,
    pub depth_fade_start: f32,
    pub depth_fade_end: f32,
    pub text_color: String,
    pub info_color: String,
    pub background_color: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BloomSettings {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
    pub threshold: f32,
    pub node_bloom_strength: f32,
    pub edge_bloom_strength: f32,
    pub environment_bloom_strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ARSettings {
    pub enable_plane_detection: bool,
    pub enable_light_estimation: bool,
    pub enable_hand_tracking: bool,
    pub enable_scene_understanding: bool,
    pub snap_to_floor: bool,
    pub room_scale: bool,
    pub hand_mesh_enabled: bool,
    pub hand_mesh_opacity: f32,
    pub hand_mesh_color: String,
    pub hand_ray_enabled: bool,
    pub hand_ray_color: String,
    pub hand_ray_width: f32,
    pub hand_point_size: f32,
    pub hand_trail_enabled: bool,
    pub hand_trail_length: f32,
    pub hand_trail_opacity: f32,
    pub pinch_threshold: f32,
    pub drag_threshold: f32,
    pub rotation_threshold: f32,
    pub enable_haptics: bool,
    pub haptic_intensity: f32,
    pub gesture_smoothing: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    pub enabled: bool,
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub damping: f32,
    pub enable_bounds: bool,
    pub bounds_size: f32,
    pub enable_collision: bool,
    pub collision_radius: f32,
    pub max_velocity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnimationSettings {
    pub enable_node_animations: bool,
    pub selection_wave_enabled: bool,
    pub selection_wave_color: String,
    pub selection_wave_speed: f32,
    pub selection_wave_size: f32,
    pub selection_wave_opacity: f32,
    pub pulse_enabled: bool,
    pub pulse_frequency: f32,
    pub pulse_amplitude: f32,
    pub pulse_color: String,
    pub ripple_enabled: bool,
    pub ripple_speed: f32,
    pub ripple_size: f32,
    pub ripple_segments: u32,
    pub ripple_color: String,
    pub ripple_decay: f32,
    pub edge_animation_enabled: bool,
    pub flow_particles_enabled: bool,
    pub flow_particle_count: u32,
    pub flow_particle_size: f32,
    pub flow_particle_speed: f32,
    pub flow_particle_color: String,
    pub flow_particle_trail: bool,
    pub flow_particle_trail_length: f32,
    pub edge_pulse_enabled: bool,
    pub edge_pulse_frequency: f32,
    pub edge_pulse_amplitude: f32,
    pub edge_pulse_color: String,
    pub edge_pulse_width: f32,
    pub animation_quality: String,
    pub enable_motion_blur: bool,
    pub motion_blur_strength: f32,
    pub animation_smoothing: f32,
    pub max_concurrent_animations: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AudioSettings {
    pub enable_spatial_audio: bool,
    pub master_volume: f32,
    pub audio_rolloff: String,
    pub max_audio_distance: f32,
    pub doppler_factor: f32,
    pub enable_interaction_sounds: bool,
    pub selection_sound_enabled: bool,
    pub selection_sound_volume: f32,
    pub hover_sound_enabled: bool,
    pub hover_sound_volume: f32,
    pub node_collision_sound: bool,
    pub node_collision_volume: f32,
    pub node_creation_sound: bool,
    pub node_creation_volume: f32,
    pub node_deletion_sound: bool,
    pub node_deletion_volume: f32,
    pub edge_creation_sound: bool,
    pub edge_creation_volume: f32,
    pub edge_deletion_sound: bool,
    pub edge_deletion_volume: f32,
    pub edge_flow_sound: bool,
    pub edge_flow_volume: f32,
    pub enable_ambient_sounds: bool,
    pub ambient_volume: f32,
    pub ambient_variation: f32,
    pub selection_frequency: u32,
    pub hover_frequency: u32,
    pub collision_frequency: u32,
    pub creation_frequency: u32,
    pub deletion_frequency: u32,
    pub flow_frequency: u32,
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
