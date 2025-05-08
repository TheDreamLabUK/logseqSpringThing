use config::{ConfigBuilder, ConfigError, Environment};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use std::path::PathBuf;

pub mod feature_access;

// Internal helper function to convert camelCase or kebab-case to snake_case
// This might still be useful for other parts or if direct snake_case interaction is needed.
fn to_snake_case(s: &str) -> String {
    let s = s.replace('-', "_");
    let mut result = String::with_capacity(s.len() + 4);
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_uppercase() {
            if !result.is_empty() {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct MovementAxes {
    pub horizontal: i32,
    pub vertical: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct NodeSettings {
    pub base_color: String,
    pub metalness: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub size_range: Vec<f32>, // TS has [number, number]
    pub quality: String, // TS has 'low' | 'medium' | 'high'
    pub enable_instancing: bool,
    pub enable_hologram: bool,
    pub enable_metadata_shape: bool,
    pub enable_metadata_visualization: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct EdgeSettings {
    pub arrow_size: f32,
    pub base_width: f32,
    pub color: String,
    pub enable_arrows: bool,
    pub opacity: f32,
    pub width_range: Vec<f32>, // TS has [number, number]
    pub quality: String, // TS has 'low' | 'medium' | 'high'
    // Fields from TS EdgeSettings not in Rust original:
    // enableFlowEffect: boolean;
    // flowSpeed: number;
    // flowIntensity: number;
    // glowStrength: number;
    // distanceIntensity: number;
    // useGradient: boolean;
    // gradientColors: [string, string];
    // For now, keeping Rust struct as is, client might send extra fields which serde will ignore by default.
    // If these are needed, add them here.
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
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
    pub repulsion_distance: f32,
    pub mass_scale: f32,
    pub boundary_damping: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
    // Fields from TS RenderingSettings not in Rust original:
    // shadowMapSize: string;
    // shadowBias: number;
    // context: 'desktop' | 'ar';
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AnimationSettings {
    pub enable_motion_blur: bool,
    pub enable_node_animations: bool,
    pub motion_blur_strength: f32,
    pub selection_wave_enabled: bool,
    pub pulse_enabled: bool,
    pub pulse_speed: f32,
    pub pulse_strength: f32,
    pub wave_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct LabelSettings {
    pub desktop_font_size: u32,
    pub enable_labels: bool,
    pub text_color: String,
    pub text_outline_color: String,
    pub text_outline_width: f32,
    pub text_resolution: u32,
    pub text_padding: u32,
    pub billboard_mode: String, // TS has 'camera' | 'vertical'
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct BloomSettings {
    pub edge_bloom_strength: f32,
    pub enabled: bool,
    pub environment_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub radius: f32,
    pub strength: f32,
    // Field from TS BloomSettings not in Rust original:
    // threshold: number;
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct HologramSettings {
    pub ring_count: u32,
    pub ring_color: String,
    pub ring_opacity: f32,
    pub sphere_sizes: Vec<f32>, // TS has [number, number]
    pub ring_rotation_speed: f32,
    pub enable_buckminster: bool,
    pub buckminster_size: f32,
    pub buckminster_opacity: f32,
    pub enable_geodesic: bool,
    pub geodesic_size: f32,
    pub geodesic_opacity: f32,
    pub enable_triangle_sphere: bool,
    pub triangle_sphere_size: f32,
    pub triangle_sphere_opacity: f32,
    pub global_rotation_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct VisualizationSettings {
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub physics: PhysicsSettings,
    pub rendering: RenderingSettings,
    pub animations: AnimationSettings,
    pub labels: LabelSettings,
    pub bloom: BloomSettings,
    pub hologram: HologramSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)] // Added Default
#[serde(rename_all = "camelCase")]
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
    pub api_client_timeout: u64,
    pub enable_metrics: bool,
    pub max_concurrent_requests: u32,
    pub max_retries: u32,
    pub metrics_port: u16,
    pub retry_delay: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)] // Default might not be needed if all fields have defaults or are covered by SystemSettings::default
#[serde(rename_all = "camelCase")]
pub struct WebSocketSettings { // This corresponds to TS WebSocketSettings which is part of TS SystemSettings
    pub binary_chunk_size: usize,
    // pub binary_update_rate: u32, // In Rust main WebSocketSettings, not TS client-side one
    // pub min_update_rate: u32,    // In Rust main WebSocketSettings
    // pub max_update_rate: u32,    // In Rust main WebSocketSettings
    // pub motion_threshold: f32,   // In Rust main WebSocketSettings
    // pub motion_damping: f32,     // In Rust main WebSocketSettings
    // pub binary_message_version: u32, // In Rust main WebSocketSettings
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    // pub heartbeat_interval: u64, // In Rust main WebSocketSettings
    // pub heartbeat_timeout: u64,  // In Rust main WebSocketSettings
    // pub max_connections: usize,  // In Rust main WebSocketSettings
    // pub max_message_size: usize, // In Rust main WebSocketSettings
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub update_rate: u32,
}

// Default for WebSocketSettings (matching TS client fields)
impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            reconnect_attempts: 3, // from client/src/features/settings/config/defaultSettings.ts
            reconnect_delay: 5000,
            binary_chunk_size: 65536,
            compression_enabled: true,
            compression_threshold: 1024,
            update_rate: 30, // from client default, not 90 from server default
        }
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, Default)] // Added Default
#[serde(rename_all = "camelCase")]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct DebugSettings { // This matches TS DebugSettings
    pub enabled: bool,
    pub enable_data_debug: bool,
    pub enable_websocket_debug: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
    // Fields from TS DebugSettings not in Rust original:
    // enablePhysicsDebug: boolean;
    // enableNodeDebug: boolean;
    // enableShaderDebug: boolean;
    // enableMatrixDebug: boolean;
    // enablePerformanceDebug: boolean;
    // Fields from Rust DebugSettings not in TS:
    // pub log_level: String,
    // pub log_format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)] // Added Default manually below
#[serde(rename_all = "camelCase")]
pub struct SystemSettings {
    #[serde(default)] // If client omits network, use default NetworkSettings
    pub network: NetworkSettings,
    pub websocket: WebSocketSettings, // This is the client-facing WebSocketSettings
    #[serde(default)] // If client omits security, use default SecuritySettings
    pub security: SecuritySettings,
    pub debug: DebugSettings,
    // TS SystemSettings has persistSettings: boolean - add if needed
    // #[serde(default)]
    // pub persist_settings: bool,
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            network: NetworkSettings::default(),
            websocket: WebSocketSettings::default(), // Client-facing defaults
            security: SecuritySettings::default(),
            debug: DebugSettings::default(),
            // persist_settings: true, // Default for the new field if added
        }
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct XRSettings {
    // pub mode: String, // TS has 'enabled' and 'displayMode'
    // pub room_scale: f32,
    // pub space_type: String,
    pub quality: String, // TS has this
    pub enable_hand_tracking: bool, // TS: handTracking
    // pub hand_mesh_enabled: bool,
    // pub hand_mesh_color: String,
    // pub hand_mesh_opacity: f32,
    // pub hand_point_size: f32,
    // pub hand_ray_enabled: bool,
    // pub hand_ray_color: String, // TS: controllerRayColor?
    // pub hand_ray_width: f32,
    // pub gesture_smoothing: f32,
    pub enable_haptics: bool,
    // pub drag_threshold: f32,
    // pub pinch_threshold: f32,
    // pub rotation_threshold: f32,
    // pub interaction_radius: f32, // TS: interactionDistance
    // pub movement_speed: f32,
    // pub dead_zone: f32,
    // pub movement_axes: MovementAxes,
    // pub enable_light_estimation: bool,
    // pub enable_plane_detection: bool,
    // pub enable_scene_understanding: bool,
    // pub plane_color: String,
    // pub plane_opacity: f32,
    // pub plane_detection_distance: f32,
    // pub show_plane_overlay: bool,
    // pub snap_to_floor: bool,
    // pub enable_passthrough_portal: bool,
    // pub passthrough_opacity: f32,
    // pub passthrough_brightness: f32,
    // pub passthrough_contrast: f32,
    // pub portal_size: f32,
    // pub portal_edge_color: String,
    // pub portal_edge_width: f32,

    // Fields from TS XRSettings
    pub enabled: bool,
    // handTracking is enable_hand_tracking
    pub controller_model: String,
    pub render_scale: f32,
    pub interaction_distance: f32, // Corresponds to interaction_radius
    pub locomotion_method: String, // TS: 'teleport' | 'continuous'
    pub teleport_ray_color: String,
    // enableHaptics is enable_haptics
    pub display_mode: String, // TS: 'stereo' | 'mono'
    #[serde(default)] // Optional in TS
    pub controller_ray_color: Option<String>,
}


#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AuthSettings {
    pub enabled: bool,
    pub provider: String,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct RagFlowSettings {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub api_base_url: Option<String>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub max_retries: Option<u32>,
    // TS has chat_id?: string - add if needed
    // #[serde(default)]
    // pub chat_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct PerplexitySettings {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub api_url: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct OpenAISettings {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct KokoroSettings {
    #[serde(default)]
    pub api_url: Option<String>,
    #[serde(default)]
    pub default_voice: Option<String>,
    #[serde(default)]
    pub default_format: Option<String>,
    #[serde(default)]
    pub default_speed: Option<f32>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub return_timestamps: Option<bool>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

// Main settings struct
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub visualization: VisualizationSettings,
    pub system: SystemSettings,
    pub xr: XRSettings,
    pub auth: AuthSettings, // Added
    #[serde(default)]       // If entire section is missing from payload
    pub ragflow: Option<RagFlowSettings>,
    #[serde(default)]
    pub perplexity: Option<PerplexitySettings>,
    #[serde(default)]
    pub openai: Option<OpenAISettings>,
    #[serde(default)]
    pub kokoro: Option<KokoroSettings>,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing settings");
        dotenvy::dotenv().ok();
        println!("[CONFIG DEBUG] Checking SYSTEM_NETWORK_PORT env var: {:?}", std::env::var("SYSTEM_NETWORK_PORT"));

        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));
        debug!("Loading settings from YAML file: {:?}", settings_path);

        let builder = ConfigBuilder::<config::builder::DefaultState>::default()
            .add_source(config::File::from(settings_path).required(true))
            .add_source(
                Environment::default()
                    .separator("_")
                    .list_separator(",")
            );
        debug!("Building configuration by layering sources (YAML -> Env Vars)");
        let config = builder.build()?;
        debug!("Configuration built successfully");
        
        // Deserialize with rename_all in mind for the struct itself
        // The config crate's try_deserialize will respect serde attributes on the Settings struct
        config.try_deserialize()
    }

    // This merge function might need review if it's still used,
    // especially the to_snake_case_value part if Settings now expects camelCase.
    // For now, focusing on the deserialization in settings_handler.
    pub fn merge(&mut self, value: Value) -> Result<(), String> {
        let new_settings: Settings = serde_json::from_value(value) // Directly deserialize (assuming value is camelCase)
            .map_err(|e| format!("Failed to deserialize settings for merge: {}", e))?;

        // Selective merge logic (example, expand as needed)
        self.visualization = new_settings.visualization; // Example: overwrite whole section
        self.system.websocket = new_settings.system.websocket; // Example: overwrite sub-section
        self.system.debug = new_settings.system.debug;
        // self.system.network and self.system.security are intentionally not merged from client payload
        
        self.xr = new_settings.xr;
        self.auth = new_settings.auth;

        if new_settings.ragflow.is_some() { self.ragflow = new_settings.ragflow; }
        if new_settings.perplexity.is_some() { self.perplexity = new_settings.perplexity; }
        if new_settings.openai.is_some() { self.openai = new_settings.openai; }
        if new_settings.kokoro.is_some() { self.kokoro = new_settings.kokoro; }
        
        Ok(())
    }

    pub fn save(&self) -> Result<(), String> {
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));
        let yaml = serde_yaml::to_string(&self)
            .map_err(|e| format!("Failed to serialize settings to YAML: {}", e))?;
        std::fs::write(&settings_path, yaml)
            .map_err(|e| format!("Failed to write settings file: {}", e))?;
        Ok(())
    }

    // This function might be obsolete if direct camelCase deserialization is used everywhere.
    // Keeping it for now in case it's used elsewhere.
    fn to_snake_case_value(&self, value: Value) -> Value {
        match value {
            Value::Object(map) => {
                let converted: serde_json::Map<String, Value> = map
                    .into_iter()
                    .map(|(k, v)| (to_snake_case(&k), self.to_snake_case_value(v)))
                    .collect();
                Value::Object(converted)
            }
            Value::Array(arr) => Value::Array(
                arr.into_iter().map(|v| self.to_snake_case_value(v)).collect(),
            ),
            _ => value,
        }
    }

    pub fn from_env() -> Result<Self, ConfigError> {
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            .add_source(Environment::default().separator("_").try_parsing(true)) // try_parsing might be an issue with complex structs
            .build()?;
        config.try_deserialize()
    }
}

// Default implementation for the main Settings struct
impl Default for Settings {
    fn default() -> Self {
        // Create default instances for nested structs that also implement Default
        Self {
            visualization: VisualizationSettings::default(),
            system: SystemSettings::default(),
            xr: XRSettings::default(),
            auth: AuthSettings::default(), // Initialize new auth field
            ragflow: None, // Default optional AI settings to None
            perplexity: None,
            openai: None,
            kokoro: None,
        }
    }
}

// Note: The original Default impl for Settings (lines 450-682) was very detailed.
// The new Default above uses the ::default() of sub-structs.
// If the sub-structs' defaults are not sufficient or differ from the original detailed main Default,
// those sub-structs' Default impls (or this main one) would need to be adjusted to match the original intent.
// For example, the original WebSocketSettings within SystemSettings had specific values
// that might differ from a simple WebSocketSettings::default().
// The current WebSocketSettings::default() above tries to match client defaults.
// The original NetworkSettings and SecuritySettings did not have Default, so their new Default impls are basic.
// This might require further refinement if specific default values from the old large block are critical.

#[cfg(test)]
mod tests {
    mod feature_access_test;
}