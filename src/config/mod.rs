use config::{ConfigBuilder, ConfigError, Environment};
use log::{debug, error}; // Added error log
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use std::path::PathBuf;
// use std::collections::BTreeMap; // For ordered map during serialization - Removed as unused

pub mod feature_access;

// Recursive function to convert JSON Value keys to snake_case
fn keys_to_snake_case(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let new_map = map.into_iter().map(|(k, v)| {
                let snake_key = k.chars().fold(String::new(), |mut acc, c| {
                    if c.is_ascii_uppercase() {
                        if !acc.is_empty() {
                            acc.push('_');
                        }
                        acc.push(c.to_ascii_lowercase());
                    } else {
                        acc.push(c);
                    }
                    acc
                });
                (snake_key, keys_to_snake_case(v))
            }).collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(keys_to_snake_case).collect())
        }
        _ => value,
    }
}

// Recursive function to convert JSON Value keys to camelCase (if needed for comparison/debugging)
fn _keys_to_camel_case(value: Value) -> Value {
     match value {
         Value::Object(map) => {
             let new_map = map.into_iter().map(|(k, v)| {
                 let camel_key = k.split('_').enumerate().map(|(i, part)| {
                     if i == 0 {
                         part.to_string()
                     } else {
                         part.chars().next().map_or(String::new(), |c| c.to_uppercase().collect::<String>() + &part[1..])
                     }
                 }).collect::<String>();
                 (camel_key, _keys_to_camel_case(v))
             }).collect();
             Value::Object(new_map)
         }
         Value::Array(arr) => {
             Value::Array(arr.into_iter().map(_keys_to_camel_case).collect())
         }
         _ => value,
     }
 }


#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct MovementAxes {
    pub horizontal: i32,
    pub vertical: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct NodeSettings {
    pub base_color: String,
    pub metalness: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub size_range: Vec<f32>,
    pub quality: String,
    pub enable_instancing: bool,
    pub enable_hologram: bool,
    pub enable_metadata_shape: bool,
    pub enable_metadata_visualisation: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct EdgeSettings {
    pub arrow_size: f32,
    pub base_width: f32,
    pub color: String,
    pub enable_arrows: bool,
    pub opacity: f32,
    pub width_range: Vec<f32>,
    pub quality: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
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
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
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
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct LabelSettings {
    pub desktop_font_size: f32,
    pub enable_labels: bool,
    pub text_color: String,
    pub text_outline_color: String,
    pub text_outline_width: f32,
    pub text_resolution: u32,
    pub text_padding: f32,
    pub billboard_mode: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct BloomSettings {
    pub edge_bloom_strength: f32,
    pub enabled: bool,
    pub environment_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub radius: f32,
    pub strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct HologramSettings {
    pub ring_count: u32,
    pub ring_color: String,
    pub ring_opacity: f32,
    pub sphere_sizes: Vec<f32>,
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
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct VisualisationSettings {
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub physics: PhysicsSettings,
    pub rendering: RenderingSettings,
    pub animations: AnimationSettings,
    pub labels: LabelSettings,
    pub bloom: BloomSettings,
    pub hologram: HologramSettings,
}

// --- Server-Specific Config Structs (from YAML, snake_case) ---

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// No rename_all needed if YAML keys are snake_case
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

#[derive(Debug, Serialize, Deserialize, Clone)]
// No rename_all needed if YAML keys are snake_case
pub struct ServerFullWebSocketSettings {
    pub binary_chunk_size: usize,
    pub binary_update_rate: u32,
    pub min_update_rate: u32,
    pub max_update_rate: u32,
    pub motion_threshold: f32,
    pub motion_damping: f32,
    pub binary_message_version: u32,
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

impl Default for ServerFullWebSocketSettings {
    fn default() -> Self { // Defaults from settings.yaml
        Self {
            binary_chunk_size: 2048, binary_update_rate: 30, min_update_rate: 5,
            max_update_rate: 60, motion_threshold: 0.05, motion_damping: 0.9,
            binary_message_version: 1, compression_enabled: false, compression_threshold: 512,
            heartbeat_interval: 10000, heartbeat_timeout: 600000, max_connections: 100,
            max_message_size: 10485760, reconnect_attempts: 5, reconnect_delay: 1000,
            update_rate: 60,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// No rename_all needed if YAML keys are snake_case
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
// #[serde(rename_all = "camelCase")] // REMOVED: YAML keys are snake_case. Kept for JSON if this struct is directly serialized to client.
pub struct DebugSettings { // Matches TS DebugSettings + YAML fields
    pub enabled: bool,
    pub enable_data_debug: bool,
    pub enable_websocket_debug: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
    // Added back from YAML - these might need snake_case for YAML loading if config crate doesn't handle rename_all
    // Let's assume config crate handles it based on struct field names for YAML.
    pub log_level: String,
    pub log_format: String,
}


#[derive(Debug, Deserialize, Clone)] // Only Deserialize needed for loading YAML
// No rename_all needed if YAML keys are snake_case
pub struct ServerSystemConfigFromFile {
    pub network: NetworkSettings,
    pub websocket: ServerFullWebSocketSettings,
    pub security: SecuritySettings,
    pub debug: DebugSettings, // Assumes YAML debug section matches DebugSettings struct fields (snake_case)
    #[serde(default)]
    pub persist_settings: bool,
}

// --- Client-Facing Config Structs (for JSON, camelCase) ---

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientWebSocketSettings { // What client sends/expects
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub binary_chunk_size: usize,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub update_rate: u32,
}

impl Default for ClientWebSocketSettings {
    fn default() -> Self {
        Self {
            reconnect_attempts: 3, reconnect_delay: 5000, binary_chunk_size: 65536,
            compression_enabled: true, compression_threshold: 1024, update_rate: 30,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SystemSettings { // Client-facing System structure
    pub websocket: ClientWebSocketSettings,
    pub debug: DebugSettings, // DebugSettings uses camelCase for JSON
    #[serde(default)]
    pub persist_settings: bool,
    // network and security are not part of client-facing system settings
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            websocket: ClientWebSocketSettings::default(),
            debug: DebugSettings::default(),
            persist_settings: true,
        }
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct XRSettings { // Client-facing XR structure + YAML fields
    // Fields from YAML (snake_case in YAML, camelCase in JSON)
    pub mode: String,
    pub room_scale: f32,
    pub space_type: String,
    pub quality: String,
    #[serde(alias = "handTracking")]
    pub enable_hand_tracking: bool,
    pub hand_mesh_enabled: bool,
    pub hand_mesh_color: String,
    pub hand_mesh_opacity: f32,
    pub hand_point_size: f32,
    pub hand_ray_enabled: bool,
    pub hand_ray_color: String,
    pub hand_ray_width: f32,
    pub gesture_smoothing: f32,
    pub enable_haptics: bool,
    pub drag_threshold: f32,
    pub pinch_threshold: f32,
    pub rotation_threshold: f32,
    #[serde(alias = "interactionDistance")]
    pub interaction_radius: f32,
    pub movement_speed: f32,
    pub dead_zone: f32,
    pub movement_axes: MovementAxes,
    pub enable_light_estimation: bool,
    pub enable_plane_detection: bool,
    pub enable_scene_understanding: bool,
    pub plane_color: String,
    pub plane_opacity: f32,
    pub plane_detection_distance: f32,
    pub show_plane_overlay: bool,
    pub snap_to_floor: bool,
    pub enable_passthrough_portal: bool,
    pub passthrough_opacity: f32,
    pub passthrough_brightness: f32,
    pub passthrough_contrast: f32,
    pub portal_size: f32,
    pub portal_edge_color: String,
    pub portal_edge_width: f32,

    // Fields from TS (camelCase in JSON)
    #[serde(default)]
    pub enabled: Option<bool>, // TS 'enabled' field
    #[serde(default)]
    pub controller_model: Option<String>,
    #[serde(default)]
    pub render_scale: Option<f32>,
    #[serde(default)]
    pub locomotion_method: Option<String>,
    #[serde(default)]
    pub teleport_ray_color: Option<String>,
    #[serde(default)]
    pub display_mode: Option<String>,
    #[serde(default)]
    pub controller_ray_color: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML (fields are single word, so no practical change)
pub struct AuthSettings { // Client-facing
    pub enabled: bool,
    pub provider: String,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct RagFlowSettings { // Client-facing
    #[serde(default)] pub api_key: Option<String>,
    #[serde(default)] pub agent_id: Option<String>,
    #[serde(default)] pub api_base_url: Option<String>,
    #[serde(default)] pub timeout: Option<u64>,
    #[serde(default)] pub max_retries: Option<u32>,
    #[serde(default)] pub chat_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct PerplexitySettings { // Client-facing
    #[serde(default)] pub api_key: Option<String>,
    #[serde(default)] pub model: Option<String>,
    #[serde(default)] pub api_url: Option<String>,
    #[serde(default)] pub max_tokens: Option<u32>,
    #[serde(default)] pub temperature: Option<f32>,
    #[serde(default)] pub top_p: Option<f32>,
    #[serde(default)] pub presence_penalty: Option<f32>,
    #[serde(default)] pub frequency_penalty: Option<f32>,
    #[serde(default)] pub timeout: Option<u64>,
    #[serde(default)] pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct OpenAISettings { // Client-facing
    #[serde(default)] pub api_key: Option<String>,
    #[serde(default)] pub base_url: Option<String>,
    #[serde(default)] pub timeout: Option<u64>,
    #[serde(default)] pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Removed for snake_case YAML
pub struct KokoroSettings { // Client-facing
    #[serde(default)] pub api_url: Option<String>,
    #[serde(default)] pub default_voice: Option<String>,
    #[serde(default)] pub default_format: Option<String>,
    #[serde(default)] pub default_speed: Option<f32>,
    #[serde(default)] pub timeout: Option<u64>,
    #[serde(default)] pub stream: Option<bool>,
    #[serde(default)] pub return_timestamps: Option<bool>,
    #[serde(default)] pub sample_rate: Option<u32>,
}

// --- Client-Facing Settings Struct (for JSON deserialization) ---
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Settings { // Renamed to ClientFacingSettings conceptually
    pub visualisation: VisualisationSettings,
    pub system: SystemSettings, // Uses ClientWebSocketSettings internally
    pub xr: XRSettings,
    pub auth: AuthSettings,
    #[serde(default)] pub ragflow: Option<RagFlowSettings>,
    #[serde(default)] pub perplexity: Option<PerplexitySettings>,
    #[serde(default)] pub openai: Option<OpenAISettings>,
    #[serde(default)] pub kokoro: Option<KokoroSettings>,
}

// --- Full App Settings Struct (for server state, loaded from YAML) ---
#[derive(Debug, Clone, Deserialize)] // Deserialize for YAML loading
// No rename_all needed if YAML keys are snake_case
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings, // Assumes YAML keys are snake_case
    pub system: ServerSystemConfigFromFile,   // Contains ServerFullWebSocketSettings
    pub xr: XRSettings,                       // Assumes YAML keys are snake_case
    pub auth: AuthSettings,                   // Assumes YAML keys are snake_case
    #[serde(default)] pub ragflow: Option<RagFlowSettings>, // Assumes YAML keys are snake_case
    #[serde(default)] pub perplexity: Option<PerplexitySettings>,
    #[serde(default)] pub openai: Option<OpenAISettings>,
    #[serde(default)] pub kokoro: Option<KokoroSettings>,
}

// Manual Serialize implementation for AppFullSettings to ensure snake_case YAML output
impl Serialize for AppFullSettings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Convert self to a serde_json::Value first.
        // The sub-structs might have rename_all="camelCase", so this Value will be camelCase.
        match serde_json::to_value(self) {
            Ok(camel_case_value) => {
                // Convert the camelCase Value to snake_case Value.
                let snake_case_value = keys_to_snake_case(camel_case_value);
                // Serialize the snake_case Value.
                snake_case_value.serialize(serializer)
            }
            Err(e) => {
                error!("Failed to convert AppFullSettings to intermediate Value for saving: {}", e);
                // Handle error appropriately, maybe serialize a default or error state
                Err(serde::ser::Error::custom(format!("Serialization error: {}", e)))
            }
        }
    }
}
// We also need Serialize for the sub-structs used by AppFullSettings
// if they are not already deriving Serialize. They are deriving it, but
// their rename_all attribute will cause camelCase serialization.
// The keys_to_snake_case function handles this during AppFullSettings serialization.


impl AppFullSettings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing AppFullSettings from YAML");
        dotenvy::dotenv().ok();

        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));
        debug!("Loading AppFullSettings from YAML file: {:?}", settings_path);

        let builder = ConfigBuilder::<config::builder::DefaultState>::default()
            .add_source(config::File::from(settings_path.clone()).required(true)) // Use path directly
            .add_source(
                Environment::default()
                    .separator("_") // Match SYSTEM_NETWORK_PORT style
                    .list_separator(",")
                // No .prefix("APP") // Allow direct environment variable override
            );
        let config = builder.build()?;
        debug!("Configuration built successfully. Deserializing AppFullSettings...");
        
        // Deserialize using field names (should match snake_case YAML)
        let result: Result<AppFullSettings, ConfigError> = config.clone().try_deserialize();
        if let Err(e) = &result {
             error!("Failed to deserialize AppFullSettings from {:?}: {}", settings_path, e);
             // Log raw value for debugging
             match config.try_deserialize::<Value>() { // config is still available here as the first try_deserialize consumed a clone
                 Ok(raw_value) => error!("Raw settings structure from YAML: {:?}", raw_value),
                 Err(val_err) => error!("Failed to deserialize into raw Value as well: {:?}", val_err),
             }
        }
        result
    }

    // Save method for AppFullSettings, ensuring snake_case YAML output
    pub fn save(&self) -> Result<(), String> {
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));
        debug!("Saving AppFullSettings to YAML file: {:?}", settings_path);

        // Serialize self using the custom Serialize impl which converts keys to snake_case
        let yaml = serde_yaml::to_string(&self)
            .map_err(|e| format!("Failed to serialize AppFullSettings to YAML: {}", e))?;

        std::fs::write(&settings_path, yaml)
            .map_err(|e| format!("Failed to write settings file {:?}: {}", settings_path, e))?;
        debug!("Successfully saved AppFullSettings to {:?}", settings_path);
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    // mod feature_access_test;
}