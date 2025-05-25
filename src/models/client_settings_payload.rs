use serde::Deserialize;

// Consistent camelCase for client JSON interaction

// --- MovementAxes DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientMovementAxes {
    pub horizontal: Option<i32>,
    pub vertical: Option<i32>,
}

// --- XR Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientXRSettings {
    pub enabled: Option<bool>,
    pub mode: Option<String>, // Maps to client's displayMode or server's mode
    pub room_scale: Option<f32>,
    pub space_type: Option<String>,
    pub quality: Option<String>,
    
    pub enable_hand_tracking: Option<bool>, // Mapped from client's handTracking
    pub hand_mesh_enabled: Option<bool>,
    pub hand_mesh_color: Option<String>,
    pub hand_mesh_opacity: Option<f32>,
    pub hand_point_size: Option<f32>,
    pub hand_ray_enabled: Option<bool>,
    pub hand_ray_color: Option<String>,
    pub hand_ray_width: Option<f32>,
    
    pub gesture_smoothing: Option<f32>,
    pub enable_haptics: Option<bool>, // Mapped from client's enableHaptics
    
    pub drag_threshold: Option<f32>,
    pub pinch_threshold: Option<f32>,
    pub rotation_threshold: Option<f32>,
    
    pub interaction_radius: Option<f32>, // Mapped from client's interactionDistance
    
    pub movement_speed: Option<f32>,
    pub dead_zone: Option<f32>,
    pub movement_axes: Option<ClientMovementAxes>,
    
    pub enable_light_estimation: Option<bool>,
    pub enable_plane_detection: Option<bool>,
    pub enable_scene_understanding: Option<bool>,
    pub plane_color: Option<String>,
    pub plane_opacity: Option<f32>,
    pub plane_detection_distance: Option<f32>,
    pub show_plane_overlay: Option<bool>,
    pub snap_to_floor: Option<bool>,
    
    pub enable_passthrough_portal: Option<bool>,
    pub passthrough_opacity: Option<f32>,
    pub passthrough_brightness: Option<f32>,
    pub passthrough_contrast: Option<f32>,
    pub portal_size: Option<f32>,
    pub portal_edge_color: Option<String>,
    pub portal_edge_width: Option<f32>,

    pub controller_model: Option<String>,   // Mapped from client's controllerModel
    pub render_scale: Option<f32>,          // Mapped from client's renderScale
    pub locomotion_method: Option<String>,  // Mapped from client's locomotionMethod
    pub teleport_ray_color: Option<String>, // Mapped from client's teleportRayColor
    pub controller_ray_color: Option<String>,// Mapped from client's controllerRayColor
    // Note: client's 'displayMode' should map to 'mode' in this DTO.
}

// --- Node Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientNodeSettings {
    pub base_color: Option<String>,
    pub metalness: Option<f32>,
    pub opacity: Option<f32>,
    pub roughness: Option<f32>,
    pub node_size: Option<f32>, // Changed from size_range: Option<Vec<f32>>
    pub quality: Option<String>, // "low" | "medium" | "high"
    pub enable_instancing: Option<bool>,
    pub enable_hologram: Option<bool>,
    pub enable_metadata_shape: Option<bool>,
    pub enable_metadata_visualisation: Option<bool>,
}

// --- Edge Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientEdgeSettings {
    pub arrow_size: Option<f32>,
    pub base_width: Option<f32>,
    pub color: Option<String>,
    pub enable_arrows: Option<bool>,
    pub opacity: Option<f32>,
    pub width_range: Option<Vec<f32>>, // Client sends [number, number]
    pub quality: Option<String>, // "low" | "medium" | "high"
    // Fields from client's EdgeSettings not in server's EdgeSettings (src/config/mod.rs)
    // These will be deserialized if present in JSON due to no deny_unknown_fields
    // but won't map to AppFullSettings unless explicitly handled.
    pub enable_flow_effect: Option<bool>, 
    pub flow_speed: Option<f32>,
    pub flow_intensity: Option<f32>,
    pub glow_strength: Option<f32>,
    pub distance_intensity: Option<f32>,
    pub use_gradient: Option<bool>,
    pub gradient_colors: Option<Vec<String>>, // Client sends [string, string]
}

// --- Physics Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientPhysicsSettings {
    pub attraction_strength: Option<f32>,
    pub bounds_size: Option<f32>,
    pub collision_radius: Option<f32>,
    pub damping: Option<f32>,
    pub enable_bounds: Option<bool>,
    pub enabled: Option<bool>,
    pub iterations: Option<u32>,
    pub max_velocity: Option<f32>,
    pub repulsion_strength: Option<f32>,
    pub spring_strength: Option<f32>,
    pub repulsion_distance: Option<f32>,
    pub mass_scale: Option<f32>,
    pub boundary_damping: Option<f32>,
}

// --- Rendering Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientRenderingSettings {
    pub ambient_light_intensity: Option<f32>,
    pub background_color: Option<String>,
    pub directional_light_intensity: Option<f32>,
    pub enable_ambient_occlusion: Option<bool>,
    pub enable_antialiasing: Option<bool>,
    pub enable_shadows: Option<bool>,
    pub environment_intensity: Option<f32>,
    // Fields from client's RenderingSettings not in server's (src/config/mod.rs)
    pub shadow_map_size: Option<String>, 
    pub shadow_bias: Option<f32>,
    pub context: Option<String>, // "desktop" | "ar"
}

// --- Animation Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientAnimationSettings {
    pub enable_motion_blur: Option<bool>,
    pub enable_node_animations: Option<bool>,
    pub motion_blur_strength: Option<f32>,
    pub selection_wave_enabled: Option<bool>,
    pub pulse_enabled: Option<bool>,
    pub pulse_speed: Option<f32>,
    pub pulse_strength: Option<f32>,
    pub wave_speed: Option<f32>,
}

// --- Label Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientLabelSettings {
    pub desktop_font_size: Option<f32>,
    pub enable_labels: Option<bool>,
    pub text_color: Option<String>,
    pub text_outline_color: Option<String>,
    pub text_outline_width: Option<f32>,
    pub text_resolution: Option<u32>,
    pub text_padding: Option<f32>,
    pub billboard_mode: Option<String>, // "camera" | "vertical" (client) vs "camera" | "fixed" | "horizontal" (server, needs mapping)
}

// --- Bloom Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientBloomSettings {
    pub edge_bloom_strength: Option<f32>,
    pub enabled: Option<bool>,
    pub environment_bloom_strength: Option<f32>,
    pub node_bloom_strength: Option<f32>,
    pub radius: Option<f32>,
    pub strength: Option<f32>,
    // Field from client's BloomSettings not in server's (src/config/mod.rs)
    pub threshold: Option<f32>,
}

// --- Hologram Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientHologramSettings {
    pub ring_count: Option<u32>,
    pub ring_color: Option<String>,
    pub ring_opacity: Option<f32>,
    pub sphere_sizes: Option<Vec<f32>>, // Client sends [number, number]
    pub ring_rotation_speed: Option<f32>,
    pub enable_buckminster: Option<bool>,
    pub buckminster_size: Option<f32>,
    pub buckminster_opacity: Option<f32>,
    pub enable_geodesic: Option<bool>,
    pub geodesic_size: Option<f32>,
    pub geodesic_opacity: Option<f32>,
    pub enable_triangle_sphere: Option<bool>,
    pub triangle_sphere_size: Option<f32>,
    pub triangle_sphere_opacity: Option<f32>,
    pub global_rotation_speed: Option<f32>,
}

// --- Camera Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientCameraPosition {
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub z: Option<f32>,
}
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientCameraSettings {
    pub fov: Option<f32>,
    pub near: Option<f32>,
    pub far: Option<f32>,
    pub position: Option<ClientCameraPosition>,
    pub look_at: Option<ClientCameraPosition>,
}


// --- Visualisation Settings DTO (Aggregator) ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientVisualisationSettings {
    pub nodes: Option<ClientNodeSettings>,
    pub edges: Option<ClientEdgeSettings>,
    pub physics: Option<ClientPhysicsSettings>,
    pub rendering: Option<ClientRenderingSettings>,
    pub animations: Option<ClientAnimationSettings>,
    pub labels: Option<ClientLabelSettings>,
    pub bloom: Option<ClientBloomSettings>,
    pub hologram: Option<ClientHologramSettings>,
    pub camera: Option<ClientCameraSettings>, // Client can send camera settings
}

// --- WebSocket Settings DTO (from client/src/features/settings/config/settings.ts) ---
// This can reuse src/config/mod.rs::ClientWebSocketSettings if it's identical
// For clarity, defining it here based on client's definition.
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientPayloadWebSocketSettings {
    pub reconnect_attempts: Option<u32>,
    pub reconnect_delay: Option<u64>, // TS number can be u64
    pub binary_chunk_size: Option<usize>, // TS number can be usize
    pub compression_enabled: Option<bool>,
    pub compression_threshold: Option<usize>,
    pub update_rate: Option<u32>,
}

// --- Debug Settings DTO (from client/src/features/settings/config/settings.ts) ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientPayloadDebugSettings {
    pub enabled: Option<bool>,
    pub enable_data_debug: Option<bool>,
    pub enable_websocket_debug: Option<bool>,
    pub log_binary_headers: Option<bool>,
    pub log_full_json: Option<bool>,
    // Fields from client's DebugSettings not in server's DebugSettings (src/config/mod.rs)
    pub enable_physics_debug: Option<bool>, 
    pub enable_node_debug: Option<bool>,
    pub enable_shader_debug: Option<bool>,
    pub enable_matrix_debug: Option<bool>,
    pub enable_performance_debug: Option<bool>,
    // Note: log_level and log_format are server-side only, not expected from client payload.
}

// --- System Settings DTO (Aggregator) ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientSystemSettings {
    pub websocket: Option<ClientPayloadWebSocketSettings>,
    pub debug: Option<ClientPayloadDebugSettings>,
    pub persist_settings: Option<bool>,
    pub custom_backend_url: Option<String>,
}

// --- Auth Settings DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientAuthSettings {
    pub enabled: Option<bool>,
    pub provider: Option<String>, // "nostr" | string
    pub required: Option<bool>,
}

// --- AI Service Settings DTOs ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientRagFlowSettings {
    pub api_key: Option<String>,
    pub agent_id: Option<String>,
    pub api_base_url: Option<String>,
    pub timeout: Option<u64>, // TS number
    pub max_retries: Option<u32>, // TS number
    pub chat_id: Option<String>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientPerplexitySettings {
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub api_url: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub timeout: Option<u64>,
    pub rate_limit: Option<u32>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientOpenAISettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Option<u64>,
    pub rate_limit: Option<u32>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientKokoroSettings {
    pub api_url: Option<String>,
    pub default_voice: Option<String>,
    pub default_format: Option<String>,
    pub default_speed: Option<f32>,
    pub timeout: Option<u64>,
    pub stream: Option<bool>,
    pub return_timestamps: Option<bool>,
    pub sample_rate: Option<u32>,
}


// --- Top-Level Client Settings Payload DTO ---
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientSettingsPayload {
    pub visualisation: Option<ClientVisualisationSettings>,
    pub system: Option<ClientSystemSettings>,
    pub xr: Option<ClientXRSettings>,
    pub auth: Option<ClientAuthSettings>,
    pub ragflow: Option<ClientRagFlowSettings>,
    pub perplexity: Option<ClientPerplexitySettings>,
    pub openai: Option<ClientOpenAISettings>,
    pub kokoro: Option<ClientKokoroSettings>,
}