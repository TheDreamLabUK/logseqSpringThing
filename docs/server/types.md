# Types Architecture

## Overview
The types module defines core data structures, type aliases, and common enums used throughout the application.

## Core Types

### Graph Types
```rust
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: MetadataStore,
    pub id_to_metadata: HashMap<String, String>, // #[serde(skip)]
}

/// Represents a 3D vector.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)] // Pod, Zeroable are for bytemuck
pub struct Vec3Data {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Binary node data structure for efficient transmission and GPU processing.
/// Mass, flags, and padding are server-side only and not typically transmitted.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)] // Pod, Zeroable are for bytemuck
pub struct BinaryNodeData {
    pub position: Vec3Data,
    pub velocity: Vec3Data,
    pub mass: u8,
    pub flags: u8,
    pub padding: [u8; 2], // For alignment
}

/// Represents a node in the graph.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: String, // Unique numeric ID for binary protocol, string for JSON
    pub metadata_id: String,  // Original filename or unique identifier for metadata lookup
    pub label: String, // Display label, often same as metadata_id
    pub data: BinaryNodeData, // Contains position, velocity, mass, flags

    // Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>, // Additional string-based metadata
    
    #[serde(skip)] // Not serialized directly, but value might be in `metadata` map
    pub file_size: u64,

    // Rendering properties (optional)
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>, // Could influence physics or rendering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>, // For grouping nodes visually or logically
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<HashMap<String, String>>, // Arbitrary user-defined data
}

/// Edge structure representing connections between nodes.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub source: String, // ID of the source node
    pub target: String, // ID of the target node
    pub weight: f32,    // Strength or importance of the connection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>, // Categorizes the edge (e.g., "hyperlink", "dependency")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>, // Additional data about the edge
}
```

### Simulation Types
```rust
pub enum SimulationPhase {
    Dynamic,
    Static,
    Paused,
}

pub enum SimulationMode {
    Local,
    Remote,
    Hybrid,
}
```

## Models

### Settings Models
```rust
// ### Settings Models
// This section outlines the primary configuration structures used by the application.
// Settings are loaded from `settings.yaml` on the server and can be partially
// exposed to or mirrored by the client.

// --- Core Client Visualisation & Interaction Settings ---
// These are used by both AppFullSettings (server) and Settings (client).
// Fields generally align with client-side expectations (often camelCase for JSON).

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MovementAxes {
    pub horizontal: i32,
    pub vertical: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NodeSettings {
    pub base_color: String,
    pub metalness: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub node_size: f32,
    pub quality: String,
    pub enable_instancing: bool,
    pub enable_hologram: bool,
    pub enable_metadata_shape: bool,
    pub enable_metadata_visualisation: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
pub struct BloomSettings {
    pub edge_bloom_strength: f32,
    pub enabled: bool,
    pub environment_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub radius: f32,
    pub strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

/// Top-level structure for all client-side visual rendering and interaction settings.
/// Used by both server-side `AppFullSettings` and client-facing `Settings`.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // Assuming sub-fields handle their casing for JSON
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

// --- Server-Side Configuration (`AppFullSettings` and components) ---
// These structs define the complete server configuration, typically loaded from snake_case YAML.

/// Server-side network configuration (part of `ServerSystemConfigFromFile`).
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// Field names match snake_case YAML
pub struct ServerNetworkConfig {
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

/// Server-side WebSocket configuration (part of `ServerSystemConfigFromFile`).
#[derive(Debug, Serialize, Deserialize, Clone)]
// Field names match snake_case YAML
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
impl Default for ServerFullWebSocketSettings { /* ... see src/config/mod.rs for defaults ... */ }


/// Server-side security configuration. Also used by `ProtectedSettings`.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// Field names match snake_case YAML
pub struct ServerSecurityConfig {
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

/// Debug settings, used by both server and client configurations.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON if client uses it directly
// YAML loading expects snake_case for log_level, log_format
pub struct DebugSettings {
    pub enabled: bool,
    pub enable_data_debug: bool,
    pub enable_websocket_debug: bool,
    pub log_binary_headers: bool,
    pub log_full_json: bool,
    pub log_level: String, // snake_case in YAML
    pub log_format: String, // snake_case in YAML
}

/// Aggregates server-specific system configurations loaded from `settings.yaml`.
#[derive(Debug, Deserialize, Clone)] // Only Deserialize for loading YAML
// Field names match snake_case YAML
pub struct ServerSystemConfigFromFile {
    pub network: ServerNetworkConfig,
    pub websocket: ServerFullWebSocketSettings,
    pub security: ServerSecurityConfig,
    pub debug: DebugSettings,
    #[serde(default)]
    pub persist_settings: bool,
}

// --- Shared & Client-Facing Configuration Components ---
// These are used by AppFullSettings (server), Settings (client), and sometimes ProtectedSettings.
// They often use camelCase for JSON serialization to the client.

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON, YAML might be snake_case
pub struct XRSettings {
    pub mode: String,
    pub room_scale: f32,
    pub space_type: String,
    pub quality: String,
    #[serde(alias = "handTracking")]
    pub enable_hand_tracking: bool,
    pub hand_mesh_enabled: bool,
    // ... many other XR fields, see src/config/mod.rs ...
    #[serde(default)]
    pub enabled: Option<bool>, // TS 'enabled' field
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON
pub struct AuthSettings {
    pub enabled: bool,
    pub provider: String,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON
pub struct RagFlowSettings {
    #[serde(default)] pub api_key: Option<String>,
    #[serde(default)] pub agent_id: Option<String>,
    // ... other fields ...
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON
pub struct PerplexitySettings {
    #[serde(default)] pub api_key: Option<String>,
    #[serde(default)] pub model: Option<String>,
    // ... other fields ...
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON
pub struct OpenAISettings {
    #[serde(default)] pub api_key: Option<String>,
    // ... other fields ...
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
// #[serde(rename_all = "camelCase")] // For JSON
pub struct KokoroSettings {
    #[serde(default)] pub api_url: Option<String>,
    // ... other fields ...
}

/// The main application settings structure used server-side, loaded from `settings.yaml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
// YAML is snake_case. Custom Serialize impl in config/mod.rs ensures snake_case output.
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings, // Uses sub-structs that might be camelCase for client
    pub system: ServerSystemConfigFromFile,   // Contains server-specific snake_case structs
    pub xr: XRSettings,                       // Mixed casing needs careful handling for YAML/JSON
    pub auth: AuthSettings,
    #[serde(default)] pub ragflow: Option<RagFlowSettings>,
    #[serde(default)] pub perplexity: Option<PerplexitySettings>,
    #[serde(default)] pub openai: Option<OpenAISettings>,
    #[serde(default)] pub kokoro: Option<KokoroSettings>,
}

// --- Client-Facing Settings (`Settings` and components) ---
// These define the structure the client application expects, typically as camelCase JSON.

/// Client-specific WebSocket settings.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientWebSocketSettings {
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub binary_chunk_size: usize,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub update_rate: u32,
}
impl Default for ClientWebSocketSettings { /* ... see src/config/mod.rs ... */ }

/// Client-facing system settings (subset of server system settings).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientSystemSettings {
    pub websocket: ClientWebSocketSettings,
    pub debug: DebugSettings, // DebugSettings needs consistent camelCase for client
    #[serde(default)]
    pub persist_settings: bool,
}
impl Default for ClientSystemSettings { /* ... see src/config/mod.rs ... */ }


/// Defines the settings structure as expected and used by the client application.
/// This is typically serialized as JSON with camelCase keys.
/// This replaces the old `UserSettings` documentation.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Settings { // This is the primary client-facing Settings struct
    pub visualisation: VisualisationSettings,
    pub system: ClientSystemSettings,
    pub xr: XRSettings,
    pub auth: AuthSettings,
    #[serde(default)] pub ragflow: Option<RagFlowSettings>,
    #[serde(default)] pub perplexity: Option<PerplexitySettings>,
    #[serde(default)] pub openai: Option<OpenAISettings>,
    #[serde(default)] pub kokoro: Option<KokoroSettings>,
}


// --- Protected Server Settings (Managed separately, not from settings.yaml) ---
// These are sensitive settings, often stored in a separate JSON file and managed via API.
// Typically use camelCase for JSON.

/// API keys for various services, part of `ProtectedSettings`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeys {
    pub perplexity: Option<String>,
    pub openai: Option<String>,
    pub ragflow: Option<String>,
}

/// Information about a Nostr user, part of `ProtectedSettings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NostrUser {
    pub pubkey: String,
    pub npub: String,
    pub is_power_user: bool,
    pub api_keys: ApiKeys,
    pub last_seen: i64, // Unix timestamp
    pub session_token: Option<String>,
}

/// Network settings specific to the `ProtectedSettings` context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProtectedNetworkConfig {
    pub bind_address: String,
    pub domain: String,
    pub port: u16,
    pub enable_http2: bool,
    pub enable_tls: bool,
    pub min_tls_version: String,
    pub max_request_size: usize,
    pub enable_rate_limiting: bool,
    pub rate_limit_requests: u32,
    pub rate_limit_window: u32,
    pub tunnel_id: String,
}
// Note: ServerSecurityConfig is reused for ProtectedSettings.security

/// WebSocket server settings specific to the `ProtectedSettings` context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProtectedWebSocketServerConfig {
    pub max_connections: usize,
    pub max_message_size: usize,
    pub url: String, // URL for the WebSocket server if managed/proxied
}

/// Contains sensitive settings not typically stored in the main `settings.yaml`.
/// Includes user-specific data, API keys, and certain operational parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ProtectedSettings {
    pub network: ProtectedNetworkConfig,
    pub security: ServerSecurityConfig, // Reusing ServerSecurityConfig
    pub websocket_server: ProtectedWebSocketServerConfig,
    pub users: std::collections::HashMap<String, NostrUser>,
    pub default_api_keys: ApiKeys,
}
```

### Metadata Models
```rust
pub struct MetadataManager {
    pub files: HashMap<String, FileMetadata>,
    pub relationships: Vec<Relationship>,
    pub statistics: Statistics,
}

pub struct FileMetadata {
    pub name: String,
    pub path: String,
    pub size: usize,
    pub node_id: String,
    pub last_modified: DateTime<Utc>,
    pub file_type: FileType,
}

pub enum FileType {
    Markdown,
    Image,
    Pdf,
    // ... other file types
}
```

## Error Types

### Service Errors
```rust
pub enum ServiceError {
    IO(std::io::Error),
    Graph(String),
    Config(String),
    GitHub(String),
    AI(String),
    Speech(SpeechError),
    // ... other service-specific errors
}

impl From<std::io::Error> for ServiceError {
    fn from(err: std::io::Error) -> Self {
        ServiceError::IO(err)
    }
}

impl From<reqwest::Error> for ServiceError {
    fn from(err: reqwest::Error) -> Self {
        ServiceError::GitHub(format!("GitHub API error: {}", err))
    }
}

## Speech Types

### Speech Commands
```rust
pub enum SpeechCommand {
    StartAudioStream,
    ProcessAudioChunk(Vec<u8>),
    EndAudioStream,
    // ... other speech commands
}
```
- Defines commands for controlling the speech processing pipeline, including starting/ending audio streams and processing audio chunks.

### Speech Errors
```rust
pub enum SpeechError {
    SttError(String),
    TtsError(String),
    RagFlowError(String),
    WebSocketError(String),
    InternalError(String),
    // ... other speech-specific errors
}
```
- Defines error types specific to speech processing, covering STT, TTS, RAGFlow interactions, and WebSocket communication.

## AI Service Models

The application interacts with various AI services, each potentially having its own specific request and response structures. This section outlines key data structures used for these interactions, as defined within the application. If a service utilizes an external SDK (e.g., `async-openai`), its service documentation will refer to the types from that SDK, and they will not be redefined here.

### RAGFlow Service Models
These structures are used for communication with the RAGFlow service.

```rust
/// Request to the RAGFlow service for a completion.
/// (Simplified from `src/services/ragflow_service.rs::CompletionRequest`)
#[derive(Debug, Serialize)]
pub struct RagflowCompletionRequest {
    pub question: String,
    pub stream: bool,
    pub session_id: Option<String>,
    // pub user_id: Option<String>, // Optional
    // pub sync_dsl: Option<bool>, // Optional
}

/// Represents the data part of a RAGFlow completion response.
/// (Simplified from `src/services/ragflow_service.rs::CompletionData`)
#[derive(Debug, Deserialize)]
pub struct RagflowCompletionData {
    pub answer: Option<String>,
    pub reference: Option<serde_json::Value>, // Can be complex structured data
    pub id: Option<String>, // Message ID
    pub session_id: Option<String>,
}

/// Represents a message within a RAGFlow session history.
/// (From `src/services/ragflow_service.rs::Message`)
#[derive(Debug, Deserialize)]
pub struct RagflowMessage {
    pub role: String, // e.g., "user", "assistant"
    pub content: String,
}

/// Data part of a RAGFlow session creation/retrieval response.
/// (Simplified from `src/services/ragflow_service.rs::SessionData`)
#[derive(Debug, Deserialize)]
pub struct RagflowSessionData {
    pub id: String, // Session ID
    pub message: Option<Vec<RagflowMessage>>, // History
}
```
- The RAGFlow service handles chat sessions and completions, potentially involving streaming responses.

### Perplexity Service Models
These structures are used for communication with the Perplexity AI service.

```rust
/// Request to the Perplexity service.
/// (From `src/services/perplexity_service.rs::QueryRequest`)
#[derive(Debug, Serialize)]
pub struct PerplexityQueryRequest {
    pub query: String,
    pub conversation_id: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// Response from the Perplexity service.
/// (From `src/services/perplexity_service.rs::PerplexityResponse`)
#[derive(Debug, Deserialize)]
pub struct PerplexityResponse {
    pub content: String,
    pub link: String, // Link to Perplexity's source/page for the query
}
```
- The Perplexity service is used for direct queries and can also process files.
```

### API Errors
```rust
pub enum APIError {
    NotFound(String),
    Unauthorized,
    RateLimit,
    Internal(String),
}
```

## Type Aliases

### Common Aliases
```rust
pub type Result<T> = std::result::Result<T, Error>;
pub type NodeMap = HashMap<String, Node>;
pub type MetadataMap = HashMap<String, Metadata>;
pub type SafeAppState = Arc<AppState>;
pub type SafeSettings = Arc<RwLock<Settings>>;
pub type SafeMetadataManager = Arc<RwLock<MetadataManager>>;
```

## Constants

### System Constants
```rust
pub const MAX_NODES: usize = 10000;
pub const DEFAULT_BATCH_SIZE: usize = 100;
pub const CACHE_DURATION: Duration = Duration::from_secs(3600);
```

### Configuration Constants
```rust
pub const DEFAULT_PORT: u16 = 8080;
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const API_VERSION: &str = "v1";