# Server-Side Data Models

This document outlines the core data structures (models) used on the server-side of the LogseqXR application. These models define how data is structured, stored, and manipulated.

## Simulation Parameters (`SimulationParams`)

Defines parameters for the physics-based graph layout simulation.

### Core Structure (from [`src/models/simulation_params.rs`](../../src/models/simulation_params.rs))
```rust
// In src/models/simulation_params.rs
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub struct SimulationParams {
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion_strength: f32, // Note: field name might be repulsion_strength
    pub damping: f32,
    pub max_repulsion_distance: f32,
    // pub viewport_bounds: f32, // This specific field might not exist; bounds are often implicitly handled or part of PhysicsSettings
    pub mass_scale: f32,
    pub boundary_damping: f32,
    pub enable_bounds: bool, // This is usually part of PhysicsSettings in AppFullSettings
    pub time_step: f32,
    // pub phase: SimulationPhase, // If used, SimulationPhase would be an enum
    // pub mode: SimulationMode,   // If used, SimulationMode would be an enum
    // Other fields like collision_radius, max_velocity, repulsion_distance might be here or in PhysicsSettings
}
```
Note: Some physics parameters like `gravity_strength` and `center_attraction_strength` are part of `PhysicsSettings` within the main `AppFullSettings` and are used to influence the `SimulationParams` at runtime, but are not direct fields of this struct.

### Usage
-   Configuring the physics engine for graph layout.
-   Allowing real-time adjustment of simulation behavior.
-   Defining boundary conditions for the simulation space.

## UI Settings (`UserSettings` and `UISettings`)

The server defines two main structures for managing UI-related settings:

1.  **`UserSettings`** (from [`src/models/user_settings.rs`](../../src/models/user_settings.rs)): This structure is user-specific and primarily stores a user's `pubkey` and their personalized `UISettings` (which itself contains `visualisation`, `system`, and `xr` settings relevant to the client). It's used for persisting individual user preferences.

    ```rust
    // In src/models/user_settings.rs
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserSettings {
        pub pubkey: String,
        pub settings: UISettings, // The actual client-facing settings structure
        pub last_modified: i64, // Unix timestamp
    }
    ```

2.  **`UISettings`** (from [`src/models/ui_settings.rs`](../../src/models/ui_settings.rs)): This structure represents the actual set of UI configurations that are sent to the client (serialized as camelCase JSON). It's derived from the global `AppFullSettings` for public/default views or from a specific user's `UserSettings`.

    ```rust
    // In src/models/ui_settings.rs
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct UISettings {
        pub visualisation: VisualisationSettings, // Sourced from AppFullSettings.visualisation
        pub system: UISystemSettings,             // Contains client-relevant parts of AppFullSettings.system
        pub xr: XRSettings,                       // Sourced from AppFullSettings.xr
        // Note: AuthSettings from AppFullSettings are used server-side; client gets tokens/features.
        // AI service configurations (like API keys) are NOT part of UISettings.
        // Client interacts with AI services via API endpoints; server uses ProtectedSettings for keys.
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct UISystemSettings {
        // Contains only the client-relevant parts of ServerSystemConfigFromFile
        pub websocket: ClientWebSocketSettings, // Derived from ServerFullWebSocketSettings
        pub debug: DebugSettings,               // Client-safe debug flags
        // persistSettings and customBackendUrl are client-side settings,
        // but their server-side counterparts might influence these.
    }
    ```
    -   **Clarification**: `LayoutConfig` and `ThemeConfig` are not distinct top-level structures within `UISettings`.
        -   Theme-related aspects (colors, styles) are primarily part of `VisualisationSettings` (e.g., `visualisation.rendering.backgroundColor`, `visualisation.nodes.baseColor`).
        -   Layout aspects are generally managed client-side or are an emergent property of the physics simulation and camera settings.
    -   **AI Settings**: AI service configurations (API keys, model choices, endpoints) are **not** part of `UISettings` sent to the client. The client interacts with AI services via dedicated API endpoints. The server manages AI API keys and configurations within `AppFullSettings` (for general AI behavior) and `ProtectedSettings` (for sensitive keys, often user-specific).

### Persistence
-   **User-Specific Settings (`UserSettings`)**: Saved to individual YAML files (e.g., `/app/user_settings/<pubkey>.yaml`).
-   **Global/Default Settings (`AppFullSettings` from which `UISettings` can be derived)**: Saved in `settings.yaml`.

## Protected Settings (`ProtectedSettings`)

This structure holds sensitive server-side configurations that are not directly exposed to clients but are used internally by the server.

### Core Structure (from [`src/models/protected_settings.rs`](../../src/models/protected_settings.rs))
```rust
// In src/models/protected_settings.rs
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ProtectedSettings {
    // These fields mirror parts of AppFullSettings.system but are managed separately for security/persistence.
    pub network: ProtectedNetworkConfig, // Contains bind_address, port etc.
    pub security: ServerSecurityConfig,    // Contains allowed_origins, session_timeout etc. (often reuses SecuritySettings from config/mod.rs)
    pub websocket_server: ProtectedWebSocketServerConfig, // Contains server-specific WebSocket settings

    // User management and API keys
    pub users: std::collections::HashMap<String, NostrUser>, // Keyed by Nostr pubkey (hex)
    pub default_api_keys: ApiKeys, // Default API keys for services if no user-specific key
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NostrUser {
    pub pubkey: String, // Hex public key
    pub npub: Option<String>,
    pub is_power_user: bool,
    pub api_keys: Option<ApiKeys>, // User-specific API keys
    pub user_settings_path: Option<String>, // Path to their persisted UserSettings YAML
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeys {
    pub perplexity: Option<String>,
    pub openai: Option<String>,
    pub ragflow: Option<String>,
    // Potentially other AI service keys
}

// ProtectedNetworkConfig, ServerSecurityConfig (may reuse config::SecuritySettings),
// and ProtectedWebSocketServerConfig are also defined in this module or imported.
```

### Features
-   Management of server network configurations.
-   Security policies (CORS, session timeouts).
-   WebSocket server parameters.
-   Storage of Nostr user profiles, including their individual API keys for AI services.
-   Default API keys for services if no user-specific key is available.

## Metadata Store (`MetadataStore` and `Metadata`)

The metadata store is responsible for holding information about each processed file (node) in the knowledge graph.

### Core Structure (from [`src/models/metadata.rs`](../../src/models/metadata.rs))
The `MetadataStore` is a type alias for `HashMap<String, Metadata>`, where the key is typically a unique identifier for the content (e.g., file path or a derived ID).

```rust
// In src/models/metadata.rs
pub type MetadataStore = std::collections::HashMap<String, Metadata>;
```

The `Metadata` struct contains details for each processed file/node:
```rust
// In src/models/metadata.rs
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    pub file_name: String, // Original file name
    pub file_size: u64,    // File size in bytes (ensure type matches actual usage, e.g., u64)
    pub node_size: f64,
    pub hyperlink_count: usize,
    pub sha1: Option<String>, // SHA1 hash of the file content, optional
    pub node_id: String,      // Unique identifier for the graph node (often derived from file_name or path)
    pub last_modified: Option<i64>, // Unix timestamp (seconds), optional

    // AI Service related fields
    pub perplexity_link: Option<String>, // Link to Perplexity discussion/page if available
    pub last_perplexity_process_time: Option<i64>, // Timestamp of last Perplexity processing, optional
    pub topic_counts: Option<std::collections::HashMap<String, usize>>, // Counts of topics/keywords, optional

    // Other potential fields:
    // pub title: Option<String>,
    // pub tags: Option<Vec<String>>,
    // pub content_type: Option<String>, // e.g., "markdown", "pdf"
    // pub created_at: Option<i64>,
}
```
-   The `MetadataStore` itself is a `HashMap`. Relationships between nodes (edges) are typically stored separately in `GraphData` within `GraphService`. Statistics are usually computed on-the-fly or by dedicated analysis processes rather than being stored directly in `MetadataStore`.
-   The `node_size` field is calculated on the server based on file size and stored in the metadata for potential use by the client or other services.

### Operations
-   The `MetadataStore` (as a `HashMap`) supports standard CRUD operations for `Metadata` entries.
-   Relationship management and statistics tracking are typically handled by services like `GraphService` or `FileService` by processing the contents of the `MetadataStore`.

## Implementation Details

### Thread Safety
Shared mutable data structures like `MetadataStore` and settings objects are wrapped in `Arc<RwLock<T>>` within `AppState` to ensure thread-safe access.
```rust
// Example from app_state.rs
// pub metadata: Arc<RwLock<MetadataStore>>,
// pub settings: Arc<RwLock<AppFullSettings>>,
```

### Serialization
Data models are designed to be serializable and deserializable using `serde` for various formats like JSON (for API communication) and YAML (for configuration files). The `#[serde(rename_all = "camelCase")]` attribute is often used for client compatibility.

### Validation
Validation logic is typically implemented within the services that manage these models or during the deserialization process (e.g., using `serde` attributes or custom validation functions).