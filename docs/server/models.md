# Server-Side Data Models

This document outlines the core data structures (models) used on the server-side of the LogseqXR application. These models define how data is structured, stored, and manipulated.

## Simulation Parameters (`SimulationParams`)

Defines parameters for the physics-based graph layout simulation.

### Core Structure (from [`src/models/simulation_params.rs`](../../src/models/simulation_params.rs))
```rust
pub struct SimulationParams {
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion: f32,
    pub damping: f32,
    pub max_repulsion_distance: f32,
    pub viewport_bounds: f32, // May relate to enable_bounds
    pub mass_scale: f32,
    pub boundary_damping: f32,
    pub enable_bounds: bool,
    pub time_step: f32,
    // pub phase: SimulationPhase, // Assuming SimulationPhase is an enum
    // pub mode: SimulationMode,   // Assuming SimulationMode is an enum
    pub gravity_strength: f32,
    pub center_attraction_strength: f32,
}
```

### Usage
-   Configuring the physics engine for graph layout.
-   Allowing real-time adjustment of simulation behavior.
-   Defining boundary conditions for the simulation space.

## UI Settings (`UserSettings` and `UISettings`)

The server defines two main structures for managing UI-related settings:

1.  **`UserSettings`** (from [`src/models/user_settings.rs`](../../src/models/user_settings.rs)): This structure is user-specific and primarily stores a user's `pubkey`, their personalized `UISettings`, and the `last_modified` timestamp. It's used for persisting individual user preferences.

    ```rust
    // From src/models/user_settings.rs
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserSettings {
        pub pubkey: String,
        pub settings: UISettings,
        pub last_modified: i64, // Unix timestamp
    }
    ```

2.  **`UISettings`** (from [`src/models/ui_settings.rs`](../../src/models/ui_settings.rs)): This structure represents the actual set of UI configurations that are sent to the client. It's derived from the global `AppFullSettings` for public/default views or from a specific user's `UserSettings`.

    ```rust
    // From src/models/ui_settings.rs
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct UISettings {
        pub visualisation: VisualisationSettings, // from config/mod.rs
        pub system: UISystemSettings,
        pub xr: XRSettings, // from config/mod.rs
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct UISystemSettings {
        pub websocket: ClientWebSocketSettings, // from config/mod.rs
        pub debug: DebugSettings, // from config/mod.rs
    }
    ```
    -   `LayoutConfig` and `ThemeConfig` are not present as distinct top-level structures within `UISettings`. Theme-related aspects are typically part of `VisualisationSettings` (e.g., `backgroundColor`, `baseColor`) and layout aspects are managed client-side or implicitly through physics and camera settings.
    -   AI settings are not directly part of `UISettings`. The client interacts with AI services via dedicated API endpoints, and AI service configurations (like API keys, models) are managed in the server's `AppFullSettings` and `ProtectedSettings`. Refer to [`src/config/mod.rs`](../../src/config/mod.rs) for detailed AI settings structures like `RagFlowSettings`, `PerplexitySettings`, etc.

### Persistence
-   **User-Specific Settings (`UserSettings`)**: Saved to individual YAML files (e.g., `/app/user_settings/<pubkey>.yaml`).
-   **Global/Default Settings (`AppFullSettings` from which `UISettings` can be derived)**: Saved in `settings.yaml`.

## Protected Settings (`ProtectedSettings`)

This structure holds sensitive server-side configurations that are not directly exposed to clients but are used internally by the server.

### Core Structure (from [`src/models/protected_settings.rs`](../../src/models/protected_settings.rs))
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProtectedSettings {
    pub network: NetworkSettings,
    pub security: SecuritySettings,
    pub websocket_server: WebSocketServerSettings,
    pub users: std::collections::HashMap<String, NostrUser>, // Stores Nostr user profiles including their API keys
    pub default_api_keys: ApiKeys, // Default API keys for unauthenticated access or as fallback
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeys {
    pub perplexity: Option<String>,
    pub openai: Option<String>,
    pub ragflow: Option<String>,
}

// Other structs like NetworkSettings, SecuritySettings, WebSocketServerSettings, NostrUser
// are also defined in protected_settings.rs
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
The `MetadataStore` is a type alias for `HashMap<String, Metadata>`:
```rust
// From src/models/metadata.rs
pub type MetadataStore = HashMap<String, Metadata>;
```

The `Metadata` struct contains details for each file:
```rust
// From src/models/metadata.rs
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    pub file_name: String,
    pub file_size: usize,
    pub node_size: f64, // Potentially derived from file_size or other metrics
    pub hyperlink_count: usize,
    pub sha1: String,
    pub node_id: String, // Unique identifier for the graph node
    pub last_modified: DateTime<Utc>,
    pub perplexity_link: String, // Link to Perplexity discussion/page if available
    pub last_perplexity_process: Option<DateTime<Utc>>, // Timestamp of last Perplexity processing
    pub topic_counts: HashMap<String, usize>, // Counts of topics/keywords in the file
}
```
The `MetadataStore` does not directly contain fields for `relationships` or `statistics` at its top level; these would be derived or managed by services that consume the `MetadataStore`.

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