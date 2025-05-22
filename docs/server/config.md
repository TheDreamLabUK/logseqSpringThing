# Configuration Architecture

## Overview
The configuration module manages application settings, environment variables, and feature flags.

## Settings Management

### Core Structure: `AppFullSettings`
The primary configuration struct is `AppFullSettings`, defined in `src/config/mod.rs`. This struct aggregates all server-side and client-facing settings, organized into logical categories.

```rust
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,
    pub system: ServerSystemConfigFromFile,
    pub xr: XRSettings,
    pub auth: AuthSettings,
    pub ragflow: Option<RagFlowSettings>,
    pub perplexity: Option<PerplexitySettings>,
    pub openai: Option<OpenAISettings>,
    pub kokoro: Option<KokoroSettings>,
}
```

#### Nested Settings Structures:
-   **`VisualisationSettings`**: Configures 3D graph rendering, physics, animations, labels, bloom, and hologram effects.
    -   `nodes: NodeSettings`
    -   `edges: EdgeSettings`
    -   `physics: PhysicsSettings`
    -   `rendering: RenderingSettings`
    -   `animations: AnimationSettings`
    -   `labels: LabelSettings`
    -   `bloom: BloomSettings`
    -   `hologram: HologramSettings`
-   **`ServerSystemConfigFromFile`**: Contains core server system settings.
    -   `network: NetworkSettings` (e.g., `bind_address`, `port`, `enable_tls`)
    -   `websocket: ServerFullWebSocketSettings` (e.g., `binary_chunk_size`, `update_rate`, `compression_enabled`)
    -   `security: SecuritySettings` (e.g., `allowed_origins`, `session_timeout`)
    -   `debug: DebugSettings` (e.g., `enabled`, `log_level`, `log_full_json`)
    -   `persist_settings: bool` (whether to save settings changes to disk)
-   **`XRSettings`**: Configures WebXR-specific parameters.
-   **`AuthSettings`**: Configures authentication provider settings.
-   **`RagFlowSettings`**: Configuration for the RAGFlow AI service.
-   **`PerplexitySettings`**: Configuration for the Perplexity AI service.
-   **`OpenAISettings`**: Configuration for OpenAI services (e.g., TTS).
-   **`KokoroSettings`**: Configuration for the Kokoro AI service (if integrated).

### Environment Loading
Settings are loaded from a YAML file (defaulting to `/app/settings.yaml`) and can be overridden by environment variables. The `config` crate is used for this hierarchical loading.

```rust
impl AppFullSettings {
    pub fn new() -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok(); // Loads .env file
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));

        let builder = config::ConfigBuilder::<config::builder::DefaultState>::default()
            .add_source(config::File::from(settings_path.clone()).required(true))
            .add_source(
                config::Environment::default()
                    .separator("_") // e.g., SYSTEM_NETWORK_PORT
                    .list_separator(",")
            );
        builder.build()?.try_deserialize()
    }
}
```

## Feature Access and Permissions

Instead of a separate `FeatureFlags` struct, feature access is managed by the `FeatureAccess` struct in `src/config/feature_access.rs`. This struct determines which features are available to different user types (unauthenticated, regular authenticated, power users).

```rust
pub struct FeatureAccess {
    pub power_users: Vec<String>, // List of Nostr pubkeys for power users
    // ... other feature-specific access controls
}

impl FeatureAccess {
    pub fn is_power_user(&self, pubkey: &str) -> bool;
    pub fn can_sync_settings(&self, pubkey: &str) -> bool;
    pub fn has_feature_access(&self, pubkey: &str, feature: &str) -> bool;
    pub fn get_available_features(&self, pubkey: &str) -> Vec<String>;
}
```

## Security Settings

Security-related configurations are part of the `SecuritySettings` struct nested within `ServerSystemConfigFromFile`.

```rust
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
```
Rate limiting is configured within `NetworkSettings` (e.g., `rate_limit_requests`, `rate_limit_window`) and applied at the network layer.

## Implementation Details

### Loading Hierarchy
1.  **YAML Configuration File**: Primary source of settings (`/app/settings.yaml`).
2.  **Environment Variables**: Overrides values from the YAML file (e.g., `APP_NETWORK_PORT`).
3.  **Default Values**: Provided by `Default` implementations for structs if not specified elsewhere.

### Validation Rules
Settings are validated during deserialization by the `config` crate. Custom validation logic can be implemented within `AppFullSettings` or its sub-structs if needed.

### Hot Reload
The current implementation does not support hot reloading of configuration. Changes to `settings.yaml` or environment variables require a server restart to take effect.

### Saving Settings
`AppFullSettings` implements a `save()` method to persist the current settings state back to the `settings.yaml` file. This is used when power users modify global settings via the UI.

```rust
impl AppFullSettings {
    pub fn save(&self) -> Result<(), String> {
        // ... serialization to YAML and file write logic
    }
}
```