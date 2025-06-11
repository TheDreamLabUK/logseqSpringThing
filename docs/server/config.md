# Configuration Architecture

## Overview
The configuration module manages application settings, environment variables, and feature flags.

## Settings Management

### Core Structure: `AppFullSettings`
The primary configuration struct is `AppFullSettings`, defined in [`src/config/mod.rs`](../../src/config/mod.rs). This struct aggregates all server-side settings, which are then used to derive client-facing settings (`UISettings`).

```rust
// In src/config/mod.rs
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,
    pub system: ServerSystemConfigFromFile, // Contains network, websocket, security, debug
    pub xr: XRSettings,
    pub auth: AuthSettings,
    // Optional AI Service Configurations
    pub ragflow: Option<RAGFlowConfig>, // Note: Name might be RAGFlowConfig or similar
    pub perplexity: Option<PerplexityConfig>,
    pub openai: Option<OpenAIConfig>,
    pub kokoro: Option<KokoroConfig>,
    pub whisper: Option<WhisperConfig>,
}
```

#### Main Categories in `AppFullSettings`:
-   **`visualisation: VisualisationSettings`**: Configures 3D graph rendering, physics, animations, labels, bloom, and hologram effects. Contains nested structs like `NodeSettings`, `EdgeSettings`, `PhysicsSettings`, etc.
-   **`system: ServerSystemConfigFromFile`**: Contains core server system settings:
    -   `network: NetworkSettings` (e.g., `bind_address`, `port`, `enable_tls`)
    -   `websocket: ServerFullWebSocketSettings` (e.g., `binary_chunk_size`, `update_rate`, `compression_enabled`, `min_update_rate`, `max_update_rate`, `motion_threshold`)
    -   `security: SecuritySettings` (e.g., `allowed_origins`, `session_timeout`)
    -   `debug: DebugSettings` (e.g., `enabled`, `log_level`, `log_full_json`)
-   **`xr: XRSettings`**: Configures WebXR-specific parameters.
-   **`auth: AuthSettings`**: Configures authentication provider settings (e.g., Nostr challenge settings).
-   **Optional AI Services**:
    -   `ragflow: Option<RAGFlowConfig>`
    -   `perplexity: Option<PerplexityConfig>`
    -   `openai: Option<OpenAIConfig>` (may include API keys for various OpenAI services like TTS, STT/Whisper)
    -   `kokoro: Option<KokoroConfig>`

Note: `whisper` is not a top-level configuration in `AppFullSettings`. Whisper STT functionality, if used via OpenAI, would typically have its API key configured within `OpenAIConfig`.

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

Feature access is managed by the `FeatureAccess` struct, typically defined in [`src/config/feature_access.rs`](../../src/config/feature_access.rs). This struct is initialized based on `AuthSettings` (e.g., list of power user pubkeys) from `AppFullSettings`.

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

Security-related configurations are defined in the `SecuritySettings` struct, which is nested within `ServerSystemConfigFromFile` (i.e., `app_full_settings.system.security`).

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
`AppFullSettings` implements a `save(&self, path: &Path) -> Result<(), ConfigError>` method (or similar signature) to persist the current settings state back to the specified YAML file (typically `settings.yaml`). This serialization is done using `serde_yaml` and handles converting the Rust struct (usually in snake_case or as defined by `serde` attributes) to YAML format. This method is invoked when power users modify global settings that need to be persisted.
The `AppFullSettings` struct itself derives `Serialize` and `Deserialize` from `serde` for this purpose.