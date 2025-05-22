# Configuration Architecture

## Overview
The configuration module manages application settings, environment variables, and feature flags.

## Settings Management

### Core Structure
```rust
pub struct Settings {
    pub server: ServerConfig,
    pub visualisation: VisualisationConfig,
    pub github: GitHubServiceConfig,
    pub security: SecurityConfig,
    pub ai: AIServiceConfig,
    pub file_service: FileServiceConfig,
}
```

### Environment Loading
```rust
impl Settings {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv().ok();
        // Load configuration from environment
    }
}
```

## Feature Flags

### Configuration
```rust
pub struct FeatureFlags {
    pub gpu_enabled: bool,
    pub websocket_enabled: bool,
    pub metrics_enabled: bool,
    pub ai_enabled: bool,
}
```

### Dynamic Updates
```rust
impl FeatureFlags {
    pub fn update_from_env(&mut self)
    pub fn is_feature_enabled(&self, feature: &str) -> bool
}
```

## Environment Configuration

### Server Settings
```rust
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}
```

### API Configuration
```rust
pub struct GitHubServiceConfig {
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub token: Option<String>,
}

pub struct AIServiceConfig {
    pub perplexity_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub ragflow_api_url: Option<String>,
}

pub struct FileServiceConfig {
    pub base_path: String,
    pub max_file_size_mb: u64,
}
```

## Security Settings

### Authentication
```rust
pub struct SecurityConfig {
    pub nostr_enabled: bool,
    pub admin_pubkeys: Vec<String>,
    pub rate_limit_per_minute: u64,
}
```

### Rate Limiting
```rust
// Rate limiting is now part of SecurityConfig
```

## Implementation Details

### Loading Hierarchy
1. Environment variables
2. Configuration files
3. Default values

### Validation Rules
```rust
impl Settings {
    pub fn validate(&self) -> Result<(), ValidationError>
}
```

### Hot Reload
```rust
pub async fn reload_config() -> Result<(), ConfigError>