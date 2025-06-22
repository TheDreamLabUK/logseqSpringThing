# Configuration Guide

## Overview

LogseqSpringThing uses a layered configuration system combining YAML files, environment variables, and runtime settings. This guide covers all configuration options and best practices.

## Configuration Hierarchy

```mermaid
graph TD
    ENV[Environment Variables] --> YAML[YAML Config Files]
    YAML --> RUNTIME[Runtime Settings]
    RUNTIME --> APP[Application State]
    
    style ENV fill:#f9f,stroke:#333,stroke-width:2px
    style YAML fill:#bbf,stroke:#333,stroke-width:2px
    style RUNTIME fill:#bfb,stroke:#333,stroke-width:2px
```

Priority order (highest to lowest):
1. Environment variables
2. Command-line arguments
3. YAML configuration files
4. Default values

## Configuration Files

### Main Configuration: `config.yml`

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  shutdown_timeout: 30

# Database Configuration
database:
  url: "postgres://user:pass@localhost/dbname"
  max_connections: 10
  connection_timeout: 30

# Graph Settings
graph:
  max_nodes: 100000
  simulation:
    repulsion_strength: 100.0
    attraction_strength: 0.01
    centering_strength: 0.01
    damping: 0.9
    time_step: 0.016

# AI Services
ragflow:
  api_key: "${RAGFLOW_API_KEY}"
  base_url: "https://api.ragflow.com"
  timeout: 30
  max_retries: 3

perplexity:
  api_key: "${PERPLEXITY_API_KEY}"
  base_url: "https://api.perplexity.ai"
  model: "mixtral-8x7b-instruct"
  max_tokens: 2048
  temperature: 0.7

# Authentication
auth:
  nostr:
    enabled: true
    relay_urls:
      - "wss://relay.damus.io"
      - "wss://nostr.wine"
  api_keys:
    enabled: true
    rotation_days: 90

# Feature Access
features:
  power_users_enabled: true
  power_user_keys:
    - "${POWER_USER_KEY_1}"
    - "${POWER_USER_KEY_2}"
  
  enabled_features:
    - graph_visualization
    - ai_chat
    - speech_recognition
    - xr_support

# GPU Configuration
gpu:
  enabled: true
  device_id: 0
  fallback_to_cpu: true
  block_size: 256
  max_memory_mb: 4096

# Logging
logging:
  level: "info"
  format: "json"
  file: "logs/server.log"
  max_size_mb: 100
  max_files: 10

# CORS Settings
cors:
  allowed_origins:
    - "http://localhost:3000"
    - "http://localhost:5173"
  allowed_methods:
    - "GET"
    - "POST"
    - "PUT"
    - "DELETE"
  allowed_headers:
    - "Content-Type"
    - "Authorization"
    - "X-API-Key"
```

### User Settings: `data/settings.yaml`

```yaml
# User-specific settings
theme: "dark"
language: "en"

# Visualization preferences
visualization:
  node_size: 10
  edge_width: 2
  label_size: 12
  show_labels: true
  animation_speed: 1.0

# Client preferences
client:
  auto_connect: true
  reconnect_attempts: 5
  reconnect_delay: 1000
```

## Environment Variables

### Core Variables

```bash
# Server
HOST=0.0.0.0
PORT=8080
RUST_LOG=info

# Database
DATABASE_URL=postgres://user:pass@localhost/dbname

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret

# AI Services
RAGFLOW_API_KEY=your-ragflow-key
PERPLEXITY_API_KEY=your-perplexity-key
OPENAI_API_KEY=your-openai-key

# Feature Access
POWER_USER_KEY_1=key1
POWER_USER_KEY_2=key2

# GPU
CUDA_ENABLED=true
CUDA_DEVICE_ID=0
CUDA_VISIBLE_DEVICES=0

# Development
DEV_MODE=false
HOT_RELOAD=true
```

### Docker Environment

```bash
# Docker-specific
DOCKER_BUILDKIT=1
COMPOSE_PROJECT_NAME=logseq-spring-thing

# Volumes
DATA_PATH=/data
LOGS_PATH=/logs

# Networking
EXTERNAL_PORT=8080
INTERNAL_PORT=8080
```

## Runtime Configuration

### Settings Actor

The `SettingsActor` manages runtime configuration changes:

```rust
// Get current settings
let settings = settings_actor.send(GetSettings).await?;

// Update specific setting
settings_actor.send(SetSettingByPath {
    path: "graph.simulation.damping".to_string(),
    value: json!(0.95),
}).await?;
```

### Client Settings Synchronization

Client settings are synchronized via WebSocket:

```typescript
// Client-side
const settings = {
  theme: 'dark',
  visualization: {
    nodeSize: 12,
    showLabels: true
  }
};

websocket.send({
  type: 'updateSettings',
  payload: settings
});
```

## Configuration Schemas

### AppFullSettings Structure

```rust
pub struct AppFullSettings {
    // Server configuration
    pub server: ServerConfig,
    
    // Graph settings
    pub graph: GraphConfig,
    
    // AI service settings
    pub ragflow: Option<RagflowConfig>,
    pub perplexity: Option<PerplexityConfig>,
    
    // Authentication
    pub auth: AuthConfig,
    
    // Feature flags
    pub features: FeatureConfig,
    
    // GPU settings
    pub gpu: GpuConfig,
    
    // User settings
    pub user_settings: UserSettings,
}
```

### Validation

Configuration is validated on load:

```rust
impl AppFullSettings {
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate port range
        if self.server.port < 1024 || self.server.port > 65535 {
            return Err(ConfigError::InvalidPort);
        }
        
        // Validate GPU settings
        if self.gpu.enabled && self.gpu.device_id > 7 {
            return Err(ConfigError::InvalidGpuDevice);
        }
        
        Ok(())
    }
}
```

## Loading Configuration

### Startup Sequence

```mermaid
sequenceDiagram
    participant Main
    participant Config
    participant Env
    participant File
    
    Main->>Config: load_configuration()
    Config->>File: Read config.yml
    Config->>File: Read settings.yaml
    Config->>Env: Override with env vars
    Config->>Config: Validate
    Config-->>Main: AppFullSettings
```

### Configuration Loading Code

```rust
pub async fn load_configuration() -> Result<AppFullSettings, ConfigError> {
    // Load base configuration
    let mut config = Config::new();
    
    // Add YAML files
    config.merge(File::with_name("config.yml"))?;
    config.merge(File::with_name("data/settings.yaml").required(false))?;
    
    // Override with environment variables
    config.merge(Environment::with_prefix("APP").separator("__"))?;
    
    // Deserialize and validate
    let settings: AppFullSettings = config.try_into()?;
    settings.validate()?;
    
    Ok(settings)
}
```

## Dynamic Configuration

### Hot Reloading

Enable configuration hot reloading:

```rust
// Watch for config file changes
let mut watcher = notify::recommended_watcher(|res| {
    match res {
        Ok(event) => reload_config(event),
        Err(e) => error!("Watch error: {:?}", e),
    }
})?;

watcher.watch(Path::new("config.yml"), RecursiveMode::NonRecursive)?;
```

### Feature Flags

Dynamic feature toggles:

```rust
// Check if feature is enabled
if settings.features.is_enabled("ai_chat") {
    // Enable AI chat functionality
}

// Toggle feature at runtime
settings_actor.send(SetSettingByPath {
    path: "features.enabled_features".to_string(),
    value: json!(["graph_visualization", "ai_chat"]),
}).await?;
```

## Security Considerations

### Sensitive Data

1. **Never commit secrets**: Use environment variables
   ```yaml
   api_key: "${API_KEY}"  # Good
   api_key: "sk-12345"    # Bad
   ```

2. **Rotate keys regularly**
   ```bash
   # Generate new API key
   openssl rand -hex 32
   ```

3. **Use secrets management**
   ```bash
   # Docker secrets
   docker secret create api_key ./api_key.txt
   ```

### Access Control

```yaml
# Restrict feature access
features:
  power_users_only:
    - admin_panel
    - gpu_compute
    - bulk_operations
```

## Best Practices

1. **Environment-Specific Configs**
   ```bash
   # Development
   cp config.dev.yml config.yml
   
   # Production
   cp config.prod.yml config.yml
   ```

2. **Configuration Validation**
   ```rust
   #[test]
   fn test_config_validation() {
       let config = load_test_config();
       assert!(config.validate().is_ok());
   }
   ```

3. **Documentation**
   ```yaml
   # Always document configuration options
   graph:
     max_nodes: 100000  # Maximum nodes in graph (affects memory usage)
   ```

4. **Defaults**
   ```rust
   impl Default for GraphConfig {
       fn default() -> Self {
           Self {
               max_nodes: 10000,
               simulation: SimulationParams::default(),
           }
       }
   }
   ```

## Troubleshooting

### Common Issues

1. **Configuration not loading**
   ```bash
   # Check file permissions
   ls -la config.yml
   
   # Validate YAML syntax
   yamllint config.yml
   ```

2. **Environment variables not working**
   ```bash
   # Debug environment
   env | grep APP_
   
   # Check variable expansion
   echo $DATABASE_URL
   ```

3. **Type mismatches**
   ```yaml
   # Ensure correct types
   port: 8080        # Number, not "8080"
   enabled: true     # Boolean, not "true"
   ```

### Debug Configuration

```rust
// Enable config debugging
std::env::set_var("CONFIG_DEBUG", "true");

// Log loaded configuration
info!("Loaded config: {:?}", settings);
```

## Migration Guide

### From v1 to v2

```yaml
# Old format
ai_service:
  type: "ragflow"
  key: "xxx"

# New format
ragflow:
  api_key: "xxx"
  base_url: "https://api.ragflow.com"
```

## Related Documentation

- [Server Configuration](../server/config.md) - Server-specific settings
- [Feature Access](../server/feature-access.md) - Feature flag system
- [Environment Setup](../development/setup.md) - Development environment
- [Deployment](../deployment/index.md) - Production configuration