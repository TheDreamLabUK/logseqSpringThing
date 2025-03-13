# Types Architecture

## Overview
The types module defines core data structures, type aliases, and common enums used throughout the application.

## Core Types

### Graph Types
```rust
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: HashMap<String, Metadata>,
}

pub struct Node {
    pub id: String,
    pub data: NodeData,
}

pub struct Edge {
    pub source: String,
    pub target: String,
    pub weight: f32,
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
pub struct UISettings {
    pub theme: String,
    pub layout: LayoutConfig,
    pub visualization: VisualizationConfig,
}

pub struct UserSettings {
    pub preferences: HashMap<String, Value>,
    pub customizations: Vec<CustomSetting>,
}

pub struct ProtectedSettings {
    pub api_keys: HashMap<String, String>,
    pub security_config: SecurityConfig,
}
```

### Metadata Models
```rust
pub struct MetadataStore {
    pub files: HashMap<String, FileMetadata>,
    pub relationships: Vec<Relationship>,
}

pub struct FileMetadata {
    pub name: String,
    pub size: usize,
    pub node_id: String,
    pub last_modified: DateTime<Utc>,
}
```

## Error Types

### Service Errors
```rust
pub enum ServiceError {
    IO(std::io::Error),
    Graph(String),
    Config(String),
}

impl From<std::io::Error> for ServiceError {
    fn from(err: std::io::Error) -> Self {
        ServiceError::IO(err)
    }
}
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
```