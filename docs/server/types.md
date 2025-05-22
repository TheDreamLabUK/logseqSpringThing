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
    pub simulation_params: SimulationParams,
}

pub struct Node {
    pub id: String,
    pub data: NodeData,
    pub position: Option<[f32; 3]>,
    pub velocity: Option<[f32; 3]>,
}

pub struct Edge {
    pub source: String,
    pub target: String,
    pub weight: f32,
    pub kind: EdgeKind,
}

pub enum EdgeKind {
    Link,
    Reference,
    // ... other edge types
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
pub struct Settings {
    pub server: ServerConfig,
    pub visualisation: VisualisationConfig,
    pub github: GitHubServiceConfig,
    pub security: SecurityConfig,
    pub ai: AIServiceConfig,
    pub file_service: FileServiceConfig,
}

pub struct UserSettings {
    pub visualisation: VisualisationConfig,
    pub layout: LayoutConfig,
    pub theme: ThemeConfig,
    pub ai: AISettings,
}

pub struct ProtectedSettings {
    pub api_keys: HashMap<String, String>,
    pub security: SecurityConfig,
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