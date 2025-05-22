# Models Architecture

## Overview
The models module defines the core data structures and their relationships within the application.

## Simulation Parameters

### Core Structure
```rust
pub struct SimulationParams {
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion: f32,
    pub damping: f32,
    pub max_repulsion_distance: f32,
    pub viewport_bounds: f32,
    pub mass_scale: f32,
    pub boundary_damping: f32,
    pub enable_bounds: bool,
    pub time_step: f32,
    pub phase: SimulationPhase,
    pub mode: SimulationMode,
    pub gravity_strength: f32,
    pub center_attraction_strength: f32,
}
```

### Usage
- Physics simulation configuration
- Real-time parameter adjustment
- Boundary conditions

## UI Settings

### Configuration
```rust
pub struct UserSettings {
    pub visualisation: VisualisationConfig,
    pub layout: LayoutConfig,
    pub theme: ThemeConfig,
    pub ai: AISettings,
}

pub struct VisualisationConfig {
    pub physics: PhysicsConfig,
    pub rendering: RenderingConfig,
    pub nodes: NodeVisualisationConfig,
    pub edges: EdgeVisualisationConfig,
    pub labels: LabelVisualisationConfig,
    pub bloom: BloomVisualisationConfig,
    pub hologram: HologramVisualisationConfig,
    pub animations: AnimationConfig,
}

pub struct AISettings {
    pub enabled: bool,
    pub default_model: String,
    pub temperature: f32,
}
```

### Features
- Theme customization
- Layout preferences
- Visualisation options

## User Settings

### Core Structure
```rust
// UserSettings is now defined above as part of the main settings structure.
```

### Persistence
- Local storage
- Profile sync
- Default values

## Protected Settings

### Security Configuration
```rust
pub struct ProtectedSettings {
    pub api_keys: HashMap<String, String>,
    pub security: SecurityConfig,
}
```

### Features
- API key management
- Security policies
- Rate limiting

## Metadata Store

### Core Structure
```rust
pub struct MetadataManager {
    pub files: HashMap<String, FileMetadata>,
    pub relationships: Vec<Relationship>,
    pub statistics: Statistics,
}
```

### Operations
- CRUD operations
- Relationship management
- Statistics tracking

## Implementation Details

### Thread Safety
```rust
pub type SafeMetadataManager = Arc<RwLock<MetadataManager>>;
pub type SafeSettings = Arc<RwLock<Settings>>;
```

### Serialization
```rust
impl Serialize for MetadataStore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
}
```

### Validation
```rust
impl SimulationParams {
    pub fn validate(&self) -> Result<(), ValidationError>
}