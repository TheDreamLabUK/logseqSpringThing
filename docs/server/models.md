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
}
```

### Usage
- Physics simulation configuration
- Real-time parameter adjustment
- Boundary conditions

## UI Settings

### Configuration
```rust
pub struct UISettings {
    pub visualisation: VisualisationConfig,
    pub layout: LayoutConfig,
    pub theme: ThemeConfig,
}

pub struct VisualisationConfig {
    pub physics: PhysicsConfig,
    pub rendering: RenderingConfig,
}
```

### Features
- Theme customization
- Layout preferences
- Visualisation options

## User Settings

### Core Structure
```rust
pub struct UserSettings {
    pub preferences: HashMap<String, Value>,
    pub display: DisplaySettings,
    pub interaction: InteractionSettings,
}
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
    pub rate_limits: RateLimitConfig,
}
```

### Features
- API key management
- Security policies
- Rate limiting

## Metadata Store

### Core Structure
```rust
pub struct MetadataStore {
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
pub type SafeMetadataStore = Arc<RwLock<MetadataStore>>;
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