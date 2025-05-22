# App State Architecture

## Overview
The app state module manages the application's shared state and provides thread-safe access to core services.

## Core Structure

### AppState
```rust
pub struct AppState {
    pub settings: Arc<RwLock<Settings>>,
    pub metadata_manager: Arc<RwLock<MetadataManager>>,
    pub graph_service: GraphService,
    pub github_service: Arc<GitHubService>,
    pub file_service: Arc<FileService>,
    pub gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>,
    pub ai_service: Arc<AIService>,
    pub websocket_tx: broadcast::Sender<Message>,
}
```

## Initialization

### Constructor
```rust
impl AppState {
    pub async fn new(
        settings: Arc<RwLock<Settings>>,
        github_service: Arc<GitHubService>,
        file_service: Arc<FileService>,
        gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>,
        metadata_manager: Option<MetadataManager>,
        graph_data: Option<GraphData>,
        ai_service: Arc<AIService>,
        websocket_tx: broadcast::Sender<Message>,
    ) -> Result<Self, Error>
}
```

### Service Setup
- GitHub client initialization
- Graph service setup
- GPU compute configuration

## State Management

### Thread Safety
```rust
pub type SafeState = Arc<AppState>;
pub type SafeMetadata = Arc<RwLock<MetadataManager>>;
```

### Access Patterns
```rust
impl AppState {
    pub async fn get_metadata(&self) -> RwLockReadGuard<MetadataManager>
    pub async fn update_metadata(&self, updates: MetadataUpdates)
}
```

## Service Integration

### Graph Service
```rust
impl AppState {
    pub async fn update_graph(&self, data: GraphData)
    pub async fn get_graph_data(&self) -> GraphData
}
```

### File Service
```rust
impl AppState {
    pub async fn read_file_content(&self, path: &str) -> Result<String>
    pub async fn write_file_content(&self, path: &str, content: &str)
}
```

## Error Handling

### State Errors
```rust
pub enum StateError {
    Initialization(String),
    ServiceUnavailable(String),
    InvalidState(String),
}
```

### Recovery
```rust
impl AppState {
    pub async fn recover_from_error(&self, error: StateError)
    pub async fn validate_state(&self) -> Result<(), StateError>
}
```

## Implementation Details

### Cleanup
```rust
impl Drop for AppState {
    fn drop(&mut self) {
        // Cleanup resources
    }
}
```

### State Validation
```rust
impl AppState {
    pub fn validate(&self) -> Result<(), ValidationError>
}
   - Zero-copy when possible

## Graph System

The graph system manages the core data structures and algorithms for the knowledge graph.

### Data Flow

```mermaid
flowchart TB
    subgraph Input
        MD[Markdown Files]
        Meta[Metadata]
        User[User Updates]
    end

    subgraph Processing
        Parser[Content Parser]
        GraphBuilder[Graph Builder]
        Layout[Layout Engine]
    end

    subgraph Storage
        DB[Graph Database]
        Cache[Memory Cache]
    end

    MD --> Parser
    Meta --> Parser
    Parser --> GraphBuilder
    User --> GraphBuilder
    GraphBuilder --> Layout
    Layout --> Cache
    Cache --> DB
```

### Optimization Strategies

1. Caching
   - In-memory graph structure
   - Computed layout positions
   - Frequently accessed metadata

2. Batch Processing
   - Grouped node updates
   - Batched layout calculations
   - Bulk metadata updates

3. Incremental Updates
   - Partial graph updates
   - Delta-based synchronization
   - Progressive loading

## Service Layer

The service layer provides high-level operations and business logic.

### Core Services

1. Graph Service
   - Graph construction
   - Layout calculations
   - Data validation

2. File Service
   - Content management
   - File system operations
   - Version control integration

3. WebSocket Service
   - Real-time updates
   - Binary protocol handling
   - Connection management

### Service Communication

```mermaid
sequenceDiagram
    participant Client
    participant SocketFlowHandler
    participant GraphService
    participant GPUComputeService
    participant FileService
    participant AIService

    Client->>SocketFlowHandler: Connect
    SocketFlowHandler->>GraphService: Request Initial Data
    GraphService->>GPUComputeService: Calculate Layout
    GPUComputeService-->>GraphService: Layout Complete
    GraphService-->>SocketFlowHandler: Send Graph Data
    SocketFlowHandler-->>Client: Binary Update

    loop Real-time Updates
        Client->>SocketFlowHandler: Position Update
        SocketFlowHandler->>GraphService: Process Update
        GraphService->>GPUComputeService: Recalculate
        GPUComputeService-->>GraphService: New Positions
        GraphService-->>SocketFlowHandler: Broadcast Update
        SocketFlowHandler-->>Client: Binary Update
    end

    Client->>AIService: RAGFlow Query
    AIService-->>Client: RAGFlow Response

    Client->>AIService: TTS Request
    AIService-->>Client: TTS Audio Stream
```

## Next Steps

For detailed information about specific components, refer to:
- [Configuration](config.md)
- [Request Handlers](handlers.md)
- [Data Models](models.md)
- [Services](services.md)
- [Type Definitions](types.md)
- [Utilities](utils.md)