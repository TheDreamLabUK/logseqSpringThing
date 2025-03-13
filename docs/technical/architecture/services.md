# Services Architecture

## Overview
The services layer provides core business logic, external integrations, and data processing capabilities.

## GitHub Service

### Client Configuration
```rust
pub struct GitHubConfig {
    pub api_token: String,
    pub repository: String,
    pub branch: String,
    pub owner: String,
}
```

### Content API
```rust
pub struct ContentAPI {
    client: Arc<GitHubClient>,
}

impl ContentAPI {
    pub async fn fetch_file_content(&self, url: &str) -> Result<String, GitHubError>
    pub async fn list_markdown_files(&self, path: &str) -> Result<Vec<GitHubFileMetadata>, GitHubError>
    pub async fn check_file_public(&self, url: &str) -> Result<bool, GitHubError>
}
```
- File content retrieval
- Markdown file listing
- Access control checks

### File Service
```rust
pub struct FileService {
    node_id_counter: AtomicU32,
}

impl FileService {
    pub async fn initialize_local_storage(settings: Arc<RwLock<Settings>>) -> Result<(), Box<dyn StdError>>
    pub async fn fetch_and_process_files(
        &self,
        content_api: Arc<ContentAPI>,
        settings: Arc<RwLock<Settings>>,
        metadata_store: &mut MetadataStore,
    ) -> Result<Vec<ProcessedFile>, Box<dyn StdError>>
}
```
- Local storage management
- File processing
- Metadata handling

### Graph Service
```rust
pub struct GraphService {
    settings: Arc<RwLock<Settings>>,
    graph_data: Arc<RwLock<GraphData>>,
    node_map: Arc<RwLock<HashMap<String, Node>>>,
    gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
}

impl GraphService {
    pub async fn new(
        settings: Arc<RwLock<Settings>>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    ) -> Self

    pub async fn calculate_layout_with_retry(
        gpu_compute: &Arc<RwLock<GPUCompute>>,
        graph: &mut GraphData,
        node_map: &mut HashMap<String, Node>,
        params: &SimulationParams,
    ) -> std::io::Result<()>
}
```
- Graph management
- Physics simulation
- Layout calculation

## Error Handling

### Service Errors
```rust
pub enum ServiceError {
    GitHub(GitHubError),
    File(std::io::Error),
    Graph(String),
    Configuration(String),
}
```

### Retry Mechanisms
```rust
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(1);
```
- Exponential backoff
- Error recovery
- Circuit breaking

## State Management

### Shared State
```rust
pub struct AppState {
    pub settings: Arc<RwLock<Settings>>,
    pub metadata: Arc<RwLock<MetadataStore>>,
    pub graph_service: GraphService,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
}
```
- Thread-safe access
- Service coordination
- Resource management

## Performance Optimization

### Caching
- In-memory caching
- File system caching
- Cache invalidation

### Batch Processing
```rust
const BATCH_SIZE: usize = 5;
for chunk in items.chunks(BATCH_SIZE) {
    // Process in batches
}
```

### Resource Management
- Connection pooling
- Memory optimization
- Resource cleanup
