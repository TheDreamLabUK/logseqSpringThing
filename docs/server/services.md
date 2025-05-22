# Services Architecture

## Overview
The services layer provides core business logic, external integrations, and data processing capabilities.

## GitHub Service

### Client Configuration
```rust
pub struct GitHubServiceConfig {
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub token: Option<String>,
}
```

### Content API
```rust
// ContentAPI has been integrated into GitHubService and FileService.

### File Service
```rust
pub struct FileService {
    config: FileServiceConfig,
}

impl FileService {
    pub async fn new(config: FileServiceConfig) -> Self
    pub async fn read_file(&self, path: &Path) -> Result<String, FileServiceError>
    pub async fn write_file(&self, path: &Path, content: &str) -> Result<(), FileServiceError>
    pub async fn list_files_recursive(&self, path: &Path) -> Result<Vec<PathBuf>, FileServiceError>
}
```
- Manages file system operations, including reading, writing, and listing files.
- Enforces file size limits and handles I/O errors.

### Graph Service
```rust
pub struct GraphService {
    settings: Arc<RwLock<Settings>>,
    graph_data: Arc<RwLock<GraphData>>,
    node_map: Arc<RwLock<HashMap<String, Node>>>,
    gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>,
    metadata_manager: Arc<RwLock<MetadataManager>>,
}

impl GraphService {
    pub async fn new(
        settings: Arc<RwLock<Settings>>,
        gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>,
        metadata_manager: Arc<RwLock<MetadataManager>>,
    ) -> Self

    pub async fn update_graph_data(&self, new_data: GraphData) -> Result<(), GraphServiceError>
    pub async fn get_graph_data(&self) -> GraphData
    pub async fn calculate_layout(&self, params: &SimulationParams) -> Result<(), GraphServiceError>
}
```
- Graph management
- Physics simulation
- Layout calculation

## Error Handling

### Service Errors
```rust
pub enum ServiceError {
    GitHub(GitHubServiceError),
    File(FileServiceError),
    Graph(GraphServiceError),
    AI(AIServiceError),
    Configuration(String),
    // ... other service-specific errors
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
    pub metadata_manager: Arc<RwLock<MetadataManager>>,
    pub graph_service: GraphService,
    pub github_service: Arc<GitHubService>,
    pub file_service: Arc<FileService>,
    pub gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>,
    pub ai_service: Arc<AIService>,
    pub websocket_tx: broadcast::Sender<Message>,
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

## AI Service

### Core Structure
```rust
pub struct AIService {
    config: AIServiceConfig,
    perplexity_client: Option<PerplexityClient>,
    openai_client: Option<OpenAIClient>,
    ragflow_client: Option<RagflowClient>,
}

impl AIService {
    pub async fn new(config: AIServiceConfig) -> Self
    pub async fn chat_completion(&self, model: &str, messages: Vec<ChatMessage>) -> Result<ChatResponse, AIServiceError>
    pub async fn text_to_speech(&self, text: &str) -> Result<Vec<u8>, AIServiceError>
    pub async fn ragflow_chat(&self, query: &str) -> Result<RagflowResponse, AIServiceError>
}
```
- Integrates with various AI providers (Perplexity, OpenAI, RAGFlow).
- Provides functionalities for chat completion, text-to-speech, and RAG-based queries.
- Manages API keys and service-specific configurations.