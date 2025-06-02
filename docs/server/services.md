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

### File Service (`src/services/file_service.rs`)
The `FileService` is responsible for interactions with the file system, particularly for managing Markdown files and their metadata.

```rust
// From src/services/file_service.rs
pub struct FileService {
    _settings: Arc<RwLock<AppFullSettings>>,
    node_id_counter: AtomicU32,
}

impl FileService {
    pub fn new(_settings: Arc<RwLock<AppFullSettings>>) -> Self;
    fn get_next_node_id(&self) -> u32;
    fn update_node_ids(&self, processed_files: &mut Vec<ProcessedFile>);
    pub async fn process_file_upload(&self, payload: web::Bytes) -> Result<GraphData, std::io::Error>;
    pub async fn list_files(&self) -> Result<Vec<String>, std::io::Error>;
    pub async fn load_file(&self, filename: &str) -> Result<GraphData, std::io::Error>;
    pub fn load_or_create_metadata() -> Result<MetadataStore, String>;
    // fn calculate_node_size(file_size: usize) -> f64; // private
    // fn extract_references(content: &str, valid_nodes: &[String]) -> Vec<String>; // private
    // fn convert_references_to_topic_counts(references: Vec<String>) -> HashMap<String, usize>; // private
    pub async fn initialize_local_storage(settings: Arc<RwLock<AppFullSettings>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    // fn update_topic_counts(metadata_store: &mut MetadataStore) -> Result<(), std::io::Error>; // private
    // fn has_valid_local_setup() -> bool; // private
    // fn ensure_directories() -> Result<(), std::io::Error>; // private
    pub fn save_metadata(metadata: &MetadataStore) -> Result<(), std::io::Error>;
    // fn calculate_sha1(content: &str) -> String; // private
    // fn count_hyperlinks(content: &str) -> usize; // private
    pub async fn fetch_and_process_files(
        &self,
        content_api: Arc<ContentAPI>,
        _settings: Arc<RwLock<AppFullSettings>>,
        metadata_store: &mut MetadataStore,
    ) -> Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>>;
}
```
- Manages file system operations, including reading, writing, and listing files.
- Handles metadata creation, loading, and saving.
- Processes file uploads and integrates with `ContentAPI` (which might use `GitHubClient`) for fetching files.
- Assigns unique node IDs.

### Graph Service (`src/services/graph_service.rs`)
The `GraphService` is central to managing the graph's structure, layout, and real-time updates.

```rust
// From src/services/graph_service.rs
pub struct GraphService {
    graph_data: Arc<RwLock<GraphData>>,
    shutdown_complete: Arc<AtomicBool>,
    node_map: Arc<RwLock<HashMap<String, Node>>>, // Node is from crate::utils::socket_flow_messages::Node
    gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    node_positions_cache: Arc<RwLock<Option<(Vec<Node>, Instant)>>>,
    last_update: Arc<RwLock<Instant>>,
    _pending_updates: Arc<RwLock<HashMap<String, (Node, Instant)>>>, // Marked as Dead Code in source
    cache_enabled: bool,
    simulation_id: String,
    _is_initialized: Arc<AtomicBool>, // Marked as Dead Code in source
    shutdown_requested: Arc<AtomicBool>,
}

impl GraphService {
    pub async fn new(
        settings: Arc<RwLock<AppFullSettings>>, // Consumes AppFullSettings
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        client_manager_for_loop: Arc<ClientManager>
    ) -> Self;

    pub async fn shutdown(&self);
    pub async fn get_simulation_diagnostics(&self) -> String;
    // async fn test_gpu_at_startup(gpu_compute: Option<Arc<RwLock<GPUCompute>>>); // private
    pub async fn wait_for_metadata_file() -> bool;
    pub async fn build_graph_from_metadata(metadata: &MetadataStore) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>>;
    pub async fn build_graph(state: &web::Data<AppState>) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>>;
    // fn initialize_random_positions(graph: &mut GraphData); // private
    pub async fn calculate_layout_with_retry(...); // public wrapper
    pub async fn calculate_layout(...); // public but usually called by retry wrapper
    pub fn calculate_layout_cpu(...); // public, CPU fallback
    pub async fn get_paginated_graph_data(...);
    pub async fn clear_position_cache(&self);
    pub async fn get_node_positions(&self) -> Vec<Node>;
    pub async fn get_graph_data_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, GraphData>;
    // pub async fn get_node_map_mut(...); // public
    pub async fn get_gpu_compute(&self) -> Option<Arc<RwLock<GPUCompute>>>;
    pub async fn update_node_positions(...);
    // pub fn update_positions(&mut self) -> Pin<Box<dyn Future<Output = Result<(), Error>> + '_>>; // public
    pub async fn initialize_gpu(&mut self, graph_data: &GraphData) -> Result<(), Error>;
    // pub fn diagnose_gpu_status(...); // public
    pub fn start_broadcast_loop(&self, client_manager: Arc<ClientManager>);
}
```
- Manages the in-memory `GraphData` (nodes, edges).
- Handles physics simulation, either via `GPUCompute` or a CPU fallback.
- Builds the graph from `MetadataStore`.
- Provides methods for accessing graph data (paginated, node positions).
- Manages a cache for node positions.
- Broadcasts updates to clients via `ClientManager`.
- Note: `settings` and `metadata_manager` are not direct fields; `GraphService` receives `settings` during construction and interacts with `MetadataStore` (often via `AppState`) to build the graph.

## AI Services

The application architecture includes several distinct AI-related services, rather than a single monolithic `AIService`. These are typically held as optional fields within `AppState`.

### Perplexity Service (`src/services/perplexity_service.rs`)
Integrates with the Perplexity AI API.
```rust
// From src/services/perplexity_service.rs
pub struct PerplexityService {
    // Fields like api_key, client, model, etc.
}
impl PerplexityService {
    // pub fn new(config: PerplexityConfig) -> Self;
    // pub async fn chat_completion(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, PerplexityError>;
}
```

### RAGFlow Service (`src/services/ragflow_service.rs`)
Integrates with a RAGFlow instance.
```rust
// From src/services/ragflow_service.rs
pub struct RAGFlowService {
    // Fields like api_key, client, api_base_url, etc.
}
impl RAGFlowService {
    // pub fn new(config: RAGFlowConfig) -> Self;
    // pub async fn chat(&self, request: RagflowChatRequest) -> Result<RagflowChatResponse, RagFlowError>;
}
```

### Speech Service (`src/services/speech_service.rs`)
Orchestrates Speech-to-Text (STT) and Text-to-Speech (TTS) functionalities. It may interact with other AI services (like OpenAI for STT/TTS).

```rust
// From src/services/speech_service.rs
pub struct SpeechService {
    pub sender: mpsc::Sender<SpeechCommand>, // For sending commands to its internal processing loop
    pub settings: Arc<RwLock<SpeechSettings>>, // Specific settings for speech services
    pub tts_provider: Arc<dyn TTSProvider + Send + Sync>, // TTS provider (e.g., OpenAI, Kokoro)
    pub audio_tx: mpsc::Sender<Vec<u8>>, // For sending processed audio (e.g., TTS output)
    pub http_client: Client, // reqwest client
}
impl SpeechService {
    // pub fn new(...) -> Self;
    // pub async fn process_command(&self, command: SpeechCommand);
    // pub async fn handle_text_to_speech(...);
    // pub async fn handle_speech_to_text(...);
}
```
- Manages audio streaming, STT processing (potentially via an external provider like OpenAI, not WhisperSttService directly as a separate struct in AppState), and TTS generation.

### Whisper STT Service
The `WhisperSttService` struct, as previously documented, is **not present** as a distinct service managed directly by `AppState`. STT functionality is likely handled within `SpeechService`, which might use OpenAI's Whisper API or another STT provider. The old documentation mentioning a separate `WhisperSttService` in `AppState` is outdated.

## Error Handling & State Management
- Each service typically defines its own error types (e.g., `GraphServiceError`, `FileServiceError`).
- Shared state (like `AppFullSettings`, `MetadataStore`) is managed within `AppState` using `Arc<RwLock<T>>` for thread-safe access. Services receive references to this state or relevant parts of it.

## Performance Optimization
- **Caching**: `GraphService` uses caching for node positions.
- **Batch Processing**: `FileService` processes GitHub files in batches.
- **GPU Acceleration**: `GraphService` uses `GPUCompute` for physics.