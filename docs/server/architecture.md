# App State Architecture

## Overview
The app state module manages the application's shared state and provides thread-safe access to core services. It acts as the central repository for all application-wide data and service instances, ensuring consistent access across different request handlers and background tasks.

## Core Structure

### AppState
The `AppState` struct holds references to all major services and shared data. It is designed for thread-safe access using `Arc<RwLock<T>>` for mutable shared state.

```rust
pub struct AppState {
    pub settings: Arc<RwLock<AppFullSettings>>,
    pub protected_settings: Arc<RwLock<ProtectedSettings>>,
    pub metadata: Arc<RwLock<MetadataStore>>,
    pub graph_service: GraphService,
    pub github_client: Arc<GitHubClient>,
    pub content_api: Arc<ContentAPI>,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub speech_service: Option<Arc<SpeechService>>,
    pub nostr_service: Option<web::Data<NostrService>>,
    pub feature_access: web::Data<FeatureAccess>,
    pub ragflow_session_id: String,
    pub active_connections: Arc<AtomicUsize>,
}
```

**ClientManager Usage:**
The `ClientManager` (from [`socket_flow_handler.rs`](../../src/handlers/socket_flow_handler.rs)) is a shared static instance, initialized lazily using `once_cell::sync::Lazy`. It is accessed within `AppState` methods or by services via `AppState::ensure_client_manager()`. This method returns a clone of the `Arc<ClientManager>`.

```rust
// In app_state.rs
static APP_CLIENT_MANAGER: Lazy<Arc<ClientManager>> =
    Lazy::new(|| Arc::new(ClientManager::new()));

impl AppState {
    pub async fn ensure_client_manager(&self) -> Arc<ClientManager> {
        APP_CLIENT_MANAGER.clone()
    }
}
```

## Initialization

### Constructor
The `AppState::new` constructor is responsible for setting up all services and loading initial data.

```rust
impl AppState {
    pub async fn new(
        settings: Arc<RwLock<AppFullSettings>>,
        github_client: Arc<GitHubClient>,
        content_api: Arc<ContentAPI>,
        perplexity_service: Option<Arc<PerplexityService>>,
        ragflow_service: Option<Arc<RAGFlowService>>,
        speech_service: Option<Arc<SpeechService>>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        ragflow_session_id: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
}
```

### Service Setup
-   **Settings Loading**: `AppFullSettings` are loaded from `settings.yaml` and environment variables.
-   **GitHub Client**: Initialized for repository content access.
-   **Content API**: Initialized for local file system operations and content fetching.
-   **Metadata Store**: Loaded or created, responsible for managing file metadata and relationships.
-   **Graph Service**: Built from initial metadata, responsible for graph data and physics simulation. It uses the shared `APP_CLIENT_MANAGER`.
-   **GPU Compute**: Optional, initialized if CUDA is available for accelerated physics calculations.
-   **AI Services**: Perplexity, RAGFlow, and Speech services are optionally initialized based on configuration.
-   **Nostr Service**: Initialized for authentication and user management (can be set later).
-   **Feature Access**: Configured to manage user permissions and feature flags.

## State Management

### Thread Safety
`AppState` and its contained mutable data are wrapped in `Arc<RwLock<T>>` to ensure safe concurrent access across multiple threads and asynchronous tasks.

```rust
pub type SafeAppState = Arc<AppState>; // This type alias might not be explicitly defined but represents Arc<AppState>
pub type SafeSettings = Arc<RwLock<AppFullSettings>>;
pub type SafeProtectedSettings = Arc<RwLock<ProtectedSettings>>;
pub type SafeMetadataStore = Arc<RwLock<MetadataStore>>;
```

### Access Patterns
Services and handlers access `AppState` fields using `read().await` for shared access and `write().await` for exclusive mutable access.

```rust
// Example access patterns (actual methods might differ)
// async fn example_read_settings(app_state: &AppState) -> AppFullSettings {
//     app_state.settings.read().await.clone()
// }
// async fn example_write_metadata(app_state: &AppState, new_metadata: MetadataStore) {
//     let mut metadata_guard = app_state.metadata.write().await;
//     *metadata_guard = new_metadata;
// }
```

## Service Integration

### Graph Service
Manages the in-memory graph data, physics simulation, and layout calculations. Interacts with `ClientManager` for broadcasting updates.

### Content API (File Service)
Handles reading and writing of files, primarily for markdown content and user settings. May interact with `GitHubClient`.

### AI Services
Provides interfaces for interacting with various AI models (Perplexity, RAGFlow, OpenAI TTS via `SpeechService`).

## Error Handling

### State Errors
Custom error types are defined for various initialization and runtime issues.
```rust
// Example (actual error types might be in specific service modules)
// pub enum AppStateError {
//     Initialization(String),
//     ServiceUnavailable(String),
// }
```

### Recovery
Mechanisms for graceful degradation or recovery from non-fatal errors are typically handled within individual services.

## Implementation Details

### Cleanup
The `Drop` trait is not explicitly implemented for `AppState` in the provided code, but `Arc` handles reference counting for automatic cleanup of shared resources when they are no longer in use. Individual services might implement `Drop` if they manage unmanaged resources.

### State Validation
Settings and other critical state components are validated during loading (e.g., `AppFullSettings::load`) or on modification to ensure data integrity.

## Graph System

The graph system manages the core data structures and algorithms for the knowledge graph, including parsing, building, and layout.

### Data Flow

```mermaid
flowchart TB
    subgraph Input
        MarkdownFiles[Markdown Files via Content API]
        UserUpdates[User Updates via API/WebSocket]
        GitHubContent[GitHub Content via GitHubClient & ContentAPI]
    end

    subgraph Processing
        ContentAPI[Content API (File Service)]
        MetadataStore[Metadata Store]
        GraphService[Graph Service]
        GPUCompute[GPU Compute]
    end

    subgraph Output
        GraphData[Graph Data (In-Memory in GraphService)]
        ClientUpdates[Client Updates (via ClientManager & WebSocket)]
        PersistedMetadata[Persisted Metadata (via MetadataStore)]
    end

    MarkdownFiles --> ContentAPI
    GitHubContent --> ContentAPI
    ContentAPI --> MetadataStore
    UserUpdates --> MetadataStore
    MetadataStore --> GraphService
    GraphService --> GPUCompute
    GPUCompute --> GraphService
    GraphService --> ClientUpdates
    MetadataStore --> PersistedMetadata
```

### Optimization Strategies

1.  **Caching**: In-memory caching of graph structure, computed layout positions, and frequently accessed metadata within `GraphService` and `MetadataStore`.
2.  **Batch Processing**: Grouped node updates and batched layout calculations for efficiency in `GraphService`.
3.  **Incremental Updates**: Partial graph updates and delta-based synchronization to minimize data transfer via WebSockets.
4.  **GPU Acceleration**: Offloading computationally intensive physics simulations to the GPU using CUDA, managed by `GPUCompute`.

## Service Layer

The service layer provides high-level operations and business logic, abstracting away lower-level details and external integrations.

### Core Services

1.  **Graph Service**: Manages graph construction, layout calculations, data validation, and broadcasting updates via `ClientManager`.
2.  **Content API (File Service)**: Handles content management, file system operations, and integration with external sources like GitHub.
3.  **Nostr Service**: Manages Nostr authentication, user sessions, and API key storage.
4.  **AI Services (Perplexity, RAGFlow, Speech)**: Provide interfaces for various AI capabilities. `SpeechService` handles TTS and STT, potentially interacting with other AI services.
5.  **GPU Compute**: Manages GPU resources and executes CUDA kernels for physics simulation.

### Service Communication Sequence Diagram - Speech Service

The sequence diagram for speech services needs correction. Speech is handled via its own WebSocket endpoint (`/speech`), not through the main `SocketFlowHandler` for graph updates.

```mermaid
sequenceDiagram
    participant Client
    participant SpeechSocketHandler as "/speech WebSocket"
    participant SpeechService
    participant STTProvider as "STT Provider (e.g., OpenAI)"
    participant RAGFlowService
    participant TTSProvider as "TTS Provider (e.g., OpenAI)"

    Client->>SpeechSocketHandler: Connect to /speech
    SpeechSocketHandler-->>Client: Connection Established

    Client->>SpeechSocketHandler: Send Audio Stream (Chunks)
    SpeechSocketHandler->>SpeechService: Forward Audio Chunks

    SpeechService->>STTProvider: Process Audio for STT
    STTProvider-->>SpeechService: Transcription Result

    opt Transcription successful & RAGFlow configured
        SpeechService->>RAGFlowService: Send Transcription for Query
        RAGFlowService-->>SpeechService: RAGFlow Response (Text)
        
        opt RAGFlow response & TTS configured
            SpeechService->>TTSProvider: Convert RAGFlow Text to Speech
            TTSProvider-->>SpeechService: TTS Audio Data (Stream/Buffer)
            SpeechService->>SpeechSocketHandler: Send TTS Audio to Client
            SpeechSocketHandler-->>Client: TTS Audio Stream/Data
        end
    else
        opt Transcription successful (no RAGFlow or direct TTS)
             SpeechService->>TTSProvider: Convert Original Transcription to Speech (if configured)
             TTSProvider-->>SpeechService: TTS Audio Data
             SpeechService->>SpeechSocketHandler: Send TTS Audio to Client
             SpeechSocketHandler-->>Client: TTS Audio Stream/Data
        end
    end
```

## Next Steps

For detailed information about specific components, refer to:
- [Configuration](config.md)
- [Request Handlers](handlers.md)
- [Data Models](models.md)
- [Services](services.md)
- [Type Definitions](types.md)
- [Utilities](utils.md)