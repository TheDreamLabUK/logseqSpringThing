# App State Architecture

## Overview
The app state module manages the application's shared state and provides thread-safe access to core services. It acts as the central repository for all application-wide data and service instances, ensuring consistent access across different request handlers and background tasks.

## Core Structure

### AppState
The `AppState` struct now holds actor addresses (`Addr<...Actor>`) for managing shared state. Communication happens via asynchronous message passing instead of `Arc<RwLock<T>>`.

```rust
// In src/app_state.rs
pub struct AppState {
    pub graph_service_addr: Addr<GraphServiceActor>,
    pub gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    pub settings_addr: Addr<SettingsActor>,
    pub protected_settings_addr: Addr<ProtectedSettingsActor>,
    pub metadata_addr: Addr<MetadataActor>,
    pub client_manager_addr: Addr<ClientManagerActor>,
    pub github_client: Arc<GitHubClient>,
    pub content_api: Arc<ContentAPI>,
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub speech_service: Option<Arc<SpeechService>>,
    pub nostr_service: Option<web::Data<NostrService>>,
    pub feature_access: web::Data<FeatureAccess>,
    pub ragflow_session_id: String,
    pub active_connections: Arc<AtomicUsize>,
}
```

### ClientManager
The `ClientManager` is now implemented as an **actor** (`ClientManagerActor`). Its address (`Addr<ClientManagerActor>`) is held in `AppState` and shared with other services that need to broadcast messages to clients, such as `GraphServiceActor`.

## Initialization

### Constructor
The `AppState::new` constructor is responsible for setting up all services and loading initial data.

```rust
impl AppState {
    pub async fn new(
        settings: AppFullSettings,
        github_client: Arc<GitHubClient>,
        content_api: Arc<ContentAPI>,
        perplexity_service: Option<Arc<PerplexityService>>,
        ragflow_service: Option<Arc<RAGFlowService>>,
        speech_service: Option<Arc<SpeechService>>,
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

`AppState` primarily manages shared mutable state through **Actix actors** and asynchronous message passing. While some services within `AppState` might still use `Arc<RwLock<T>>` for internal concurrency, the recommended pattern for interacting with shared application state is by sending messages to the appropriate actor addresses held by `AppState`.

```rust
pub type AppState = web::Data<crate::AppState>; // In handlers, AppState is typically wrapped in web::Data
// Individual actors manage their own internal state with thread-safe primitives (e.g., Arc<RwLock<T>>)
```

### Access Patterns
Services and handlers access state by sending messages to the actors via their addresses in `AppState`.

```rust
// Example access pattern using actor messages
// async fn example_get_settings(app_state: &AppState) -> AppFullSettings {
//     app_state.settings_addr.send(GetSettings).await.unwrap().unwrap()
// }
// async fn example_update_metadata(app_state: &AppState, new_metadata: MetadataStore) {
//     app_state.metadata_addr.do_send(UpdateMetadata { metadata: new_metadata });
// }
```

## Service Integration

### Graph Service ([`src/services/graph_service.rs`](../../src/services/graph_service.rs))
Manages the in-memory graph data (`GraphData`), runs the physics simulation (CPU or GPU via `GPUComputeActor`), calculates layout, and broadcasts updates to connected clients via the `ClientManagerActor`. It's typically started as a long-running task.

### File Service ([`src/services/file_service.rs`](../../src/services/file_service.rs))
This service, often referred to as `FileService` in the codebase, handles reading/writing local files, fetching content (e.g., from GitHub via `GitHubClient`), and managing the `MetadataStore`. It interacts with `ContentAPI` for GitHub operations.

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
        MarkdownFiles["Markdown Files"]
        UserUpdates["User Updates"]
        GitHubContent["GitHub Content"]
    end

    subgraph Processing
        ContentAPI["ContentAPI"]
        MetadataStore["MetadataStore"]
        GraphService["GraphService"]
        GPUComputeActor["GPUComputeActor"]
    end

    subgraph Output
        GraphData["GraphData"]
        ClientUpdates["ClientUpdates"]
        PersistedMetadata["PersistedMetadata"]
    end

    MarkdownFiles --> ContentAPI
    GitHubContent --> ContentAPI
    ContentAPI --> MetadataStore
    UserUpdates --> MetadataStore
    MetadataStore --> GraphService
    GraphService --> GPUComputeActor
    GPUComputeActor --> GraphService
    GraphService --> ClientUpdates
    MetadataStore --> PersistedMetadata

    style Input fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style Processing fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style Output fill:#3A3F47,stroke:#F7DF1E,color:#FFFFFF
    style MarkdownFiles fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style UserUpdates fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style GitHubContent fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style ContentAPI fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style MetadataStore fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style GraphService fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style GPUComputeActor fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style GraphData fill:#3A3F47,stroke:#F7DF1E,color:#FFFFFF
    style ClientUpdates fill:#3A3F47,stroke:#F7DF1E,color:#FFFFFF
    style PersistedMetadata fill:#3A3F47,stroke:#F7DF1E,color:#FFFFFF
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
5. **GPU Compute Actor**: Manages GPU resources and executes CUDA kernels for physics simulation.

### Service Communication Sequence Diagram - Speech Service

The client connects to the `/speech` WebSocket endpoint, handled by [`speech_socket_handler.rs`](../../src/handlers/speech_socket_handler.rs). The `SpeechService` processes audio, interacts with STT (e.g., OpenAI Whisper via an internal module or direct API call if not a separate "WhisperSttService" struct) and TTS providers (e.g., Kokoro, OpenAI), and can optionally query `RAGFlowService`. Audio responses are broadcast back to the client via the `speech_socket_handler` using a Tokio broadcast channel managed by `SpeechService`.

```mermaid
sequenceDiagram
    participant ClientApp as "Client Application"
    participant SpeechWSHandler as "Speech WebSocket Handler (/speech)"
    participant SpeechSvc as "SpeechService"
    participant ExternalSTT as "External STT Provider (e.g., OpenAI Whisper)"
    participant RAGFlowSvc as "RAGFlowService (Optional)"
    participant ExternalTTS as "External TTS Provider (e.g., Kokoro/OpenAI)"

    ClientApp->>SpeechWSHandler: WebSocket Connect to /speech
    SpeechWSHandler-->>ClientApp: Connection Established

    ClientApp->>SpeechWSHandler: Send Audio Stream (e.g., for STT)
    SpeechWSHandler->>SpeechSvc: Forward Audio Data / Command

    SpeechSvc->>ExternalSTT: Process Audio for STT
    ExternalSTT-->>SpeechSvc: Transcription Result (Text)

    alt If RAGFlow interaction is triggered
        SpeechSvc->>RAGFlowSvc: Send Transcription as Query
        RAGFlowSvc-->>SpeechSvc: RAGFlow Response (Text)
        SpeechSvc->>ExternalTTS: Convert RAGFlow Text to Speech
    else Else (Direct TTS of transcription or other command)
        SpeechSvc->>ExternalTTS: Convert Original Transcription/Text to Speech
    end

    ExternalTTS-->>SpeechSvc: TTS Audio Data (Stream/Buffer)
    SpeechSvc->>SpeechWSHandler: Broadcast TTS Audio Data (via tokio::sync::broadcast)
    SpeechWSHandler-->>ClientApp: Stream TTS Audio Data

    style ClientApp fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style SpeechWSHandler fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style SpeechSvc fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style ExternalSTT fill:#3A3F47,stroke:#F7DF1E,color:#FFFFFF
    style RAGFlowSvc fill:#3A3F47,stroke:#A2AAAD,color:#FFFFFF
    style ExternalTTS fill:#3A3F47,stroke:#F7DF1E,color:#FFFFFF
```

## Next Steps

For detailed information about specific components, refer to:
- [Configuration](config.md)
- [Request Handlers](handlers.md)
- [Data Models](models.md)
- [Services](services.md)
- [Type Definitions](types.md)
- [Utilities](utils.md)