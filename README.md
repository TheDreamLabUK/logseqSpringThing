# LogseqXR: Immersive WebXR Visualisation for Logseq Knowledge Graphs

![image](https://github.com/user-attachments/assets/269a678d-88a5-42de-9d67-d73b64f4e520)

**Inspired by the innovative work of Prof. Rob Aspin:** [https://github.com/trebornipsa](https://github.com/trebornipsa)

![P1080785_1728030359430_0](https://github.com/user-attachments/assets/3ecac4a3-95d7-4c75-a3b2-e93deee565d6)

## About LogseqXR

LogseqXR transforms your Logseq knowledge base into an immersive 3D visualisation that you can explore in VR/AR. Experience your ideas as tangible objects in space, discover new connections, and interact with your knowledge in ways never before possible.

## Quick Links

- [Project Overview](docs/index.md)
- [Development Setup](docs/development/setup.md)
- [API Documentation](docs/api/index.md)
- [Contributing Guidelines](docs/contributing.md)

## Documentation

Our documentation is organised into several key sections:

### Client Documentation
- [Architecture](docs/client/architecture.md)
- [Components](docs/client/components.md)
- [Core Utilities](docs/client/core.md)
- [Rendering System](docs/client/rendering.md)
- [State Management](docs/client/state.md)
- [Type Definitions](docs/client/types.md)
- [Visualisation](docs/client/visualisation.md)
- [WebSocket Communication](docs/client/websocket.md)
- [WebXR Integration](docs/client/xr.md)

### Server Documentation
- [Architecture](docs/server/architecture.md)
- [Configuration](docs/server/config.md)
- [Request Handlers](docs/server/handlers.md)
- [Data Models](docs/server/models.md)
- [Services](docs/server/services.md)
- [Type Definitions](docs/server/types.md)
- [Utilities](docs/server/utils.md)

### API Documentation
- [REST API](docs/api/rest.md)
- [WebSocket API](docs/api/websocket.md)

### Development and Deployment
- [Development Setup](docs/development/setup.md)
- [Debugging Guide](docs/development/debugging.md)
- [Docker Deployment](docs/deployment/docker.md)
- [Contributing Guidelines](docs/contributing.md)

### System Architecture Diagram

```mermaid
graph TB
    %% Frontend Components
    subgraph Frontend
        AppInitializer[App Initializer]
        R3FRenderer[React Three Fiber Renderer]
        XR[WebXR Integration]
        WSClient[WebSocket Client]
        GraphManager[Graph Manager]
        RightPaneControlPanel[Right Pane Control Panel]
        ControlPanelLayout[Control Panel Layout]
        XRControlPanel[XR Control Panel]
        WSService[WebSocket Service]
        GraphDataManager[Graph Data Manager]
        PlatformManager[Platform Manager]
        XRSessionManager[XR Session Manager]
        XRInitializer[XR Initializer]
        HologramManager[Hologram Manager]
        TextRenderer[Text Renderer]
        SettingsStore[Settings Store]
        NostrAuthClient[Nostr Auth Client UI]
        GraphCanvas[Graph Canvas]
    end

    %% Backend Components
    subgraph Backend
        ActixServer[Actix Web Server]
        FileHandler[File Handler]
        GraphHandler[Graph Handler]
        SocketFlowHandler[Socket Flow Handler]
        PerplexityHandler[Perplexity Handler]
        RagFlowHandler[RagFlow Handler]
        VisualisationHandler[Visualisation Handler]
        NostrAuthHandler[Nostr Auth Handler]
        HealthHandler[Health Handler]
        PagesHandler[Pages Handler]
        SettingsHandler[Settings Handler]
        FileService[File Service]
        GraphService[Graph Service]
        GPUComputeService[GPU Compute Service]
        PerplexityService[Perplexity Service]
        RagFlowService[RagFlow Service]
        SpeechService[Speech Service]
        NostrService[Nostr Service]
        PhysicsEngine[Physics Engine]
        AudioProcessor[Audio Processor]
        MetadataManager[Metadata Manager]
        ProtectedSettings[Protected Settings]
        AIService[AI Service]
    end

    %% External Components
    subgraph External
        GitHubAPI[GitHub API]
        PerplexityAI[Perplexity AI]
        RagFlowAPI[RagFlow API]
        OpenAI_API[OpenAI API]
        NostrPlatformAPI[Nostr Platform API]
    end

    %% Connections between Frontend Components
    AppInitializer --> GraphCanvas
    AppInitializer --> RightPaneControlPanel
    AppInitializer --> ControlPanelLayout
    AppInitializer --> NostrAuthClient
    AppInitializer --> XRControlPanel
    GraphCanvas --> R3FRenderer
    WSClient --> WSService
    WSService --> ActixServer

    %% Connections between Backend Components
    ActixServer --> FileHandler
    ActixServer --> GraphHandler
    ActixServer --> SocketFlowHandler
    ActixServer --> PerplexityHandler
    ActixServer --> RagFlowHandler
    ActixServer --> VisualisationHandler
    ActixServer --> NostrAuthHandler
    ActixServer --> HealthHandler
    ActixServer --> PagesHandler
    ActixServer --> SettingsHandler

    FileHandler --> FileService
    GraphHandler --> GraphService
    %% SocketFlowHandler handles client connections directly
    PerplexityHandler --> PerplexityService
    RagFlowHandler --> RagFlowService
    NostrAuthHandler --> NostrService

    GraphService --> PhysicsEngine
    PhysicsEngine --> GPUComputeService

    %% Connections to External Components
    FileService --> GitHubAPI
    PerplexityService --> PerplexityAI
    RagFlowService --> RagFlowAPI
    SpeechService --> OpenAI_API
    NostrService --> NostrPlatformAPI
    AIService --> PerplexityAI
    AIService --> RagFlowAPI
    AIService --> OpenAI_API

    %% Styling for clarity
    style Frontend fill:#f9f,stroke:#333,stroke-width:2px
    style Backend fill:#bbf,stroke:#333,stroke-width:2px
    style External fill:#bfb,stroke:#333,stroke-width:2px
```

### Class Diagram

```mermaid
classDiagram
    direction LR

    class AppInitializer {
        <<React Component>>
        +graphDataManager: GraphDataManager
        +webSocketService: WebSocketService
        +settingsStore: SettingsStore
        +platformManager: PlatformManager
        +xrSessionManager: XRSessionManager
        +nostrAuthService: NostrAuthService
        +initialize(): Promise<void>
        +render(): JSX.Element
    }

    class GraphManager {
        <<React Component>>
        +nodes: Node[]
        +edges: Edge[]
        +updateNodePositions(data: ArrayBuffer): void
        +renderGraph(): JSX.Element
    }

    class WebSocketService {
        <<TypeScript Service>>
        -socket: WebSocket
        +connect(): Promise<void>
        +sendMessage(data: object): void
        +onBinaryMessage(callback: (data: ArrayBuffer) => void): () => void
        +onConnectionStatusChange(callback: (status: boolean) => void): () => void
        +isReady(): boolean
    }

    class SettingsStore {
        <<Zustand Store>>
        +settings: Settings
        +setSetting(path: string, value: any): void
        +initialize(): Promise<void>
    }

    class GraphDataManager {
        <<TypeScript Service>>
        -data: GraphData
        +fetchInitialData(): Promise<GraphData>
        +updateNodePositions(data: ArrayBuffer): void
        +sendNodePositions(nodes: Node[]): void
        +getGraphData(): GraphData
    }

    class NostrAuthService {
        <<TypeScript Service>>
        +login(): Promise<AuthResponse>
        +logout(): Promise<void>
        +onAuthStateChanged(listener: (state: AuthState) => void): () => void
        +isAuthenticated(): boolean
    }

    AppInitializer --> GraphManager
    AppInitializer --> WebSocketService
    AppInitializer --> SettingsStore
    AppInitializer --> GraphDataManager
    AppInitializer --> NostrAuthService
    GraphDataManager --> WebSocketService

    class AppState {
        <<Rust Struct>>
        +settings: Arc<RwLock<AppFullSettings>>
        +protected_settings: Arc<RwLock<ProtectedSettings>>
        +metadata_manager: Arc<RwLock<MetadataManager>>
        +graph_service: GraphService
        +github_service: Arc<GitHubService>
        +file_service: Arc<FileService>
        +gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>
        +ai_service: Arc<AIService>
        +nostr_service: Arc<NostrService>
        +websocket_tx: broadcast::Sender<Message>
        +new(settings, github_service, file_service, gpu_compute_service, metadata_manager, graph_data, ai_service, websocket_tx): Result<Self, Error>
    }

    class GraphService {
        <<Rust Struct>>
        +settings: Arc<RwLock<AppFullSettings>>
        +graph_data: Arc<RwLock<GraphData>>
        +node_map: Arc<RwLock<HashMap<String, Node>>>
        +gpu_compute_service: Option<Arc<RwLock<GPUComputeService>>>
        +metadata_manager: Arc<RwLock<MetadataManager>>
        +new(settings, gpu_compute_service, metadata_manager): Self
        +update_graph_data(new_data: GraphData): Result<(), GraphServiceError>
        +get_graph_data(): GraphData
        +calculate_layout(params: &SimulationParams): Result<(), GraphServiceError>
    }

    class AIService {
        <<Rust Struct>>
        +config: AIServiceConfig
        +perplexity_client: Option<PerplexityClient>
        +openai_client: Option<OpenAIClient>
        +ragflow_client: Option<RagflowClient>
        +new(config: AIServiceConfig): Self
        +chat_completion(model: &str, messages: Vec<ChatMessage>): Result<ChatResponse, AIServiceError>
        +text_to_speech(text: &str) -> Result<Vec<u8>, AIServiceError>
        +ragflow_chat(query: &str) -> Result<RagflowResponse, AIServiceError>
    }

    class NostrService {
        <<Rust Struct>>
        +users: Arc<RwLock<HashMap<String, NostrUser>>>
        +verify_auth_event(event: &AuthEvent): Result<NostrUser, NostrError>
        +validate_session(pubkey: &str, token: &str): bool
        +get_user(pubkey: &str): Option<NostrUser>
    }
    
    class GPUComputeService {
        <<Rust Struct>>
        +device: Arc<CudaDevice>
        +force_kernel: CudaFunction
        +node_data: CudaSlice<BinaryNodeData>
        +new(graph: &GraphData): Result<Arc<RwLock<Self>>, GPUComputeError>
        +compute_forces(): Result<(), GPUComputeError>
        +get_node_data(): Result<Vec<BinaryNodeData>, GPUComputeError>
    }
    
    AppState --> GraphService
    AppState --> NostrService
    AppState --> AIService
    AppState --> GPUComputeService
    AppState --> FileService
    AppState --> GitHubService
    GraphService --> GPUComputeService
```

### Sequence Diagrams

#### Server Initialization Sequence

```mermaid
sequenceDiagram
    participant Server as Actix Server
    participant AppState as App State
    participant Services as Services
    participant GPU as GPU Service

    Server->>Server: Load settings.yaml
    Server->>AppState: Create AppState
    AppState->>Services: Initialize services
    Services->>Services: Create AI, Nostr, File services
    AppState->>GPU: Initialize GPU compute
    GPU-->>AppState: GPU ready
    AppState-->>Server: Server ready
```

#### Client Initialization Sequence

```mermaid
sequenceDiagram
    participant Client as App
    participant Store as Settings Store
    participant Auth as Nostr Auth
    participant WS as WebSocket
    participant Server as Server

    Client->>Store: Initialize settings
    Store->>Store: Load from localStorage
    Store->>Server: GET /api/user-settings
    Server-->>Store: Settings data
    
    Client->>Auth: Check auth status
    Auth->>Auth: Check stored session
    opt Has stored session
        Auth->>Server: POST /api/auth/nostr/verify
        Server-->>Auth: Verification result
    end
    
    Client->>WS: Connect WebSocket
    WS->>Server: WebSocket handshake
    Server-->>WS: Connection established
    WS->>Server: Request initial data
    Server-->>WS: Graph data (binary)
```

#### Real-time Graph Updates Sequence

```mermaid
sequenceDiagram
    participant Client as Client
    participant WS as WebSocket
    participant Server as Server
    participant GPU as GPU Service

    loop Physics Simulation
        Server->>GPU: Compute forces
        GPU-->>Server: Updated positions
        Server->>WS: Broadcast positions
        WS-->>Client: Binary position data
        Client->>Client: Update visualization
    end

    opt User interaction
        Client->>Client: Drag node
        Client->>WS: Send position update
        WS->>Server: Position data
        Server->>Server: Update graph
        Server->>GPU: Recalculate
    end
```

#### Authentication Flow Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant Client as Client
    participant Nostr as Nostr Provider
    participant Server as Server

    User->>Client: Click login
    Client->>Nostr: Request signature
    Nostr->>Nostr: Sign auth event
    Nostr-->>Client: Signed event
    Client->>Server: POST /api/auth/nostr
    Server->>Server: Verify signature
    Server->>Server: Create session
    Server-->>Client: Auth token
    Client->>Client: Store token
    Client->>Client: Update UI
```

#### Settings Synchronization Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant Client as Client
    participant Server as Server
    participant WS as WebSocket

    User->>Client: Change setting
    Client->>Client: Update local state
    Client->>Server: POST /api/user-settings/sync
    
    alt Power User
        Server->>Server: Update global settings
        Server->>Server: Save to settings.yaml
        Server->>WS: Broadcast to all clients
        WS-->>Client: Settings update
    else Regular User
        Server->>Server: Update user settings
        Server->>Server: Save to user file
        Server-->>Client: Confirmation
    end
```

### AR Features Implementation Status

#### Hand Tracking (Meta Quest 3)
- XR Interaction is primarily managed by `client/src/features/xr/systems/HandInteractionSystem.tsx` and related hooks/providers like `useSafeXRHooks.tsx`.
- Session management is in `client/src/features/xr/managers/xrSessionManager.ts`.
- Initialisation logic is in `client/src/features/xr/managers/xrInitializer.ts`.
- Currently addressing:
  - Performance optimisation for AR passthrough mode.
  - Virtual desktop cleanup during AR activation (conceptual, not explicitly in code).
  - Type compatibility for WebXR hand input APIs (e.g., `XRHand`, `XRJointSpace` as seen in `webxr-extensions.d.ts`).
  - Joint position extraction methods for gesture recognition.

##### Current Challenges
- Ensuring robust type definitions for WebXR extensions across different browsers/devices (see `client/src/features/xr/types/webxr-extensions.d.ts`).
- Extracting and interpreting joint positions from `XRJointSpace` for reliable gesture recognition (conceptual, `HandInteractionSystem.tsx` has stubs).
- Performance optimisation in AR passthrough mode, especially with complex scenes.

##### Next Steps
- Refine `webxr-extensions.d.ts` for better type safety with hand tracking APIs.
- Implement more sophisticated gesture recognition in `HandInteractionSystem.tsx`.
- Optimise AR mode transitions and rendering performance.
- Enhance Meta Quest 3 specific features if possible (e.g., passthrough quality).

### Authentication and Settings Inheritance

#### Unauthenticated Users
- Use browser's localStorage for settings persistence (via Zustand `persist` middleware in `client/src/store/settingsStore.ts`).
- Settings are stored locally and not synced to a user-specific backend store.
- Default to basic settings visibility.
- Limited to local visualisation features; AI and GitHub features requiring API keys will not be available unless default keys are configured on the server.

#### Authenticated Users (Nostr)
- **Regular Users**:
    - Settings are loaded from and saved to user-specific files on the server (e.g., `/app/user_settings/<pubkey>.yaml`), managed by `src/handlers/settings_handler.rs` using `UserSettings` model.
    - These user-specific settings are a subset of the global settings (typically UI/visualisation preferences defined in `UISettings`).
    - Can access features based on their `feature_access.rs` configuration (e.g., RAGFlow, OpenAI by default for new users).
    - Can manage their own API keys for these services via `/api/auth/nostr/api-keys` endpoint, stored in their `NostrUser` profile on the server.
- **Power Users**:
    - Directly load and modify the global server settings from `settings.yaml` (represented by `AppFullSettings` in Rust).
    - Have full access to all settings and advanced API features (Perplexity, RAGFlow, GitHub, OpenAI TTS) which use API keys configured in `settings.yaml` or environment variables.
    - Settings modifications made by power users are persisted to the main `settings.yaml` and broadcast to all connected clients.

### Settings Inheritance Flow

```mermaid
graph TD
    A[Start] --> B{"Authenticated?"}
    B -->|No| C["Load LocalSettings (localStorage via Zustand)"]
    B -->|Yes| D{"Is Power User? (feature_access.rs)"}
    D -->|No| E["Load UserSpecificSettings (user_settings/pubkey.yaml via API)"]
    D -->|Yes| F["Load GlobalServerSettings (settings.yaml via API)"]
    C --> X["Apply Settings to UI"]
    E --> X
    F --> X
```

### Settings Sync Flow

```mermaid
graph TD
    A["Setting Changed in UI"] --> B{"Authenticated?"}
    B -->|No| C["Save Locally (localStorage via Zustand)"]
    B -->|Yes| D{"Is Power User?"}
    D -->|No| E["Save to UserSpecificSettings (user_settings/pubkey.yaml via API)"]
    D -->|Yes| F["Save to GlobalServerSettings (settings.yaml via API)"]
    F --> G["Server Broadcasts GlobalSettingsUpdate to All Clients"]
    G --> H["Other Clients Update Local Store"]
    E --> I["User's Local Store Updated"]
    C --> I
```

### Modular Control Panel Architecture

The client's user interface for settings and controls is primarily managed by the `client/src/app/TwoPaneLayout.tsx` component, which uses `client/src/app/components/RightPaneControlPanel.tsx` for the right-hand side. The `client/src/components/layout/ControlPanel.tsx` component provides the tabbed interface for organizing different categories of settings and tools within the right pane. Some sections, like those within `SettingsSection.tsx`, support being "detached" into floating draggable windows.

#### Component Structure

The main UI is structured as follows:
- **`RightPaneControlPanel.tsx`**: This component manages the content of the right pane, which includes:
    - Tabs for core settings:
        - Nostr Authentication (`NostrAuthSection.tsx`)
        - System Settings (`SystemPanel.tsx`)
        - Visualisation Settings (`VisualisationPanel.tsx`)
        - XR Settings (`XRPanel.tsx`)
        - AI Services Settings (`AIPanel.tsx`)
    - Tabs for features/tools:
        - Embedded "Narrative Gold Mine" iframe.
        - Markdown Renderer (`MarkdownRenderer.tsx`) for displaying content.
        - LLM Query interface (basic textarea and button).
- **`ControlPanel.tsx`**: This component provides the tabbed layout and manages the active tab within the `RightPaneControlPanel.tsx`.
- **`SettingsSection.tsx`**: Used within panels (e.g., `VisualisationPanel.tsx`) to group related settings. Supports:
    - Collapsible sections.
    - Detaching into a draggable, floating window using `react-draggable`.
- **`SettingControlComponent.tsx`**: Renders individual UI controls (sliders, toggles, inputs) for each setting, including dynamic tooltips using `Tooltip.tsx`.

The conceptual interfaces for settings provided in the original README are useful for understanding the data structure but are not direct props to a single "ModularControlPanel" component. Instead, settings are managed by `zustand` (`SettingsStore.ts`) and individual panel components consume and update this store.

#### Layout Management
The overall layout is a fixed two-pane structure managed by `TwoPaneLayout.tsx`.
Individual `SettingsSection` components can be detached, and their position is managed by `react-draggable` locally. There isn't a global `LayoutConfig` prop managing all detachable panel positions in the way the conceptual interface suggested. User preferences for advanced settings visibility are handled by `control-panel-context.tsx` and `useControlPanelContext`.

#### Performance Optimisations
- **Debounced Updates**: `SettingControlComponent.tsx` uses `onBlur` or Enter key for text/number inputs, which acts as a form of debouncing for settings changes that might trigger expensive re-renders or API calls.
- **CSS Transforms**: Used by `react-draggable` for smooth movement of detached panels.
- **Memoisation**: `useMemo` is used in components like `GraphManager.tsx` to stabilise expensive calculations or object references.
- **Targeted Re-renders**: Zustand store selectors for primitive values are used in some places (e.g., `App.tsx`) to avoid unnecessary re-renders.

The goal is to maintain responsiveness, especially during interactions with the 3D visualisation and real-time updates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Prof Rob Aspin: For inspiring the project's vision and providing valuable resources.
- OpenAI: For their advanced AI models powering the question-answering features.
- Perplexity AI and RAGFlow: For their AI services enhancing content processing and interaction.
- Three.js: For the robust 3D rendering capabilities utilised in the frontend.
- Actix: For the high-performance web framework powering the backend server.
