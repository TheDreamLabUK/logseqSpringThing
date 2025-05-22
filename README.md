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
        UI[User Interface Layer]
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
        HologramRenderer[Hologram Renderer]
        TextRenderer[Text Renderer]
        SettingsStore[Settings Store]
        NostrAuthClient[Nostr Auth Client UI]
    end

    %% Backend Components
    subgraph Backend
        ActixServer[Actix Web Server]
        FileHandler[File Handler]
        GraphHandler[Graph Handler]
        SocketFlowHandler[Socket Flow Handler]
        PerplexityHandler[Perplexity Handler]
        RagFlowHandler[RagFlow Handler]
        VisualizationHandler[Visualization Handler]
        NostrAuthHandler[Nostr Auth Handler]
        HealthHandler[Health Handler]
        PagesHandler[Pages Handler]
        SettingsHandler[Settings Handler]
        FileService[File Service]
        GraphService[Graph Service]
        GPUCompute[GPU Compute]
        PerplexityService[Perplexity Service]
        RagFlowService[RagFlow Service]
        SpeechService[Speech Service]
        NostrService[Nostr Service]
        ClientManager[Client Manager]
        PhysicsEngine[Physics Engine]
        AudioProcessor[Audio Processor]
        MetadataStoreModel[Metadata Store Model]
        ProtectedSettingsModel[Protected Settings Model]
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
    UI --> GraphDisplay
    UI --> RightPaneControlPanel
    UI --> ControlPanelLayout
    UI --> NostrAuthClient
    UI --> XRControlPanel

    XR --> R3FRenderer
    WSClient --> WSService
    WSService --> ActixServer

    %% Connections between Backend Components
    ActixServer --> FileHandler
    ActixServer --> GraphHandler
    ActixServer --> SocketFlowHandler
    ActixServer --> PerplexityHandler
    ActixServer --> RagFlowHandler
    ActixServer --> VisualizationHandler
    ActixServer --> NostrAuthHandler
    ActixServer --> HealthHandler
    ActixServer --> PagesHandler
    ActixServer --> SettingsHandler

    FileHandler --> FileService
    GraphHandler --> GraphService
    SocketFlowHandler --> ClientManager
    PerplexityHandler --> PerplexityService
    RagFlowHandler --> RagFlowService
    NostrAuthHandler --> NostrService

    GraphService --> PhysicsEngine
    PhysicsEngine --> GPUCompute
    PhysicsEngine --> ClientManager

    %% Connections to External Components
    FileService --> GitHubAPI
    PerplexityService --> PerplexityAI
    RagFlowService --> RagFlowAPI
    SpeechService --> OpenAI_API
    NostrAuthService --> NostrPlatformAPI

    %% Styling for clarity
    style Frontend fill:#f9f,stroke:#333,stroke-width:2px
    style Backend fill:#bbf,stroke:#333,stroke-width:2px
    style External fill:#bfb,stroke:#333,stroke-width:2px
```

### Class Diagram

```mermaid
classDiagram
    direction LR

    class AppClient {
        <<React Component>>
        +graphDataManager: GraphDataManager
        +webSocketService: WebSocketService
        +settingsStore: SettingsStore
        +platformManager: PlatformManager
        +xrSessionManager: XRSessionManager
        +nostrAuthService: NostrAuthService (Client)
        +initialize()
        +render()
    }

    class GraphManager {
        <<React Component>>
        +nodes: GraphNode[]
        +edges: Edge[]
        +updateNodePositions(data: ArrayBuffer)
        +renderGraph()
    }

    class WebSocketService {
        <<TypeScript Service>>
        -socket: WebSocket
        +connect()
        +sendMessage(data: object)
        +onBinaryMessage(callback: function)
        +onConnectionStatusChange(callback: function)
        +isReady(): boolean
    }

    class SettingsStore {
        <<Zustand Store>>
        +settings: Settings
        +get(path: string): any
        +set(path: string, value: any)
        +initialize(): Promise<Settings>
    }

    class GraphDataManager {
        <<TypeScript Service>>
        -data: GraphData
        +fetchInitialData(): Promise<GraphData>
        +updateNodePositions(data: ArrayBuffer)
        +sendNodePositions()
        +getGraphData(): GraphData
    }

    class NostrAuthService {
        <<TypeScript Service>>
        +login(): Promise<AuthState>
        +logout(): Promise<void>
        +onAuthStateChanged(listener: function): function
        +isAuthenticated(): boolean
    }

    AppClient --> GraphManager
    AppClient --> WebSocketService
    AppClient --> SettingsStore
    AppClient --> GraphDataManager
    AppClient --> NostrAuthService
    GraphDataManager --> WebSocketService

    class AppState {
        <<Rust Struct>>
        +graph_service: GraphService_Server
        +gpu_compute: Option<Arc<RwLock<GPUCompute_Util>>>
        +settings: Arc<RwLock<AppFullSettings>>
        +protected_settings: Arc<RwLock<ProtectedSettings_Model>>
        +metadata: Arc<RwLock<MetadataStore_Model>>
        +github_client: Arc<GitHubClient_Service>
        +content_api: Arc<ContentAPI_Service>
        +speech_service: Option<Arc<SpeechService_Server>>
        +nostr_service: Option<web::Data<NostrService_Server>>
        +client_manager: Arc<ClientManager_Server>
        +new(settings, github_client, content_api, speech_service, gpu_compute, client_manager)
    }

    class GraphService {
        <<Rust Struct>>
        +graph_data: Arc<RwLock<GraphData_Model>>
        +node_map: Arc<RwLock<HashMap_String_NodeModel_>>
        +gpu_compute: Option<Arc<RwLock<GPUCompute_Util>>>
        +client_manager: Arc<ClientManager_Server>
        +new(settings, gpu_compute, client_manager)
        +build_graph_from_metadata(metadata: &MetadataStore_Model): Result<GraphData_Model>
        +calculate_layout(gpu_compute, graph, node_map, params): Result<()>
        +start_broadcast_loop(client_manager)
        +get_node_positions(): Vec<Node_Model>
    }

    class SpeechService {
        <<Rust Struct>>
        +settings: Arc<RwLock<AppFullSettings>>
        +tts_provider: Arc<RwLock<TTSProvider_Enum>>
        +audio_tx: broadcast.Sender_Vec_u8_
        +new(settings)
        +text_to_speech(text: String, options: SpeechOptions): Result<()>
    }

    class NostrService {
        <<Rust Struct>>
        +users: Arc<RwLock<HashMap_String_NostrUser_>>
        +verify_auth_event(event: AuthEvent): Result<NostrUser_Model>
        +validate_session(pubkey: str, token: str): bool
        +get_user(pubkey: str): Option<NostrUser_Model>
    }
    
    class GPUCompute {
        <<Rust Struct>>
        +device: Arc<CudaDevice>
        +force_kernel: CudaFunction
        +node_data: CudaSlice_BinaryNodeData_
        +new(graph: &GraphData_Model): Result<Arc<RwLock<Self>>>
        +compute_forces(): Result<()>
        +get_node_data(): Result<Vec<BinaryNodeData>>
    }
    
    class ClientManager {
        <<Rust Struct>>
        +clients: RwLock<HashMap_usize_Addr_SocketFlowServer_>>
        +register(addr): usize
        +unregister(id: usize)
        +broadcast_node_positions(nodes: Vec<Node_Model>)
    }

    AppState --> GraphService
    AppState --> NostrService
    AppState --> SpeechService
    AppState --> GPUCompute
    AppState --> ClientManager
    GraphService --> GPUCompute
    GraphService --> ClientManager
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant ClientUI as Client UI (React)
    participant GraphDataManager as GraphDataManager (Client)
    participant WebSocketService as WebSocketService (Client)
    participant SettingsStore as SettingsStore (Client)
    participant NostrAuthService as NostrAuthService (Client)
    participant PlatformManager as PlatformManager (Client)
    participant ReactThreeFiber as ReactThreeFiber (Client)

    participant ActixServer as Actix Web Server (Backend)
    participant AppState as AppState (Backend)
    participant GraphService as GraphService (Backend)
    participant GPUCompute as GPUCompute (Backend)
    participant ClientManager as ClientManager (Backend)
    participant FileService as FileService (Backend)
    participant NostrService as NostrService (Backend)
    participant SpeechService as SpeechService (Backend)
    participant SettingsHandler as SettingsHandler (Backend)
    participant NostrHandler as NostrHandler (Backend)
    participant FileHandler as FileHandler (Backend)
    participant GraphHandler as GraphHandler (Backend)
    participant SocketFlowHandler as SocketFlowHandler (Backend)
    
    participant GitHubAPI as GitHub API (External)
    participant PerplexityAPI as Perplexity AI (External)
    participant RagFlowAPI as RagFlow API (External)
    participant OpenAI_API as OpenAI API (External)
    participant NostrPlatform as Nostr Platform (External)

    %% === Server Initialisation ===
    activate ActixServer
    ActixServer->>ActixServer: Load AppFullSettings (settings.yaml, env)
    alt Settings Load Error
        ActixServer-->>ClientUI: HTTP 500 (Conceptual)
    else Settings Loaded
        ActixServer->>AppState: new(AppFullSettings, GitHubClient, ContentAPI, SpeechService, GPUCompute, ClientManager)
        activate AppState
            Note over AppState: Initialises services like GitHubClient, ContentAPI
            AppState->>SpeechService: new(AppFullSettings)
            activate SpeechService; deactivate SpeechService
            AppState->>NostrService: new() (via init_nostr_service)
            activate NostrService; deactivate NostrService
            AppState->>FileService: load_or_create_metadata()
            activate FileService; deactivate FileService
            AppState->>GraphService: build_graph_from_metadata()
            activate GraphService
                GraphService->>GraphService: Initialise random positions
            deactivate GraphService
            AppState->>GPUCompute: new(GraphData) (or test_gpu)
            activate GPUCompute; deactivate GPUCompute
            AppState->>GraphService: new(AppFullSettings, GPUCompute, ClientManager)
            activate GraphService
                GraphService->>GraphService: Start physics simulation loop (async)
                GraphService->>GraphService: Start broadcast loop (async)
            deactivate GraphService
        AppState-->>ActixServer: Initialised AppState
        deactivate AppState
    end
    deactivate ActixServer

    %% === Client Initialisation ===
    activate ClientUI
    ClientUI->>PlatformManager: initialise()
    activate PlatformManager; deactivate PlatformManager
    ClientUI->>SettingsStore: initialise()
    activate SettingsStore
        SettingsStore->>SettingsStore: Load from localStorage
        SettingsStore->>ActixServer: GET /api/user-settings (fetchSettings)
        activate ActixServer
            ActixServer->>SettingsHandler: get_public_settings(AppState)
            SettingsHandler-->>ActixServer: UISettings (JSON)
        deactivate ActixServer
        ActixServer-->>SettingsStore: Settings JSON
        SettingsStore->>SettingsStore: Merge and store settings
    deactivate SettingsStore
    
    ClientUI->>NostrAuthService: initialise()
    activate NostrAuthService
        NostrAuthService->>NostrAuthService: Check localStorage for session
        alt Stored Session Found
            NostrAuthService->>ActixServer: POST /api/auth/nostr/verify (token)
            activate ActixServer
                ActixServer->>NostrHandler: verify(AppState, token_payload)
                NostrHandler->>NostrService: validate_session(pubkey, token)
                NostrService-->>NostrHandler: Validation Result
            deactivate ActixServer
            ActixServer-->>NostrAuthService: VerificationResponse
            NostrAuthService->>SettingsStore: Update auth state
        end
    deactivate NostrAuthService

    ClientUI->>WebSocketService: connect()
    activate WebSocketService
        WebSocketService->>ActixServer: WebSocket Handshake (/wss)
        activate ActixServer
            ActixServer->>SocketFlowHandler: handle_connection(AppState, ClientManager)
            activate SocketFlowHandler
                SocketFlowHandler->>ClientManager: register(client_addr)
                activate ClientManager; deactivate ClientManager
            deactivate SocketFlowHandler
        deactivate ActixServer
        ActixServer-->>WebSocketService: WebSocket Opened
        WebSocketService->>WebSocketService: isConnected = true
        WebSocketService-->>ActixServer: {"type":"requestInitialData"} (on connection_established from server)
        activate ActixServer
            ActixServer->>SocketFlowHandler: Handle requestInitialData
            SocketFlowHandler->>GraphService: get_node_positions()
            GraphService-->>SocketFlowHandler: Vec<Node_Model>
            SocketFlowHandler->>SocketFlowHandler: Encode to binary
            SocketFlowHandler-->>WebSocketService: Binary Position Data (Initial Graph)
        deactivate ActixServer
        WebSocketService->>GraphDataManager: updateNodePositions(binary_data)
        activate GraphDataManager
            GraphDataManager->>GraphDataManager: Parse binary, update internal graph
            GraphDataManager->>ReactThreeFiber: Trigger re-render
        deactivate GraphDataManager
    deactivate WebSocketService
    
    ClientUI->>GraphDataManager: fetchInitialData() (if WebSocket initial data is not primary)
    activate GraphDataManager
        GraphDataManager->>ActixServer: GET /api/graph/data
        activate ActixServer
            ActixServer->>GraphHandler: get_graph_data(AppState)
            GraphHandler->>GraphService: get_graph_data_mut()
            GraphService-->>GraphHandler: GraphData_Model
        deactivate ActixServer
        ActixServer-->>GraphDataManager: GraphData JSON
        GraphDataManager->>GraphDataManager: Set graph data
        GraphDataManager->>ReactThreeFiber: Trigger re-render
    deactivate GraphDataManager
    deactivate ClientUI

    %% === Continuous Graph Updates (Server to Client) ===
    loop Physics Simulation & Broadcast (Backend)
        GraphService->>GPUCompute: compute_forces()
        GPUCompute-->>GraphService: Updated Node Data
        GraphService->>ClientManager: broadcast_node_positions(updated_nodes)
        activate ClientManager
            ClientManager->>SocketFlowHandler: Send binary to all clients
            SocketFlowHandler-->>WebSocketService: Binary Position Data
        deactivate ClientManager
        WebSocketService->>GraphDataManager: updateNodePositions(binary_data)
        activate GraphDataManager
            GraphDataManager->>GraphDataManager: Parse binary, update internal graph
            GraphDataManager->>ReactThreeFiber: Trigger re-render
        deactivate GraphDataManager
    end

    %% === User Drags Node (Client to Server) ===
    ClientUI->>ReactThreeFiber: User interacts with node
    ReactThreeFiber->>GraphDataManager: Node position changed by user
    activate GraphDataManager
        GraphDataManager->>GraphDataManager: Update local node position
        GraphDataManager->>WebSocketService: sendNodePositions() (sends binary update)
        activate WebSocketService
            WebSocketService->>ActixServer: Binary Position Data (Client Update)
            activate ActixServer
                ActixServer->>SocketFlowHandler: Handle binary message
                SocketFlowHandler->>GraphService: update_node_positions(client_updates)
                activate GraphService
                    GraphService->>GraphService: Update internal graph, resolve conflicts
                    GraphService->>GPUCompute: compute_forces() (recalculate layout)
                    GPUCompute-->>GraphService: Updated Node Data
                deactivate GraphService
                Note over ActixServer: Server now has authoritative positions.
                Note over ActixServer: Broadcast loop will send these out.
            deactivate ActixServer
        deactivate WebSocketService
    deactivate GraphDataManager

    %% === Settings Update Flow ===
    ClientUI->>SettingsStore: User changes a setting
    activate SettingsStore
        SettingsStore->>SettingsStore: Update local settings state
        SettingsStore->>ActixServer: POST /api/user-settings/sync (settings JSON)
        activate ActixServer
            ActixServer->>SettingsHandler: update_user_settings(AppState, settings_payload)
            activate SettingsHandler
                SettingsHandler->>AppState: settings.write().await (AppFullSettings)
                AppState->>AppState: Merge client settings into AppFullSettings
                AppState->>AppState: AppFullSettings.save() to settings.yaml
                SettingsHandler->>ClientManager: Broadcast settings_updated JSON
                activate ClientManager
                    ClientManager->>SocketFlowHandler: Send JSON to all clients
                    SocketFlowHandler-->>WebSocketService: {"type":"settings_updated", "payload":...}
                deactivate ClientManager
            deactivate SettingsHandler
            SettingsHandler-->>ActixServer: Updated UISettings (JSON)
        deactivate ActixServer
        ActixServer-->>SettingsStore: Confirmation
    deactivate SettingsStore
    WebSocketService->>SettingsStore: Receive settings_updated message
    activate SettingsStore
        SettingsStore->>SettingsStore: Update local settings store
        SettingsStore->>ClientUI: Notify UI components of change
    deactivate SettingsStore

    %% === Nostr Authentication Flow ===
    ClientUI->>NostrAuthService: User clicks Login
    activate NostrAuthService
        NostrAuthService->>NostrAuthService: Interact with NIP-07 Provider (e.g., window.nostr)
        NostrAuthService->>NostrAuthService: Get pubkey, sign auth event
        NostrAuthService->>ActixServer: POST /api/auth/nostr (signed_event_payload)
        activate ActixServer
            ActixServer->>NostrHandler: login(AppState, event_payload)
            activate NostrHandler
                NostrHandler->>NostrService: verify_auth_event(event)
                activate NostrService
                    NostrService->>NostrService: Verify signature, manage user session
                    NostrService-->>NostrHandler: NostrUser_Model with session_token
                deactivate NostrService
            deactivate NostrHandler
            NostrHandler-->>ActixServer: AuthResponse (user_dto, token, expires_at, features)
        deactivate ActixServer
        ActixServer-->>NostrAuthService: AuthResponse JSON
        NostrAuthService->>NostrAuthService: Store token, update user state
        NostrAuthService->>SettingsStore: Update auth state in store
        NostrAuthService-->>ClientUI: Login successful / UI update
    deactivate NostrAuthService
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

```typescript
// Conceptual structure of a setting item (managed by SettingsStore)
interface UISetting { // From client/src/features/settings/types/uiSetting.ts
  type: 'slider' | 'toggle' | 'colour' | 'select' | 'number' | 'text'; // Simplified
  id?: string;
  label?: string;
  value?: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string; label: string }>;
  // ... other properties like description, help, advanced
}

// Conceptual structure for how settings are organised in the store (e.g., settings.visualisation.nodes)
interface SettingsCategory {
  [settingId: string]: UISetting | SettingsCategory;
}
```

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

