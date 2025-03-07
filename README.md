# LogseqXR: Immersive WebXR Visualization for Logseq Knowledge Graphs

![image](https://github.com/user-attachments/assets/269a678d-88a5-42de-9d67-d73b64f4e520)

**Inspired by the innovative work of Prof. Rob Aspin:** [https://github.com/trebornipsa](https://github.com/trebornipsa)

![P1080785_1728030359430_0](https://github.com/user-attachments/assets/3ecac4a3-95d7-4c75-a3b2-e93deee565d6)

## About LogseqXR

LogseqXR transforms your Logseq knowledge base into an immersive 3D visualization that you can explore in VR/AR. Experience your ideas as tangible objects in space, discover new connections, and interact with your knowledge in ways never before possible.

## Quick Links

- [Project Overview](docs/overview/introduction.md)
- [Technical Architecture](docs/overview/architecture.md)
- [Development Setup](docs/development/setup.md)
- [API Documentation](docs/api/index.md)
- [Contributing Guidelines](docs/contributing/guidelines.md)

## Documentation Structure

### Overview
- [Introduction & Features](docs/overview/introduction.md)
- [System Architecture](docs/overview/architecture.md)

### Technical Documentation
- [WebSocket Communication](docs/technical/websockets.md)
- [Binary Protocol](docs/technical/binary-protocol.md)
- [Performance Optimizations](docs/technical/performance.md)
- [Class Diagrams](docs/technical/class-diagrams.md)

### Development
- [Setup Guide](docs/development/setup.md)
- [Debugging Guide](docs/development/debugging.md)

### API Documentation
- [API Overview](docs/api/index.md)
- [REST API](docs/api/rest.md)
- [WebSocket API](docs/api/websocket.md)

### Deployment
- [Docker Deployment](docs/deployment/docker.md)

### Contributing
- [Contributing Guidelines](docs/contributing/guidelines.md)

### Diagrams

```mermaid
graph TB
    %% Frontend Components
    subgraph Frontend
        UI[User Interface Layer]
        VR[WebXR Controller]
        WS[WebSocket Client]
        GPU[GPU Compute Layer]
        ThreeJS[Three.js Renderer]
        ChatUI[Chat Interface]
        GraphUI[Graph Interface]
        ControlPanel["Modular Control Panel (with Nostr Auth)"]
        VRControls[VR Control System]
        WSService[WebSocket Service]
        DataManager[Graph Data Manager]
        LayoutEngine[Layout Engine]
        SpaceMouse[SpaceMouse Controller]
        PlatformManager[Platform Manager]
        XRSession[XR Session Manager]
        XRInit[XR Initializer]
        SceneManager[Scene Manager]
        NodeManager[Enhanced Node Manager]
        EdgeManager[Edge Manager]
        HologramManager[Hologram Manager]
        TextRenderer[Text Renderer]
        SettingsStore[Settings Store]
    end

    %% Backend Components
    subgraph Backend
        Server[Actix Web Server]
        FileH[File Handler]
        GraphH[Graph Handler]
        WSH[WebSocket Handler]
        PerplexityH[Perplexity Handler]
        RagFlowH[RagFlow Handler]
        VisualizationH[Visualization Handler]
        NostrH[Nostr Handler]
        HealthH[Health Handler]
        PagesH[Pages Handler]
        SettingsH[Settings Handler]
        FileS[File Service]
        GraphS[Graph Service]
        GPUS[GPU Compute Service]
        PerplexityS[Perplexity Service]
        RagFlowS[RagFlow Service]
        SpeechS[Speech Service]
        NostrS[Nostr Service]
        WSManager[WebSocket Manager]
        GPUCompute[GPU Compute]
        Compression[Compression Utils]
        AudioProc[Audio Processor]
        MetadataStore[Metadata Store]
        ProtectedSettings[Protected Settings]
    end

    %% External Components
    subgraph External
        GitHub[GitHub API]
        Perplexity[Perplexity AI]
        RagFlow[RagFlow API]
        OpenAI[OpenAI API]
        NostrAPI[Nostr API]
    end

    %% Connections between Frontend Components
    UI --> ChatUI
    UI --> GraphUI
    UI --> ControlPanel
    UI --> VRControls

    VR --> ThreeJS
    WS --> WSService
    WSService --> Server

    %% Connections between Backend Components
    Server --> FileH
    Server --> GraphH
    Server --> WSH
    Server --> PerplexityH
    Server --> RagFlowH
    Server --> VisualizationH
    Server --> NostrH
    Server --> HealthH
    Server --> PagesH
    Server --> SettingsH

    FileH --> FileS
    GraphH --> GraphS
    WSH --> WSManager
    PerplexityH --> PerplexityS
    RagFlowH --> RagFlowS
    NostrH --> NostrS

    %% Connections to External Components
    FileS --> GitHub
    PerplexityS --> Perplexity
    RagFlowS --> RagFlow
    SpeechS --> OpenAI
    NostrS --> NostrAPI

    %% Styling for clarity
    style Frontend fill:#f9f,stroke:#333,stroke-width:2px
    style Backend fill:#bbf,stroke:#333,stroke-width:2px
    style External fill:#bfb,stroke:#333,stroke-width:2px
```

### Class Diagram

```mermaid
classDiagram
    class App {
        +sceneManager: SceneManager
        +nodeManager: EnhancedNodeManager
        +edgeManager: EdgeManager
        +hologramManager: HologramManager
        +textRenderer: TextRenderer
        +websocketService: WebSocketService
        +settingsStore: SettingsStore
        +platformManager: PlatformManager
        +xrSessionManager: XRSessionManager
        +start()
        +initializeEventListeners()
        +handleSettingsUpdate(settings: Settings)
        +dispose()
    }

    class SceneManager {
        -static instance: SceneManager
        +scene: Scene
        +camera: Camera
        +renderer: Renderer
        +controls: Controls
        +composer: Composer
        +getInstance(canvas: HTMLCanvasElement): SceneManager
        +getScene(): Scene
        +getRenderer(): Renderer
        +getCamera(): Camera
        +start()
        +handleSettingsUpdate(settings: Settings)
        +cleanup()
    }

    class WebsocketService {
        -static instance: WebsocketService
        +socket: WebSocket
        +listeners: Object
        +reconnectAttempts: number
        +maxReconnectAttempts: number
        +reconnectInterval: number
        +getInstance(): WebsocketService
        +connect()
        +onBinaryMessage(callback: function)
        +onSettingsUpdate(callback: function)
        +onConnectionStatusChange(callback: function)
        +sendMessage(data: object)
        +close()
    }

    class AppState {
        +graph_service: GraphService
        +gpu_compute: Option<Arc<RwLock<GPUCompute>>>
        +settings: Arc<RwLock<Settings>>
        +protected_settings: Arc<RwLock<ProtectedSettings>>
        +metadata: Arc<RwLock<MetadataStore>>
        +github_client: Arc<GitHubClient>
        +content_api: Arc<ContentAPI>
        +perplexity_service: Option<Arc<PerplexityService>>
        +ragflow_service: Option<Arc<RAGFlowService>>
        +nostr_service: Option<web::Data<NostrService>>
        +ragflow_conversation_id: String
        +active_connections: Arc<AtomicUsize>
        +new()
        +increment_connections(): usize
        +decrement_connections(): usize
        +get_api_keys(pubkey: str): ApiKeys
        +get_nostr_user(pubkey: str): Option<NostrUser>
        +validate_nostr_session(pubkey: str, token: str): bool
        +update_nostr_user_api_keys(pubkey: str, api_keys: ApiKeys): Result<NostrUser>
    }

    class GraphService {
        +build_graph(app_state: AppState): Result<GraphData>
        +calculate_layout(gpu_compute: GPUCompute, graph: GraphData, params: SimulationParams): Result<void>
        +initialize_random_positions(graph: GraphData)
    }

    class EnhancedNodeManager {
        +scene: Scene
        +settings: Settings
        +nodeMeshes: Map<string, Mesh>
        +updateNodes(nodes: Node[])
        +updateNodePositions(nodes: NodeData[])
        +handleSettingsUpdate(settings: Settings)
        +dispose()
    }

    class SpeechService {
        +websocketManager: WebSocketManager
        +settings: Settings
        +start(receiver: Receiver<SpeechCommand>)
        +initialize(): Result<void>
        +send_message(message: string): Result<void>
        +close(): Result<void>
        +set_tts_provider(use_openai: boolean): Result<void>
    }

    class NostrService {
        +settings: Settings
        +validate_session(pubkey: str, token: str): bool
        +get_user(pubkey: str): Option<NostrUser>
        +update_user_api_keys(pubkey: str, api_keys: ApiKeys): Result<NostrUser>
    }

    App --> SceneManager
    App --> WebsocketService
    App --> EnhancedNodeManager
    SceneManager --> WebXRVisualization
    WebsocketService --> GraphDataManager
    AppState --> GraphService
    AppState --> NostrService
    AppState --> SpeechService
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Client as Client (Browser)
    participant Platform as PlatformManager
    participant XR as XRSessionManager
    participant Scene as SceneManager
    participant Node as EnhancedNodeManager
    participant Edge as EdgeManager
    participant Hologram as HologramManager
    participant Text as TextRenderer
    participant WS as WebSocketService
    participant Settings as SettingsStore
    participant Server as Actix Server
    participant AppState as AppState
    participant FileH as FileHandler
    participant GraphH as GraphHandler
    participant WSH as WebSocketHandler
    participant PerplexityH as PerplexityHandler
    participant RagFlowH as RagFlowHandler
    participant NostrH as NostrHandler
    participant SettingsH as SettingsHandler
    participant FileS as FileService
    participant GraphS as GraphService
    participant GPUS as GPUService
    participant PerplexityS as PerplexityService
    participant RagFlowS as RagFlowService
    participant NostrS as NostrService
    participant SpeechS as SpeechService
    participant WSM as WebSocketManager
    participant GitHub as GitHub API
    participant Perplexity as Perplexity AI
    participant RagFlow as RagFlow API
    participant OpenAI as OpenAI API
    participant Nostr as Nostr API

    %% Server initialization and AppState setup
    activate Server
    Server->>Server: Load settings.yaml & env vars (config.rs)
    alt Settings Load Error
        Server-->>Client: Error Response (500)
    else Settings Loaded Successfully
        Server->>AppState: new() (app_state.rs)
        activate AppState
            AppState->>GPUS: initialize_gpu_compute()
            activate GPUS
                GPUS->>GPUS: setup_compute_pipeline()
                GPUS->>GPUS: load_wgsl_shaders()
                GPUS-->>AppState: GPU Compute Instance
            deactivate GPUS
            
            AppState->>WSM: initialize()
            activate WSM
                WSM->>WSM: setup_binary_protocol()
                WSM-->>AppState: WebSocket Manager
            deactivate WSM
            
            AppState->>SpeechS: start()
            activate SpeechS
                SpeechS->>SpeechS: initialize_tts()
                SpeechS-->>AppState: Speech Service
            deactivate SpeechS
            
            AppState->>NostrS: initialize()
            activate NostrS
                NostrS->>NostrS: setup_nostr_client()
                NostrS-->>AppState: Nostr Service
            deactivate NostrS
            
            AppState-->>Server: Initialized AppState
        deactivate AppState

        Server->>FileS: fetch_and_process_files()
        activate FileS
            FileS->>GitHub: fetch_files()
            activate GitHub
                GitHub-->>FileS: Files or Error
            deactivate GitHub
            
            loop For Each File
                FileS->>FileS: should_process_file()
                alt File Needs Processing
                    FileS->>PerplexityS: process_file()
                    activate PerplexityS
                        PerplexityS->>Perplexity: analyze_content()
                        Perplexity-->>PerplexityS: Analysis Results
                        PerplexityS-->>FileS: Processed Content
                    deactivate PerplexityS
                    FileS->>FileS: save_metadata()
                end
            end
            FileS-->>Server: Processed Files
        deactivate FileS

        Server->>GraphS: build_graph()
        activate GraphS
            GraphS->>GraphS: create_nodes_and_edges()
            GraphS->>GPUS: calculate_layout()
            activate GPUS
                GPUS->>GPUS: bind_gpu_buffers()
                GPUS->>GPUS: dispatch_compute_shader()
                GPUS->>GPUS: read_buffer_results()
                GPUS-->>GraphS: Updated Positions
            deactivate GPUS
            GraphS-->>Server: Graph Data
        deactivate GraphS
    end

    %% Client and Platform initialization
    Client->>Platform: initialize()
    activate Platform
        Platform->>Platform: detect_capabilities()
        Platform->>Settings: load_settings()
        activate Settings
            Settings->>Settings: validate_settings()
            Settings-->>Platform: Settings Object
        deactivate Settings
        
        Platform->>WS: connect()
        activate WS
            WS->>Server: ws_connect
            Server->>WSH: handle_connection()
            WSH->>WSM: register_client()
            WSM-->>WS: connection_established
            
            WS->>WS: setup_binary_handlers()
            WS->>WS: initialize_reconnection_logic()
            
            WSM-->>WS: initial_graph_data (Binary)
            WS->>WS: decode_binary_message()
        deactivate WS
        
        Platform->>XR: initialize()
        activate XR
            XR->>XR: check_xr_support()
            XR->>Scene: create()
            activate Scene
                Scene->>Scene: setup_three_js()
                Scene->>Scene: setup_render_pipeline()
                Scene->>Node: initialize()
                activate Node
                    Node->>Node: create_geometries()
                    Node->>Node: setup_materials()
                deactivate Node
                Scene->>Edge: initialize()
                activate Edge
                    Edge->>Edge: create_line_geometries()
                    Edge->>Edge: setup_line_materials()
                deactivate Edge
                Scene->>Hologram: initialize()
                activate Hologram
                    Hologram->>Hologram: setup_hologram_shader()
                    Hologram->>Hologram: create_hologram_geometry()
                deactivate Hologram
                Scene->>Text: initialize()
                activate Text
                    Text->>Text: load_fonts()
                    Text->>Text: setup_text_renderer()
                deactivate Text
            deactivate Scene
        deactivate XR
    deactivate Platform

    Note over Client, Nostr: User Interaction Flows

    %% User drags a node
    alt User Drags Node
        Client->>Node: handle_node_drag()
        Node->>WS: send_position_update()
        WS->>Server: binary_position_update
        Server->>GraphS: update_layout()
        GraphS->>GPUS: recalculate_forces()
        GPUS-->>Server: new_positions
        Server->>WSM: broadcast()
        WSM-->>WS: binary_update
        WS->>Node: update_positions()
        Node-->>Client: render_update
    end

    %% User asks a question
    alt User Asks Question
        Client->>RagFlowH: send_query()
        RagFlowH->>RagFlowS: process_query()
        activate RagFlowS
            RagFlowS->>RagFlow: get_context()
            RagFlow-->>RagFlowS: relevant_context
            RagFlowS->>OpenAI: generate_response()
            OpenAI-->>RagFlowS: ai_response
            RagFlowS-->>Client: streaming_response
        deactivate RagFlowS
        alt Speech Enabled
            Client->>SpeechS: synthesize_speech()
            activate SpeechS
                SpeechS->>OpenAI: text_to_speech()
                OpenAI-->>SpeechS: audio_stream
                SpeechS-->>Client: audio_data
            deactivate SpeechS
        end
    end

    %% User updates the graph
    alt User Updates Graph
        Client->>FileH: update_file()
        FileH->>FileS: process_update()
        FileS->>GitHub: create_pull_request()
        GitHub-->>FileS: pr_created
        FileS-->>Client: success_response
    end

    %% WebSocket reconnection flow
    alt WebSocket Reconnection
        WS->>WS: connection_lost()
        loop Until Max Attempts
            WS->>WS: attempt_reconnect()
            WS->>Server: ws_connect
            alt Connection Successful
                Server-->>WS: connection_established
                WSM-->>WS: resend_graph_data
                WS->>Node: restore_state()
            else Connection Failed
                Note right of WS: Continue reconnect attempts
            end
        end
    end

    %% Settings update flow
    alt Settings Update
        Client->>SettingsH: update_settings()
        SettingsH->>AppState: apply_settings()
        AppState->>WSM: broadcast_settings()
        WSM-->>WS: settings_update
        WS->>Settings: update_settings()
        Settings->>Platform: apply_platform_settings()
        Platform->>Scene: update_rendering()
        Scene->>Node: update_visuals()
        Scene->>Edge: update_visuals()
        Scene->>Hologram: update_effects()
    end

    %% Nostr authentication flow
    alt Nostr Authentication
        Client->>NostrH: authenticate()
        NostrH->>NostrS: validate_session()
        NostrS->>Nostr: verify_credentials()
        Nostr-->>NostrS: auth_result
        NostrS-->>Client: session_token
    end

    deactivate Server
```

### AR Features Implementation Status

#### Hand Tracking (Meta Quest 3)
- Implementation in `client/xr/xrSessionManager.ts`
- Currently addressing:
  - Performance optimization for AR passthrough mode
  - Virtual desktop cleanup during AR activation
  - Type compatibility between `XRHand` and custom `XRHandWithHaptics`
  - Joint position extraction methods

##### Current Challenges
- Type mismatches between standard `XRHand` and custom `XRHandWithHaptics`
- Joint position extraction from `XRJointSpace`
- Performance optimization in AR passthrough mode

##### Next Steps
- Implement adapter for `XRHand` to `XRHandWithHaptics` conversion
- Refactor VisualizationController for native XRHand compatibility
- Optimize AR mode transitions
- Enhance Meta Quest 3 performance

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Prof Rob Aspin: For inspiring the project's vision and providing valuable resources.
- OpenAI: For their advanced AI models powering the question-answering features.
- Perplexity AI and RAGFlow: For their AI services enhancing content processing and interaction.
- Three.js: For the robust 3D rendering capabilities utilized in the frontend.
- Actix: For the high-performance web framework powering the backend server.

### Authentication and Settings Inheritance

#### Unauthenticated Users
- Use browser's localStorage for settings persistence
- Settings are stored locally and not synced
- Default to basic settings visibility
- Limited to local visualization features

#### Authenticated Users (Nostr)
- Inherit settings from server's settings.yaml
- Settings are synced across all authenticated users
- Access to advanced settings based on role

#### Power Users
- Full access to all settings
- Can modify server's settings.yaml
- Access to advanced API features:
  - Perplexity API for AI assistance
  - RagFlow for document processing
  - GitHub integration for PR management
  - OpenAI voice synthesis
- Settings modifications are persisted to settings.yaml

### Settings Inheritance Flow

```mermaid
graph TD
    A[Start] --> B{Authenticated?}
    B -->|No| C[Load Local Settings]
    B -->|Yes| D[Load Server Settings]
    D --> E{Is Power User?}
    E -->|No| F[Apply Read-Only]
    E -->|Yes| G[Enable Full Access]
```

### Settings Sync Flow

```mermaid
graph TD
    A[Setting Changed] --> B{Authenticated?}
    B -->|No| C[Save Locally]
    B -->|Yes| D{Is Power User?}
    D -->|No| E[Preview Only]
    D -->|Yes| F[Update Server]
    F --> G[Sync to All Users]
```

### Modular Control Panel Architecture

The control panel is built with a modular architecture that supports:
- Detachable sections
- Real-time preview integration
- Drag and drop functionality
- Dynamic tooltips
- Performance optimizations

#### Component Structure

```typescript
interface ModularControlPanelProps {
  sections: ControlSection[];
  layout: LayoutConfig;
  onLayoutChange: (newLayout: LayoutConfig) => void;
}

interface ControlSection {
  id: string;
  title: string;
  settings: Setting[];
  isDetached: boolean;
  position?: { x: number, y: number };
  size?: { width: number, height: number };
}

interface Setting {
  id: string;
  type: 'slider' | 'toggle' | 'color' | 'select';
  value: any;
  metadata: SettingMetadata;
}
```

#### Layout Management

```typescript
interface LayoutConfig {
  sections: {
    [sectionId: string]: {
      position: { x: number, y: number };
      size: { width: number, height: number };
      isDetached: boolean;
      isCollapsed: boolean;
    };
  };
  userPreferences: {
    showAdvanced: boolean;
    activeFilters: string[];
    customOrder: string[];
  };
}
```

#### Performance Optimizations

- ResizeObserver for efficient size tracking
- Virtual scrolling for large setting lists
- Debounced real-time preview updates
- CSS transforms for smooth animations
- Lazy loading for visual aids
- Efficient memory management with WeakMap
- Real-time preview integration with ~60fps target