# Class Diagrams

This document provides detailed class diagrams and relationships for the major components of LogseqXR.

## Core Application Structure

```mermaid
classDiagram
    class App {
        +websocketService: WebsocketService
        +graphDataManager: GraphDataManager
        +visualization: WebXRVisualization
        +chatManager: ChatManager
        +interface: Interface
        +ragflowService: RAGFlowService
        +nostrAuthService: NostrAuthService
        +settingsStore: SettingsStore
        +start()
        +initializeEventListeners()
        +toggleFullscreen()
    }
    class WebsocketService {
        +socket: WebSocket
        +listeners: Object
        +reconnectAttempts: number
        +maxReconnectAttempts: number
        +reconnectInterval: number
        +connect()
        +on(event: string, callback: function)
        +emit(event: string, data: any)
        +send(data: object)
        +reconnect()
    }
    class GraphDataManager {
        +websocketService: WebsocketService
        +graphData: GraphData
        +requestInitialData()
        +updateGraphData(newData: GraphData)
        +getGraphData(): GraphData
        +recalculateLayout()
        +updateForceDirectedParams(name: string, value: any)
    }
    class WebXRVisualization {
        +graphDataManager: GraphDataManager
        +scene: Scene
        +camera: Camera
        +renderer: Renderer
        +controls: Controls
        +composer: Composer
        +gpu: GPUUtilities
        +nodeManager: EnhancedNodeManager
        +edgeManager: EdgeManager
        +hologramManager: HologramManager
        +textRenderer: TextRenderer
        +initialize()
        +updateVisualization()
        +initThreeJS()
        +setupGPU()
        +initPostProcessing()
        +addLights()
        +createHologramStructure()
        +handleSpacemouseInput(x: number, y: number, z: number)
        +handleBinaryPositionUpdate(buffer: ArrayBuffer)
        +animate()
        +updateVisualFeatures(control: string, value: any)
        +onWindowResize()
        +handleNodeDrag(nodeId: string, position: Vector3)
        +getNodePositions(): PositionUpdate[]
        +showError(message: string)
    }

    App --> WebsocketService
    App --> GraphDataManager
    App --> WebXRVisualization
    App --> NostrAuthService
    App --> SettingsStore
    GraphDataManager --> WebsocketService
    WebXRVisualization --> GraphDataManager
```

## Backend Services

```mermaid
classDiagram
    class GraphService {
        +build_graph(app_state: AppState): Result<GraphData, Error>
        +calculate_layout(gpu_compute: GPUCompute, graph: GraphData, params: SimulationParams): Result<void, Error>
        +initialize_random_positions(graph: GraphData)
    }
    class PerplexityService {
        +process_file(file: ProcessedFile, settings: Settings, api_client: ApiClient): Result<ProcessedFile, Error>
    }
    class FileService {
        +fetch_and_process_files(github_service: GitHubService, settings: Settings, metadata_map: Map<String, Metadata>): Result<Vec<ProcessedFile>, Error>
        +load_or_create_metadata(): Result<Map<String, Metadata>, Error>
        +save_metadata(metadata: Map<String, Metadata>): Result<void, Error>
        +calculate_node_size(file_size: number): number
        +extract_references(content: string, valid_nodes: String[]): Map<String, ReferenceInfo>
        +convert_references_to_topic_counts(references: Map<String, ReferenceInfo>): Map<String, number>
        +initialize_local_storage(github_service: GitHubService, settings: Settings): Result<void, Error>
        +count_hyperlinks(content: string): number
    }
    class NostrService {
        +settings: Settings
        +validate_session(pubkey: str, token: str): bool
        +get_user(pubkey: str): Option<NostrUser>
        +update_user_api_keys(pubkey: str, api_keys: ApiKeys): Result<NostrUser>
    }
    class GitHubService {
        +fetch_file_metadata(): Result<Vec<GithubFileMetadata>, Error>
        +get_download_url(file_name: string): Result<string, Error>
        +fetch_file_content(download_url: string): Result<string, Error>
        +get_file_last_modified(file_path: string): Result<Date, Error>
    }
    class GitHubPRService {
        +create_pull_request(file_name: string, content: string, original_sha: string): Result<string, Error>
    }
    class ApiClient {
        +post_json(url: string, body: PerplexityRequest, perplexity_api_key: string): Result<string, Error>
    }
    class SpeechService {
        +websocketManager: WebSocketManager
        +settings: Settings
        +start(receiver: Receiver<SpeechCommand>)
        +initialize(): Result<void, Error>
        +send_message(message: string): Result<void, Error>
        +close(): Result<void, Error>
        +set_tts_provider(use_openai: boolean): Result<void, Error>
    }

    GraphService --> GPUCompute
    PerplexityService --> ApiClient
    FileService --> GitHubService
    GitHubPRService --> GitHubService
    SpeechService --> WebSocketManager
```

## Frontend Components

```mermaid
classDiagram
    class ChatManager {
        +websocketService: WebsocketService
        +ragflowService: RAGFlowService
        +sendMessage(message: string)
        +receiveMessage()
        +handleIncomingMessage(message: string)
    }
    class Interface {
        +chatManager: ChatManager
        +visualization: WebXRVisualization
        +controlPanel: ModularControlPanel
        +handleUserInput(input: string)
        +displayChatMessage(message: string)
        +setupEventListeners()
        +renderUI()
        +updateNodeInfoPanel(node: object)
        +displayErrorMessage(message: string)
    }
    class ModularControlPanel {
        +settingsStore: SettingsStore
        +validationDisplay: ValidationErrorDisplay
        +sections: Map<string, SectionConfig>
        +initializePanel()
        +initializeNostrAuth()
        +updateAuthUI(user: NostrUser)
        +createSection(config: SectionConfig, paths: string[])
        +toggleDetached(sectionId: string)
        +toggleCollapsed(sectionId: string)
        +show()
        +hide()
        +dispose()
    }
    class NostrAuthService {
        +currentUser: NostrUser
        +eventEmitter: SettingsEventEmitter
        +settingsPersistence: SettingsPersistenceService
        +initialize()
        +login(): Promise<AuthResult>
        +logout()
        +getCurrentUser(): NostrUser
        +isAuthenticated(): boolean
        +isPowerUser(): boolean
        +hasFeatureAccess(feature: string): boolean
        +checkAuthStatus(pubkey: string)
        +onAuthStateChanged(callback: function)
    }
    class SettingsStore {
        +settings: Settings
        +observers: Set<Observer>
        +initialize()
        +get(path: string): any
        +set(path: string, value: any)
        +subscribe(observer: Observer)
        +unsubscribe(observer: Observer)
        +validate(settings: Settings)
        +persist()
    }
    class RAGFlowService {
        +settings: Settings
        +apiClient: ApiClient
        +createConversation(userId: string): Promise<string>
        +sendMessage(conversationId: string, message: string): Promise<string>
        +getConversationHistory(conversationId: string): Promise<object>
    }

    Interface --> ChatManager
    Interface --> WebXRVisualization
    Interface --> ModularControlPanel
    ChatManager --> RAGFlowService
    ModularControlPanel --> SettingsStore
    ModularControlPanel --> NostrAuthService
```

## WebSocket Components

```mermaid
classDiagram
    class SpeechWs {
        +websocketManager: WebSocketManager
        +settings: Settings
        +hb(ctx: Context)
        +check_heartbeat(ctx: Context)
        +started(ctx: Context)
        +handle(msg: Message, ctx: Context)
    }
    class WebSocketManager {
        +connections: Map<String, WebSocket>
        +add_connection(id: String, ws: WebSocket)
        +remove_connection(id: String)
        +broadcast(message: Message)
        +send_to(id: String, message: Message)
    }

    SpeechWs --> WebSocketManager
```

## Key Relationships

- The `App` class serves as the main entry point and coordinates all major components
- `WebsocketService` handles real-time communication between frontend and backend
- `GraphDataManager` manages the graph data structure and coordinates with the visualization
- `WebXRVisualization` handles the 3D rendering and XR interactions
- `ModularControlPanel` provides a flexible UI with dockable sections and Nostr authentication
- `NostrAuthService` manages user authentication and feature access
- `SettingsStore` provides centralized settings management
- Backend services are organized around specific responsibilities (files, graph, AI, auth, etc.)
- Frontend components handle user interaction and visualization updates

## Related Documentation
- [Technical Architecture](../overview/architecture.md)
- [Development Setup](../development/setup.md)
- [API Documentation](../api/rest.md)