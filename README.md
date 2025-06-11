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
graph TD
    subgraph ClientApp [Frontend (TypeScript, React, R3F)]
        direction LR
        AppInit[AppInitializer.tsx]
        TwoPane[TwoPaneLayout.tsx]
        GraphView[GraphViewport.tsx]
        RightCtlPanel[RightPaneControlPanel.tsx]
        SettingsUI[SettingsPanelRedesign.tsx]
        ConvoPane[ConversationPane.tsx]
        NarrativePane[NarrativeGoldminePanel.tsx]
        SettingsMgr[settingsStore.ts]
        GraphDataMgr[GraphDataManager.ts]
        RenderEngine[Rendering Engine (GraphCanvas, GraphManager)]
        WebSocketSvc[WebSocketService.ts]
        APISvc[api.ts]
        NostrAuthSvcClient[nostrAuthService.ts]
        XRController[XRController.tsx]

        AppInit --> TwoPane
        AppInit --> SettingsMgr
        AppInit --> NostrAuthSvcClient
        AppInit --> WebSocketSvc
        AppInit --> GraphDataMgr

        TwoPane --> GraphView
        TwoPane --> RightCtlPanel
        TwoPane --> ConvoPane
        TwoPane --> NarrativePane
        RightCtlPanel --> SettingsUI

        SettingsUI --> SettingsMgr
        GraphView --> RenderEngine
        RenderEngine <--> GraphDataMgr
        GraphDataMgr <--> WebSocketSvc
        GraphDataMgr <--> APISvc
        NostrAuthSvcClient <--> APISvc
        XRController <--> RenderEngine
        XRController <--> SettingsMgr
    end

    subgraph ServerApp [Backend (Rust, Actix)]
        direction LR
        Actix[Actix Web Server]

        subgraph Handlers_Srv [API & WebSocket Handlers]
            direction TB
            SettingsH[SettingsHandler]
            NostrAuthH[NostrAuthHandler]
            GraphAPI_H[GraphAPI Handler]
            FilesAPI_H[FilesAPI Handler]
            RAGFlowH_Srv[RAGFlowHandler]
            SocketFlowH[SocketFlowHandler]
            SpeechSocketH[SpeechSocketHandler]
            HealthH[HealthHandler]
        end

        subgraph Services_Srv [Core Services]
            direction TB
            GraphSvc_Srv[GraphService (PhysicsEngine)]
            FileSvc_Srv[FileService]
            NostrSvc_Srv[NostrService]
            SpeechSvc_Srv[SpeechService]
            RAGFlowSvc_Srv[RAGFlowService]
            PerplexitySvc_Srv[PerplexityService]
        end

        subgraph Actors_Srv [Actor System]
            direction TB
            GraphServiceActor[GraphServiceActor]
            SettingsActor[SettingsActor]
            MetadataActor[MetadataActor]
            ClientManagerActor[ClientManagerActor]
            GPUComputeActor[GPUComputeActor]
            ProtectedSettingsActor[ProtectedSettingsActor]
        end
        AppState_Srv[AppState holds Addr<...>]

        Actix --> Handlers_Srv

        Handlers_Srv --> AppState_Srv
        SocketFlowH --> ClientManagerActor
        GraphAPI_H --> GraphServiceActor
        SettingsH --> SettingsActor
        NostrAuthH --> ProtectedSettingsActor

        GraphServiceActor --> ClientManagerActor
        GraphServiceActor --> MetadataActor
        GraphServiceActor --> GPUComputeActor
        GraphServiceActor --> SettingsActor

        FileSvc_Srv --> MetadataActor
        NostrSvc_Srv --> ProtectedSettingsActor
        SpeechSvc_Srv --> SettingsActor
        RAGFlowSvc_Srv --> SettingsActor
        PerplexitySvc_Srv --> SettingsActor
    end

    subgraph External_Srv [External Services]
        direction LR
        GitHub[GitHub API]
        NostrRelays_Ext[Nostr Relays]
        OpenAI[OpenAI API]
        PerplexityAI_Ext[Perplexity AI API]
        RAGFlow_Ext[RAGFlow API]
        Kokoro_Ext[Kokoro API]
    end

    WebSocketSvc <--> SocketFlowH
    APISvc <--> Actix

    FileSvc_Srv --> GitHub
    NostrSvc_Srv --> NostrRelays_Ext
    SpeechSvc_Srv --> OpenAI
    SpeechSvc_Srv --> Kokoro_Ext
    PerplexitySvc_Srv --> PerplexityAI_Ext
    RAGFlowSvc_Srv --> RAGFlow_Ext

    style ClientApp fill:#lightgrey,stroke:#333,stroke-width:2px
    style ServerApp fill:#lightblue,stroke:#333,stroke-width:2px
    style External_Srv fill:#lightgreen,stroke:#333,stroke-width:2px
```

### Class Diagram

```mermaid
classDiagram
    direction LR

    package "Frontend (TypeScript)" {
        class AppInitializer {
            <<React Component>>
            +initializeServices()
        }
        class GraphManager {
            <<React Component>>
            +renderNodesAndEdges()
        }
        class WebSocketService {
            <<Service>>
            +connect()
            +sendMessage()
            +onBinaryMessage()
            +isReady()
        }
        class SettingsStore {
            <<Zustand Store>>
            +settings: Settings
            +updateSettings()
        }
        class GraphDataManager {
            <<Service>>
            +fetchInitialData()
            +updateNodePositions()
            +getGraphData()
            +setWebSocketService()
        }
        class NostrAuthService {
            <<Service>>
            +loginWithNostr()
            +verifySession()
            +logout()
        }
        AppInitializer --> SettingsStore
        AppInitializer --> NostrAuthService
        AppInitializer --> WebSocketService
        AppInitializer --> GraphDataManager
        GraphDataManager --> WebSocketService
        GraphDataManager --> GraphManager
    }

    package "Backend (Rust)" {
        class AppState {
            <<Struct>>
            +graph_service_addr: Addr<GraphServiceActor>
            +settings_addr: Addr<SettingsActor>
            +metadata_addr: Addr<MetadataActor>
            +client_manager_addr: Addr<ClientManagerActor>
            +gpu_compute_addr: Option<Addr<GPUComputeActor>>
            +protected_settings_addr: Addr<ProtectedSettingsActor>
        }
        class GraphService {
            <<Struct>>
            +graph_data: Arc<RwLock<GraphData>>
            +start_simulation_loop()
            +broadcast_updates()
        }
        class PerplexityService {
            <<Struct>>
            +query()
        }
        class RagFlowService {
            <<Struct>>
            +chat()
        }
        class SpeechService {
            <<Struct>>
            +process_stt_request()
            +process_tts_request()
        }
        class NostrService {
            <<Struct>>
            +verify_auth_event()
            +validate_session()
            +manage_user_api_keys()
        }
        class GPUCompute {
            <<Struct>>
            +run_simulation_step()
        }
        class FileService {
            <<Struct>>
            +fetch_and_process_content()
            +update_metadata_store()
        }
        AppState --> GraphService : "<<holds>> Addr"
        AppState --> NostrService : "<<holds>> Addr"
        AppState --> PerplexityService : "<<holds>> Addr"
        AppState --> RagFlowService : "<<holds>> Addr"
        AppState --> SpeechService : "<<holds>> Addr"
        AppState --> GPUCompute : "<<holds>> Addr"
        AppState --> FileService : "<<holds>> Addr"

        WebSocketService -->> GraphServiceActor : "<<sends>> UpdateNodePositions"
        GraphService ..> GPUCompute : "uses (optional)"
        NostrService ..> ProtectedSettingsActor : "uses"
    }
```

### Sequence Diagrams

#### Server Initialization Sequence

```mermaid
sequenceDiagram
    participant Main as main.rs
    participant AppStateMod as app_state.rs
    participant ConfigMod as config/mod.rs
    participant Services as Various Services (Graph, File, Nostr, AI)
    participant ClientMgr as ClientManager (Static)
    participant GraphSvc as GraphService

    Main->>ConfigMod: AppFullSettings::load()
    ConfigMod-->>Main: loaded_settings
    Main->>AppStateMod: AppState::new(loaded_settings, /* other deps */)
    AppStateMod->>Services: Initialize FileService, NostrService, AI Services with configs
    AppStateMod->>GraphSvc: GraphService::new(settings, gpu_compute_opt, ClientMgr::instance())
    GraphSvc->>GraphSvc: Start physics_loop (async task)
    GraphSvc->>ClientMgr: (inside loop) Send updates
    AppStateMod-->>Main: app_state_instance
    Main->>ActixServer: .app_data(web::Data::new(app_state_instance))
```

#### Client Initialization Sequence

```mermaid
sequenceDiagram
    participant ClientApp as AppInitializer.tsx
    participant SettingsStoreSvc as settingsStore.ts
    participant NostrAuthSvcClient as nostrAuthService.ts
    participant WebSocketSvcClient as WebSocketService.ts
    participant ServerAPI as Backend REST API
    participant ServerWS as Backend WebSocket Handler

    ClientApp->>SettingsStoreSvc: Load settings (from localStorage & defaults)
    SettingsStoreSvc-->>ClientApp: Initial settings

    ClientApp->>NostrAuthSvcClient: Check current session (e.g., from localStorage)
    alt Session token exists
        NostrAuthSvcClient->>ServerAPI: POST /api/auth/nostr/verify (token)
        ServerAPI-->>NostrAuthSvcClient: Verification Result (user, features)
        NostrAuthSvcClient->>ClientApp: Auth status updated
    else No session token
        NostrAuthSvcClient->>ClientApp: Auth status (unauthenticated)
    end

    ClientApp->>WebSocketSvcClient: connect()
    WebSocketSvcClient->>ServerWS: WebSocket Handshake
    ServerWS-->>WebSocketSvcClient: Connection Established (e.g., `onopen`)
    WebSocketSvcClient->>WebSocketSvcClient: Set isConnected = true
    ServerWS-->>WebSocketSvcClient: Send {"type": "connection_established"} (or similar)
    WebSocketSvcClient->>WebSocketSvcClient: Set isServerReady = true

    alt WebSocket isReady()
        WebSocketSvcClient->>ServerWS: Send {"type": "requestInitialData"}
        ServerWS-->>WebSocketSvcClient: Initial Graph Data (e.g., large JSON or binary)
        WebSocketSvcClient->>GraphDataManager: Process initial data
    end
```

#### Real-time Graph Updates Sequence

```mermaid
sequenceDiagram
    participant ClientApp
    participant WebSocketSvcClient as WebSocketService.ts
    participant GraphDataMgrClient as GraphDataManager.ts
    participant ServerGraphSvc as GraphService (Backend)
    participant ServerGpuUtil as GPUCompute (Backend, Optional)
    participant ServerClientMgr as ClientManager (Backend, Static)
    participant ServerSocketFlowH as SocketFlowHandler (Backend)

    %% Continuous Server-Side Loop
    ServerGraphSvc->>ServerGraphSvc: physics_loop() iteration
    alt GPU Enabled
        ServerGraphSvc->>ServerGpuUtil: run_simulation_step()
        ServerGpuUtil-->>ServerGraphSvc: updated_node_data_from_gpu
    else CPU Fallback
        ServerGraphSvc->>ServerGraphSvc: calculate_layout_cpu()
    end
    ServerGraphSvc->>ServerClientMgr: BroadcastBinaryPositions(updated_node_data)

    ServerClientMgr->>ServerSocketFlowH: Distribute to connected clients
    ServerSocketFlowH-->>WebSocketSvcClient: Binary Position Update (Chunk)

    WebSocketSvcClient->>GraphDataMgrClient: onBinaryMessage(chunk)
    GraphDataMgrClient->>GraphDataMgrClient: Decompress & Parse chunk
    GraphDataMgrClient->>ClientApp: Notify UI/Renderer of position changes

    %% Optional: Client sends an update (e.g., user drags a node)
    opt User Interaction
        ClientApp->>GraphDataMgrClient: User moves node X to new_pos
        GraphDataMgrClient->>WebSocketSvcClient: sendRawBinaryData(node_X_new_pos_update) %% Or JSON message
        WebSocketSvcClient->>ServerSocketFlowH: Forward client update
        ServerSocketFlowH->>ServerGraphSvc: Apply client update to physics model (if supported)
    end
```

#### Authentication Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant ClientUI
    participant NostrAuthSvcClient as nostrAuthService.ts
    participant WindowNostr as "window.nostr (Extension)"
    participant APISvcClient as api.ts
    participant ServerNostrAuthH as NostrAuthHandler (Backend)
    participant ServerNostrSvc as NostrService (Backend)

    User->>ClientUI: Clicks Login Button
    ClientUI->>NostrAuthSvcClient: initiateLogin()
    NostrAuthSvcClient->>ServerNostrAuthH: GET /api/auth/nostr/challenge (via APISvcClient)
    ServerNostrAuthH-->>NostrAuthSvcClient: challenge_string

    NostrAuthSvcClient->>WindowNostr: signEvent(kind: 22242, content: "auth", tags:[["challenge", challenge_string], ["relay", ...]])
    WindowNostr-->>NostrAuthSvcClient: signed_auth_event

    NostrAuthSvcClient->>APISvcClient: POST /api/auth/nostr (signed_auth_event)
    APISvcClient->>ServerNostrAuthH: Forward request
    ServerNostrAuthH->>ServerNostrSvc: verify_auth_event(signed_auth_event)
    alt Event Valid
        ServerNostrSvc->>ServerNostrSvc: Generate session_token, store user session
        ServerNostrSvc-->>ServerNostrAuthH: AuthResponse (user, token, expiresAt, features)
        ServerNostrAuthH-->>APISvcClient: AuthResponse
        APISvcClient-->>NostrAuthSvcClient: AuthResponse
        NostrAuthSvcClient->>NostrAuthSvcClient: Store token, user data
        NostrAuthSvcClient->>ClientUI: Update auth state (Authenticated)
    else Event Invalid
        ServerNostrSvc-->>ServerNostrAuthH: Error
        ServerNostrAuthH-->>APISvcClient: Error Response
        APISvcClient-->>NostrAuthSvcClient: Error
        NostrAuthSvcClient->>ClientUI: Show Login Error
    end
```

#### Settings Synchronization Sequence

```mermaid
sequenceDiagram
    participant User
    participant ClientUI
    participant SettingsStoreClient as settingsStore.ts
    participant SettingsSvcClient as settingsService.ts (part of api.ts or separate)
    participant ServerSettingsH as SettingsHandler (Backend)
    participant ServerAppState as AppState (Backend)
    participant ServerUserSettings as UserSettings Model (Backend)
    participant ServerClientMgr as ClientManager (Backend, Static, for broadcast if applicable)

    User->>ClientUI: Modifies a setting (e.g., node size)
    ClientUI->>SettingsStoreClient: updateSettings({ visualisation: { nodes: { nodeSize: newValue }}})
    SettingsStoreClient->>SettingsStoreClient: Update local state (Zustand) & persist to localStorage

    alt User is Authenticated
        SettingsStoreClient->>SettingsSvcClient: POST /api/user-settings/sync (ClientSettingsPayload)
        SettingsSvcClient->>ServerSettingsH: Forward request
        ServerSettingsH->>ServerAppState: Get current AppFullSettings / UserSettings
        alt User is PowerUser
            ServerSettingsH->>ServerAppState: Update AppFullSettings in memory
            ServerAppState->>ServerAppState: AppFullSettings.save() to settings.yaml
            ServerSettingsH-->>SettingsSvcClient: Updated UISettings (reflecting global)
            %% Optional: Server broadcasts global settings change if implemented
            %% ServerAppState->>ServerClientMgr: BroadcastGlobalSettingsUpdate(updated_AppFullSettings)
            %% ServerClientMgr-->>OtherClients: Global settings update message
        else Regular User
            ServerSettingsH->>ServerUserSettings: Load or create user's UserSettings file
            ServerUserSettings->>ServerUserSettings: Update UISettings part of UserSettings
            ServerUserSettings->>ServerUserSettings: Save UserSettings to user-specific YAML
            ServerSettingsH-->>SettingsSvcClient: Updated UISettings (user-specific)
        end
        SettingsSvcClient-->>SettingsStoreClient: Confirmation / Updated settings (if different)
        %% Client store might re-sync if server response indicates changes
    end
```

### AR Features Implementation Status

#### Hand Tracking (Meta Quest 3)
- XR Interaction is primarily managed by [`client/src/features/xr/systems/HandInteractionSystem.tsx`](client/src/features/xr/systems/HandInteractionSystem.tsx:1) and related hooks/providers like [`useSafeXRHooks.tsx`](client/src/features/xr/hooks/useSafeXRHooks.tsx:1).
- Session management is in [`client/src/features/xr/managers/xrSessionManager.ts`](client/src/features/xr/managers/xrSessionManager.ts:1).
- Initialisation logic is in [`client/src/features/xr/managers/xrInitializer.ts`](client/src/features/xr/managers/xrInitializer.ts:1).
- The main XR entry point and controller is [`client/src/features/xr/components/XRController.tsx`](client/src/features/xr/components/XRController.tsx:1).
- Type definitions for WebXR, including hand tracking, are in [`client/src/features/xr/types/xr.ts`](client/src/features/xr/types/xr.ts:1) and potentially augmented by [`client/src/features/xr/types/webxr-extensions.d.ts`](client/src/features/xr/types/webxr-extensions.d.ts:1) (though this file is noted as mostly commented out).

##### Current Challenges
- The `webxr-extensions.d.ts` file is largely commented out, indicating potential gaps or reliance on default browser types for hand tracking APIs, which might vary.
- Robust gesture recognition based on joint positions requires significant implementation in `HandInteractionSystem.tsx`.

##### Next Steps
- Review and complete necessary type definitions in `webxr-extensions.d.ts` if standard types are insufficient.
- Implement gesture recognition logic.
- Optimize performance for AR/passthrough modes.

### Authentication and Settings Inheritance

#### Unauthenticated Users
- Use browser's localStorage for settings persistence (via Zustand `persist` middleware in [`client/src/store/settingsStore.ts`](client/src/store/settingsStore.ts:1)).
- Settings are stored locally and not synced to a user-specific backend store.
- Default to basic settings visibility.
- Limited to local visualisation features; AI and GitHub features requiring API keys will not be available unless default API keys are configured in the server's `ProtectedSettings`.

#### Authenticated Users (Nostr)
- **Regular Users**:
    - Settings are loaded from and saved to user-specific files on the server (e.g., `/app/user_settings/<pubkey>.yaml`), managed by [`src/handlers/settings_handler.rs`](src/handlers/settings_handler.rs:1) using the `UserSettings` model (which contains `UISettings`).
    - These user-specific settings are primarily UI/visualisation preferences defined in `UISettings`.
    - Can access features based on their configuration in [`src/config/feature_access.rs`](src/config/feature_access.rs:1).
    - Can manage their own API keys for AI services via the `/api/auth/nostr/api-keys` endpoint. These keys are stored in their `NostrUser` profile within the server's `ProtectedSettings`.
- **Power Users**:
    - Directly load and modify the global server settings from `settings.yaml` (represented by `AppFullSettings` in Rust, which is then used to derive `UISettings`).
    - Have full access to all settings and advanced API features. API keys for these might come from `AppFullSettings` (if globally configured for all power users) or their own `NostrUser` profile in `ProtectedSettings`.
    - Settings modifications made by power users to `AppFullSettings` are persisted to the main `settings.yaml` and potentially broadcast to other clients (if implemented).

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

The client's user interface for settings and controls is structured as follows:
-   **Main Layout**: [`client/src/app/TwoPaneLayout.tsx`](client/src/app/TwoPaneLayout.tsx:1) divides the screen.
-   **Right Pane Host**: [`client/src/app/components/RightPaneControlPanel.tsx`](client/src/app/components/RightPaneControlPanel.tsx:1) hosts various panels within the right-hand side.
-   **Settings UI Core**: [`client/src/features/settings/components/panels/SettingsPanelRedesign.tsx`](client/src/features/settings/components/panels/SettingsPanelRedesign.tsx:1) provides the tabbed interface for different setting categories (Visualisation, System, AI, XR).
    -   **Tabs Component**: Uses a generic [`client/src/ui/Tabs.tsx`](client/src/ui/Tabs.tsx:1) component for tab navigation.
    -   **Settings Sections**: Each tab within `SettingsPanelRedesign.tsx` renders one or more [`SettingsSection.tsx`](client/src/features/settings/components/SettingsSection.tsx:1) components to group related settings. These sections can be collapsible.
    -   **Individual Controls**: Each [`SettingsSection.tsx`](client/src/features/settings/components/SettingsSection.tsx:1) uses multiple [`SettingControlComponent.tsx`](client/src/features/settings/components/SettingControlComponent.tsx:1) instances to render the actual UI controls (sliders, toggles, inputs, etc.) for each setting.
-   **State Management**:
    -   Settings values are primarily managed by the Zustand store defined in [`client/src/store/settingsStore.ts`](client/src/store/settingsStore.ts:1).
    -   Context for control panel specific state (like detached panel states or advanced view toggles) is managed by [`client/src/features/settings/components/control-panel-context.tsx`](client/src/features/settings/components/control-panel-context.tsx:1).

The `client/src/components/layout/ControlPanel.tsx` mentioned in the original README seems to be superseded or refactored into the `SettingsPanelRedesign.tsx` and its constituent parts. Detachable sections are a feature of `SettingsSection.tsx`.

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
