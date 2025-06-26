# Repository Map

This document provides a visual map of the repository, generated from the project's structure and README file. The diagrams below illustrate the high-level architecture, component relationships, and data flows within the system.

## System Architecture

This diagram provides a high-level overview of the entire system, including the frontend, backend, and external services.

```mermaid
graph TD
    subgraph ClientApp ["Frontend"]
        direction LR
        AppInit[AppInitializer]
        TwoPane[TwoPaneLayout]
        GraphView[GraphViewport]
        RightCtlPanel[RightPaneControlPanel]
        SettingsUI[SettingsPanelRedesign]
        ConvoPane[ConversationPane]
        NarrativePane[NarrativeGoldminePanel]
        SettingsMgr[settingsStore]
        GraphDataMgr[GraphDataManager]
        RenderEngine[GraphCanvas & GraphManager]
        WebSocketSvc[WebSocketService]
        APISvc[api]
        NostrAuthSvcClient[nostrAuthService]
        XRController[XRController]

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

    subgraph ServerApp ["Backend"]
        direction LR
        Actix[ActixWebServer]

        subgraph Handlers_Srv ["API_WebSocket_Handlers"]
            direction TB
            SettingsH[SettingsHandler]
            NostrAuthH[NostrAuthHandler]
            GraphAPI_H[GraphAPIHandler]
            FilesAPI_H[FilesAPIHandler]
            RAGFlowH_Srv[RAGFlowHandler]
            SocketFlowH[SocketFlowHandler]
            SpeechSocketH[SpeechSocketHandler]
            HealthH[HealthHandler]
        end

        subgraph Services_Srv ["Core_Services"]
            direction TB
            GraphSvc_Srv[GraphService]
            FileSvc_Srv[FileService]
            NostrSvc_Srv[NostrService]
            SpeechSvc_Srv[SpeechService]
            RAGFlowSvc_Srv[RAGFlowService]
            PerplexitySvc_Srv[PerplexityService]
        end

        subgraph Actors_Srv ["Actor_System"]
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

    subgraph External_Srv ["External_Services"]
        direction LR
        GitHub[GitHubAPI]
        NostrRelays_Ext[NostrRelays]
        OpenAI[OpenAIAPI]
        PerplexityAI_Ext[PerplexityAIAPI]
        RAGFlow_Ext[RAGFlowAPI]
        Kokoro_Ext[KokoroAPI]
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

## Class Diagram

This diagram shows the key classes and their relationships in the frontend and backend.

```mermaid
classDiagram
    direction LR

    %% Frontend Classes
    class AppInitializer {
        <<ReactComponent>>
        +initializeServices()
    }
    class GraphManager {
        <<ReactComponent>>
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
        <<ZustandStore>>
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

    %% Backend Classes
    class AppState {
        <<Struct>>
        +graph_service_addr: Addr_GraphServiceActor
        +settings_addr: Addr_SettingsActor
        +metadata_addr: Addr_MetadataActor
        +client_manager_addr: Addr_ClientManagerActor
        +gpu_compute_addr: Option_Addr_GPUComputeActor
        +protected_settings_addr: Addr_ProtectedSettingsActor
    }
    class GraphService {
        <<Struct>>
        +graph_data: Arc_RwLock_GraphData
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
    AppState --> GraphService : holds_Addr
    AppState --> NostrService : holds_Addr
    AppState --> PerplexityService : holds_Addr
    AppState --> RagFlowService : holds_Addr
    AppState --> SpeechService : holds_Addr
    AppState --> GPUCompute : holds_Addr
    AppState --> FileService : holds_Addr

    WebSocketService ..> GraphServiceActor : sends_UpdateNodePositions
    GraphService ..> GPUCompute : uses_optional
    NostrService ..> ProtectedSettingsActor : uses
```
