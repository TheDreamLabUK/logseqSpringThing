LogseqXR Networking and Data Flow Briefing
This document outlines the networking architecture and data flow for the LogseqXR application, clarifying the roles of REST and WebSockets.

1. Overall Architecture

The application follows a client-server model, with the server responsible for data storage, processing, and settings management, while the client handles visualization and user interaction. Communication occurs through REST API calls for initial setup and settings management, and WebSockets for real-time position updates.

2. Server-Side (Rust)

Data Storage: Graph data (nodes, edges, metadata) is stored on the server, potentially in a database or file system. Settings are stored in settings.toml.

REST API (actix-web): The server exposes a REST API for:

Graph Data: /api/graph/data (full graph) and /api/graph/data/paginated (paginated graph).

Settings: /api/visualization/settings (GET all settings), /api/visualization/settings (PUT all settings), and individual setting endpoints like /api/visualization/{category}/{setting} (GET/PUT).

Other API endpoints: /api/files/fetch, /api/chat/*, /api/perplexity.

WebSocket Handling (actix-web-actors): The server uses WebSockets only for real-time, binary position and velocity updates. The ws_handler function establishes WebSocket connections and handles incoming/outgoing messages. It uses a binary protocol for efficiency.

Periodic Updates: The server periodically checks for updates to the graph data (e.g., from GitHub) and updates the graph data store accordingly. This is independent of the WebSocket connection.

Settings Broadcast: The server broadcasts settings updates to all connected clients via WebSockets whenever settings are changed through the REST API. This uses the SettingsBroadcaster and the settingsUpdated message type.

3. Client-Side (TypeScript)

Initialization:

The client loads initial graph data from /api/graph/data/paginated using pagination.

The client loads all visualization settings from /api/visualization/settings.

REST API Interaction: The client uses REST API calls for:

Initial Graph Data: Retrieving the initial graph data using pagination.

Settings: Loading all settings, getting individual settings, updating individual settings, and updating all settings at once.

WebSocket Connection: The client establishes a WebSocket connection to the server for receiving real-time position updates and sending client position updates.

Control Panel: The ControlPanel component interacts with the SettingsManager to display and update settings. It uses REST API calls to get and update settings.

Visualization: The Three.js visualization components (SceneManager, NodeManager, etc.) subscribe to settings changes in the SettingsManager and update the visualization accordingly.

Case Conversion: The client handles case conversion between camelCase (TypeScript) and snake_case (Rust) using utility functions.

4. Data Flow Diagrams

sequenceDiagram
    participant Client
    participant Server

    alt Initial Setup
        Client->>Server: GET /api/visualization/settings (all settings)
        Server-->>Client: Settings (camelCase)
        Client->>Server: GET /api/graph/data/paginated?page=0&pageSize=100
        Server-->>Client: Paginated Graph Data (camelCase)
    end
    
    alt User Updates Setting
        Client->>Server: PUT /api/visualization/{category}/{setting} (snake_case, new value)
        Server-->>Client: Updated Setting Value (wrapped in JSON, camelCase)
        Server->>WebSocket: Broadcast settingsUpdated (all settings, camelCase)
        WebSocket->>Client: settingsUpdated (all settings, camelCase)
    end

    alt Periodic Graph Update
        Server->>GitHub: Fetch updated graph data
        Server->>Server: Update graph data store
        Server->>WebSocket: Broadcast graphUpdated (if changes)
        WebSocket->>Client: graphUpdated
    end

    alt Realtime Position Updates
        Client->>WebSocket: Send binary position/velocity data
        WebSocket->>Server: Binary data
        Server->>WebSocket: Broadcast binary position/velocity updates
        WebSocket->>Client: Binary position/velocity updates
    end
Use code with caution.
Mermaid
5. Key Improvements

Clear Separation: WebSockets are now exclusively used for binary position/velocity updates, simplifying the communication model.

REST for Settings: All settings interactions are handled via REST, improving reliability and simplifying the client-side logic.

Case Conversion: Consistent use of case conversion functions ensures correct data exchange between client and server.

Broadcast Mechanism: The server broadcasts settings updates to all clients, ensuring synchronization.

6. Remaining Challenges

Dynamic UI Updates: The client-side control panel UI still needs a mechanism to dynamically re-render after settings changes. This might involve a reactive UI library or manual DOM manipulation.

Error Handling: Robust error handling should be implemented for all network requests and WebSocket communication.

Testing: Thorough testing of all data flows and synchronization mechanisms is crucial.

This briefing document provides a clear overview of the LogseqXR networking architecture and data flow. By addressing the remaining challenges, the application should achieve robust settings management and real-time visualization updates.