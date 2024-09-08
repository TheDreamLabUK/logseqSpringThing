# WebXR Graph Visualization of Logseq Knowledge Graphs with RAGFlow Integration

This project visualizes privately hosted GitHub Markdown files created by LogSeq and integrates with RAGFlow for question answering capabilities in a 3D, WebXR-compatible environment.

![image](https://github.com/user-attachments/assets/fcbd6eb1-e2a1-4fea-a3df-4303b17e2b48)

![image](https://github.com/user-attachments/assets/873809d5-d8bd-44c3-884c-ce9418e273ef)

![image](./optimized-output.gif)

## Project Overview

This application transforms a LogSeq personal knowledge base into an interactive 3D graph, viewable in mixed reality. It automatically parses pages from a private GitHub repository, processes them via OpenWebUI, and creates a force-directed 3D graph using WebXR and Three.js. The processed and raw files are analyzed, and JSON metadata is generated for both versions, enabling a comparison of graph nodes and edges.

Key features include:
- **3D Visualization** of knowledge graph nodes and edges
- **WebXR Compatibility** for immersive exploration
- **OpenWebUI API Integration** for file pre-processing
- **Integration with RAGFlow** for AI-powered question answering
- **Real-Time Updates** via WebSocket for both client and server
- **Mandatory GPU Acceleration** on the server-side for graph computations
- **Optional GPU Acceleration** on the client-side for enhanced performance
- **One-Time File Pre-Processing** for GitHub file updates, comparing processed and raw files

## Key Technical Features

### WebSocket Integration

WebSockets are crucial for real-time communication between the client and server:

- **Server-side** (`websocketUtils.js`): Manages WebSocket connections, broadcasts updates to connected clients, and handles incoming messages.
- **Client-side** (`websocketService.js` and `graphDataManager.js`): Establishes connection with the server, sends user actions, and receives real-time graph updates.

### GPU Acceleration

- **Server-side** (`gpuUtils.js`): Mandatory GPU acceleration for efficient graph computations, force calculations, and position updates.
- **Client-side** (`gpuUtils.js`): Optional GPU acceleration for enhanced performance in graph rendering and local computations. Falls back to CPU processing if GPU is not available.


## Architecture

The project consists of server-side and client-side components, with well-defined classes and services based on the diagram below:

### Class Diagram

```mermaid
classDiagram
    class Server {
        +start()
        +initialize()
        +listen(port: int)
        +setupWebSocket()
    }

    class App {
        +initialize()
        +setMiddleware()
        +setRoutes()
        +startServer()
    }

    class GraphController {
        +getGraphData(req: Request, res: Response)
        +refreshGraph(req: Request, res: Response)
    }

    class FileController {
        +fetchAndProcessFiles(req: Request, res: Response)
    }

    class RAGFlowController {
        +sendMessage(req: Request, res: Response)
    }

    class GraphService {
        +getGraphData()
        +refreshGraphData()
        +buildEdges()
    }

    class FileService {
        +fetchFilesFromGitHub()
        +compareAndIdentifyUpdates(githubFiles: Array)
        +sendToOpenWebUI(file: String)
        +saveFileMetadata(metadata: Object)
    }

    class RAGFlowService {
        +createConversation(userId: String)
        +sendMessage(conversationId: String, message: String)
        +getChatHistory(conversationId: String)
    }

    class OpenWebUiService {
        +processFile(file: String)
    }

    class GraphModel {
        +graphData: Object
        +edges: Array
        +nodes: Array
    }

    class MetadataModel {
        +fileName: String
        +lastModified: Date
        +processedFile: String
        +originalFile: String
    }

    class NodeModel {
        +id: String
        +label: String
        +metadata: Object
    }

    class WebSocketUtils {
        +setupWebSocket()
        +broadcastMessage(message: String)
    }

    class GPUUtils {
        +initializeGPU()
        +computeForces()
        +updatePositions()
    }

    class Client {
        +start()
        +initialize()
        +connectWebSocket()
    }

    class WebXRVisualization {
        +initScene()
        +initCamera()
        +initRenderer()
        +updateGraph(data: Object)
        +animate()
    }

    class GraphSimulation {
        +initialize(nodes: Array, edges: Array)
        +updateGraph()
        +tick()
    }

    class Interface {
        +initSpaceMouse()
        +handleInput(event: Object)
        +updateCameraPosition()
    }

    class ChatManager {
        +initializeChat()
        +sendMessage(question: String)
        +updateChatDisplay(response: Object)
    }

    class GraphDataManager {
        +connectWebSocket()
        +handleMessage(event: Object)
        +sendStartSimulation()
        +sendStopSimulation()
        +sendSetSimulationParams(params: Object)
    }

    class ThreeSetup {
        +initThreeScene()
        +initThreeCamera()
    }

    class ThreeGraph {
        +renderGraph(nodes: Array, edges: Array)
    }

    class XRSetup {
        +initXR()
    }

    class XRInteraction {
        +handleXRInput()
    }

    Server --> App
    Server --> WebSocketUtils
    Server --> GPUUtils
    App --> GraphController
    App --> FileController
    App --> RAGFlowController
    GraphController --> GraphService
    FileController --> FileService
    RAGFlowController --> RAGFlowService
    FileService --> OpenWebUiService
    FileService --> MetadataModel
    GraphService --> GraphModel
    GraphModel --> NodeModel
    WebXRVisualization --> GraphSimulation
    WebXRVisualization --> Interface
    WebXRVisualization --> ChatManager
    WebXRVisualization --> GraphDataManager
    GraphDataManager --> Client
    WebXRVisualization --> ThreeSetup
    WebXRVisualization --> ThreeGraph
    WebXRVisualization --> XRSetup
    WebXRVisualization --> XRInteraction

```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant WebXRVisualization
    participant GraphDataManager
    participant Server
    participant ServerGraphSimulation
    participant GitHub
    participant OpenWebUIAPI
    participant RAGFlowIntegration

    activate Server
    Server->>Server: initialize()
    Server->>Server: initializeHttpsOptions()
    Server->>Server: initializeGPU()
    Server->>Server: create HTTPS server & WebSocket server
    Server->>Server: listen on port 8443
    Server->>Server: refreshGraphData()
    Server->>GitHub: fetchMarkdownMetadata()
    GitHub-->>Server: Markdown file metadata
    Server->>Server: compareAndIdentifyUpdates()
    loop for each file to update
        Server->>GitHub: fetch file content
        GitHub-->>Server: file content
        Server->>Server: save file content & metadata
        
        %% OpenWebUI Processing Step %%
        Server->>OpenWebUIAPI: send file for processing
        OpenWebUIAPI-->>Server: processed file & JSON metadata
        Server->>Server: store processed file & metadata
        Server->>Server: generate edges and nodes from raw & processed files
    end
    Server->>Server: buildEdges()
    Server->>Server: loadGraphData()
    Server->>ServerGraphSimulation: initialize(graphData)
    Server->>ServerGraphSimulation: startSimulation()
    
    activate Client
    Client->>WebXRVisualization: initialize()
    WebXRVisualization->>GraphSimulation: initialize()
    WebXRVisualization->>Interface: initialize()
    WebXRVisualization->>ChatManager: initialize()
    WebXRVisualization->>GraphDataManager: initialize()
    
    GraphDataManager->>Server: WebSocket connection
    Server-->>GraphDataManager: Connection established
    
    Client->>Server: /graph-data (HTTP)
    Server-->>Client: graph data (JSON)
    
    GraphDataManager->>Server: { type: 'startSimulation' } (WebSocket)
    
    loop Simulation loop
        ServerGraphSimulation->>ServerGraphSimulation: computeForces()
        ServerGraphSimulation->>ServerGraphSimulation: updatePositions()
        ServerGraphSimulation->>GraphDataManager: { type: 'nodePositions', positions: [...] } (WebSocket)
        GraphDataManager->>GraphSimulation: updateNodePositions(positions)
        GraphSimulation->>WebXRVisualization: updateGraph()
    end
    
    Client->>ChatManager: sendMessage(question)
    ChatManager->>Server: /api/chat/message (HTTP)
    Server->>RAGFlowIntegration: sendMessage(conversationId, message)
    RAGFlowIntegration-->>Server: response
    Server-->>ChatManager: response (HTTP)
    ChatManager->>WebXRVisualization: updateChatDisplay(response)
    
    Client->>Interface: user input (e.g., SpaceMouse movement)
    Interface->>WebXRVisualization: updateCameraPosition()
    
    Client->>Server: /refresh-graph (HTTP)
    Server->>Server: refreshGraphData()
    Server->>GitHub: fetchMarkdownMetadata()
    GitHub-->>Server: Markdown file metadata
    Server->>Server: compareAndIdentifyUpdates()
    loop for each file to update
        Server->>GitHub: fetch file content
        GitHub-->>Server: file content
        Server->>Server: save file content & metadata
        
        %% OpenWebUI Processing Step %%
        Server->>OpenWebUIAPI: send file for processing
        OpenWebUIAPI-->>Server: processed file & JSON metadata
        Server->>Server: store processed file & metadata
        Server->>Server: generate edges and nodes from raw & processed files
    end
    Server->>Server: buildEdges()
    Server->>Server: loadGraphData()
    Server->>ServerGraphSimulation: updateGraphData(newGraphData)
    ServerGraphSimulation->>GraphDataManager: { type: 'graphUpdate', data: newGraphData } (WebSocket)
    GraphDataManager->>GraphSimulation: updateData(newGraphData)
    GraphSimulation->>WebXRVisualization: updateGraph()
    
    deactivate Client
    deactivate Server

```

## File Structure

### Server-Side

- **Controllers**: `server/src/controllers/`
  - `graphController.js`: Handles graph data requests
  - `fileController.js`: Manages file operations and GitHub interactions
  - `ragflowController.js`: Handles RAGFlow API interactions

- **Services**: `server/src/services/`
  - `graphService.js`: Core graph processing and management
  - `fileService.js`: File handling and OpenWebUI integration
  - `ragflowService.js`: RAGFlow conversation management
  - `openWebUiService.js`: Interaction with OpenWebUI API

- **Models**: `server/src/models/`
  - `graphModel.js`: Graph data structure
  - `metadataModel.js`: File metadata representation
  - `nodeModel.js`: Graph node structure

- **Utilities**: `server/src/utils/`
  - `websocketUtils.js`: Server-side WebSocket management
  - `gpuUtils.js`: GPU acceleration for server-side computations

### Client-Side

- **Core**: `public/js/`
  - `index.js`: Entry point for client-side application
  - `app.js`: Main application setup and initialization

- **Components**: `public/js/components/`
  - `webXRVisualization.js`: Manages WebXR rendering and interactions
  - `graphSimulation.js`: Handles graph physics and layout
  - `interface.js`: User input handling
  - `chatManager.js`: Manages chat interface and RAGFlow interactions

- **Services**: `public/js/services/`
  - `graphDataManager.js`: Manages graph data and WebSocket communication
  - `websocketService.js`: Client-side WebSocket handling

- **ThreeJS Components**: `public/js/threeJS/`
  - `threeSetup.js`: Three.js scene initialization
  - `threeGraph.js`: Three.js graph rendering

- **XR Components**: `public/js/xr/`
  - `xrSetup.js`: WebXR session setup
  - `xrInteraction.js`: XR-specific interaction handling

- **Utilities**: `public/js/`
  - `gpuUtils.js`: Optional GPU acceleration for client-side computations
  
### Tests

Unit tests are provided for all major components, both on the server and client side, under the `tests` directory.

## Installation and Setup

### Prerequisites

- Docker
- Node.js
- GitHub Personal Access Token
- RAGFlow API Key
- OpenWebUI API
- GPU-enabled server for mandatory server-side acceleration
- (Optional) GPU-enabled client device for enhanced performance

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/webxr-graph.git
   cd webxr-graph
   ```

2. Create a `.env` file in the root directory:
   ```
   GITHUB_ACCESS_TOKEN=your_token_here
   GITHUB_OWNER=your_github_username
   GITHUB_REPO=your_repo_name
   GITHUB_DIRECTORY=path/to/markdown/files
   RAGFLOW_API_KEY=your_ragflow_api_key_here
   RAGFLOW_BASE_URL=http://your_ragflow_base_url/v1/
   OPENWEBUI_API=http://your_openwebui_url/
   ```

3. Build and run with Docker:
   ```bash
   ./start_docker.sh
   ```

4. Access the application at `https://localhost:8443` using a WebXR-compatible browser.

## Development Status

The project is under active development. Areas of focus include:
- Finalizing the integration with OpenWebUI for file processing.
- Expanding unit tests and improving test coverage.
- Optimizing the GPU-based graph simulation for larger datasets.

## Contributing

Contributions are welcome! Please submit issues or pull requests.

## License

This project is licensed under the Creative Commons CC0 license.

---

This updated `README.md` reflects the architecture and design changes, including the OpenWebUI integration and the separation of concerns across various components. Let me know if you’d like to refine anything further!
