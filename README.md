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
- **Real-Time Updates** via WebSocket
- **GPU-Accelerated** graph layout with CPU fallback
- **One-Time File Pre-Processing** for GitHub file updates, comparing processed and raw files.

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

The project is divided into two main components: server-side and client-side. Here’s the file structure generated by the setup script.

### Server-Side

- **Controllers**: Handles incoming HTTP requests.
  - `GraphController`: Fetches graph data.
  - `FileController`: Fetches and processes files from GitHub.
  - `RAGFlowController`: Manages interactions with RAGFlow API.
  
- **Services**: Core business logic.
  - `GraphService`: Builds and updates the graph.
  - `FileService`: Fetches and processes files from GitHub and OpenWebUI.
  - `RAGFlowService`: Manages conversation with RAGFlow.
  
- **Models**: Data structures representing nodes, graphs, and metadata.
  - `GraphModel`, `MetadataModel`, `NodeModel`
  
- **Utilities**: Helper functions.
  - `WebSocketUtils`, `GPUUtils`

### Client-Side

- **WebXRVisualization**: Manages rendering and WebXR integration.
- **GraphSimulation**: Simulates the graph’s

 physics and layout.
- **Interface**: Handles user input.
- **ChatManager**: Manages chat and interaction with RAGFlow.
- **GraphDataManager**: Manages WebSocket communication.
- **ThreeJS Components**: Handles the Three.js scene setup and rendering.
  - `ThreeSetup`, `ThreeGraph`
  
### Tests

Unit tests are provided for all major components, both on the server and client side, under the `tests` directory.

## Installation and Setup

### Prerequisites

- Docker
- Node.js
- GitHub Personal Access Token
- RAGFlow API Key
- OpenWebUI API

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
