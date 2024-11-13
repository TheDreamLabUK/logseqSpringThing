# WebXR Graph Visualization of Logseq Knowledge Graphs with RAGFlow Integration

![WebXR Graph Visualization](https://github.com/user-attachments/assets/39dd3101-d616-46d6-8d7a-1d02998eb8d9)

Inspired by Prof Rob Aspin's work 
https://github.com/trebornipsa

![P1080785_1728030359430_0](https://github.com/user-attachments/assets/3ecac4a3-95d7-4c75-a3b2-e93deee565d6)

## Project Overview

The **WebXR Graph Visualization** project transforms a Logseq personal knowledge base into an interactive 3D graph, accessible in mixed-reality environments. The system automates the parsing of Markdown files from a privately hosted GitHub repository, enhances the content using the **Perplexity AI API**, and integrates with **RAGFlow** for AI-powered question answering. Processed changes are submitted back to the source repository as pull requests (PRs).

**Key Features:**

Inspired by Prof Rob Aspin's work 
https://github.com/trebornipsa

Integrates Sonata Rust wrapper for Piper
https://github.com/mush42/sonata

![P1080785_1728030359430_0](https://github.com/user-attachments/assets/3ecac4a3-95d7-4c75-a3b2-e93deee565d6)

- **3D Visualization:** Dynamic representation of knowledge graph nodes and edges with real-time updates.
- **WebXR Compatibility:** Immersive exploration on Augmented Reality (AR) and Virtual Reality (VR) devices.
- **Efficient WebSocket Communication:** Facilitates dynamic node position updates and real-time interactions.
- **GPU Acceleration:** Enhances performance on both server and client sides using WebGPU.
- **Node Labels as Billboards:** Clear and interactive identification of nodes within the graph.
- **Integration with RAGFlow:** Enables AI-powered question answering directly within the graph interface.
- **Spacemouse Support:** Offers intuitive navigation within immersive environments.
- **Automatic GitHub PR Submissions:** Streamlines the process of updating processed content back to GitHub.
- **Comprehensive Metadata Management:** Handles both processed and raw JSON metadata for enhanced data representation.
- **OpenAI Integration:** Provides text-to-speech capabilities for enhanced accessibility.

## Architecture

The project comprises a **Rust-based server** running within a Docker container and a **JavaScript client-side application**. The architecture emphasizes GPU acceleration, efficient real-time updates, and immersive AR experiences.

### Class Diagram

```mermaid
classDiagram
    class App {
        - websocketService: WebsocketService
        - graphDataManager: GraphDataManager
        - visualization: WebXRVisualization
        - chatManager: ChatManager
        - interface: Interface
        - ragflowService: RAGFlowService
        + start()
        - initializeEventListeners()
        - toggleFullscreen()
    }

    class WebsocketService {
        - socket: WebSocket
        - listeners: Object
        - reconnectAttempts: number
        - maxReconnectAttempts: number
        - reconnectInterval: number
        + connect()
        + on(event: string, callback: function)
        + emit(event: string, data: any)
        + send(data: object)
        - reconnect()
    }

    class GraphDataManager {
        - websocketService: WebsocketService
        - graphData: GraphData
        + requestInitialData()
        + updateGraphData(newData: GraphData)
        + getGraphData(): GraphData
    }

    class WebXRVisualization {
        - graphDataManager: GraphDataManager
        - nodeManager: NodeManager
        - layoutManager: LayoutManager
        - effectsManager: EffectManager
        - lastPositionUpdate: number
        - positionUpdateThreshold: number
        + handleBinaryPositionUpdate(buffer: ArrayBuffer)
        + handleNodeDrag(nodeId: string, position: Vector3)
        + updateSettings(settings: object)
        + animate()
    }

    class NodeManager {
        - nodeMeshes: Map<string, THREE.Mesh>
        - nodeLabels: Map<string, THREE.Sprite>
        - edgeMeshes: Map<string, THREE.Line>
        - oldestDate: number
        - newestDate: number
        + updateNodePositions(updates: PositionUpdate[])
        + updateNodePosition(nodeId: string, position: Vector3)
        + getNodePositions(): PositionUpdate[]
        + calculateNodeColor(lastModified: Date): THREE.Color
    }

    class LayoutManager {
        - lastPositions: Position[]
        - updateThreshold: number
        - lastUpdateTime: number
        - updateInterval: number
        - currentGraphData: GraphData
        + performLayout(graphData: GraphData)
        + updateNodePosition(nodeId: string, position: Vector3)
        + sendPositionUpdates(nodes: Node[])
        + applyPositionUpdates(buffer: ArrayBuffer)
    }

    class ChatManager {
        - websocketService: WebsocketService
        - ragflowService: RAGflowService
        + sendMessage(message: string)
        + receiveMessage()
        - handleIncomingMessage(message: string)
    }

    class Interface {
        - chatManager: ChatManager
        - visualization: WebXRVisualization
        + handleUserInput(input: string)
        + displayChatMessage(message: string)
        - setupEventListeners()
        - renderUI()
    }

    class RAGFlowService {
        - settings: Settings
        - apiClient: ApiClient
        + askQuestion(question: string): Promise<string>
        + processAnswer(answer: string): string
    }

    class GraphService {
        + build_graph_from_metadata(metadata: Map): GraphData
        + calculate_layout(gpu_compute: GPUCompute, graph: GraphData)
        - initialize_random_positions(graph: GraphData)
    }

    class PerplexityService {
        + process_markdown(file_content: &str, settings: &Settings, api_client: &dyn ApiClient): Result<String, PerplexityError>
        + process_markdown_block(input: &str, prompt: &str, topics: &[String], api_response: &str): String
    }

    class FileService {
        + process_files(github_files: Vec<GithubFile>, settings: &Settings, metadata_map: &mut HashMap<String, Metadata>): Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>>
        + should_process_file(file: &GithubFile): bool
        + strip_double_brackets(content: &str): String
        + process_against_topics(content: &str, metadata_map: &HashMap<String, Metadata>): String
        + count_hyperlinks(content: &str): usize
        + count_topics(content: &str, metadata_map: &HashMap<String, Metadata>): HashMap<String, usize>
    }

    class GPUCompute {
        - device: GPUDevice
        - nodes_buffer: GPUBuffer
        - position_update_buffer: GPUBuffer
        - position_pipeline: GPUComputePipeline
        + update_positions(binary_data: Uint8Array)
        + get_position_updates(): ArrayBuffer
        + step()
    }

    App --> WebsocketService
    App --> GraphDataManager
    App --> WebXRVisualization
    App --> ChatManager
    App --> Interface
    App --> RAGFlowService
    WebsocketService --> GraphDataManager
    GraphDataManager --> WebXRVisualization
    ChatManager --> RAGFlowService
    Interface --> ChatManager
    Interface --> WebXRVisualization
    App --> GraphService
    App --> PerplexityService
    App --> FileService
    WebXRVisualization --> NodeManager
    WebXRVisualization --> LayoutManager
    WebXRVisualization --> WebSocketService
    LayoutManager --> WebSocketService
    WebSocketService --> GPUCompute
    GraphService --> GPUCompute
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Client
    participant WebXRVisualization
    participant NodeManager
    participant LayoutManager
    participant WebSocket
    participant Server
    participant GPUCompute

    Note over Client,GPUCompute: Initial Setup Phase

    Client->>WebXRVisualization: initialize()
    activate WebXRVisualization
        WebXRVisualization->>WebSocket: connect()
        WebSocket->>Server: establish connection
        Server->>GPUCompute: initialize()
        Server-->>Client: connection established
        
        WebXRVisualization->>WebSocket: request initial data
        WebSocket->>Server: getInitialData
        Server->>GPUCompute: calculate_initial_layout()
        GPUCompute-->>Server: initial positions
        Server-->>Client: graph data + binary positions
        
        WebXRVisualization->>NodeManager: initialize(graphData)
        WebXRVisualization->>LayoutManager: initialize(graphData)
    deactivate WebXRVisualization

    Note over Client,GPUCompute: Real-time Position Update Flow (60fps)

    Client->>WebXRVisualization: nodeDragged event
    activate WebXRVisualization
        WebXRVisualization->>NodeManager: updateNodePosition()
        WebXRVisualization->>LayoutManager: handleNodeDrag()
        
        alt Time since last update >= 16.67ms
            LayoutManager->>LayoutManager: prepare binary position data
            LayoutManager->>WebSocket: send binary position update
            WebSocket->>Server: binary message
            
            Server->>GPUCompute: update_positions(binary)
            GPUCompute->>GPUCompute: validate positions
            GPUCompute-->>Server: validated positions
            
            Server->>Server: broadcast to other clients
            Server-->>Client: binary position update
            
            WebXRVisualization->>NodeManager: handleBinaryPositionUpdate()
            NodeManager->>NodeManager: update visual positions
        end
    deactivate WebXRVisualization

    Note over Client,GPUCompute: Continuous Physics Simulation

    loop Every animation frame
        GPUCompute->>GPUCompute: step()
        GPUCompute-->>Server: new positions
        Server-->>Client: binary position broadcast
        
        WebXRVisualization->>NodeManager: updateNodePositions()
        WebXRVisualization->>WebXRVisualization: render()
    end
```

## Installation

### Prerequisites

Ensure that the following dependencies are installed on your system:

- **Rust** (version 1.70 or later)
- **Node.js** (version 14 or later)
- **Docker** (for containerization)
- **Git** (for version control)

### Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/webxr-graph.git
    cd webxr-graph
    ```

2. **Configure Environment Variables:**

    Create a `.env` file in the root directory and populate it with your API keys and configurations.

    ```env
    PERPLEXITY_API_KEY=your_perplexity_api_key
    GITHUB_ACCESS_TOKEN=your_github_token
    RAGFLOW_API_KEY=your_ragflow_api_key
    RAGFLOW_API_BASE_URL=your_ragflow_base_url
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=https://api.openai.com/v1
    TUNNEL_TOKEN=your_cloudflare_tunnel_token
    DOMAIN=your_domain_name
    ```

    **Note:** Ensure that sensitive information like API keys is **never** hardcoded and is managed securely.

3. **Update Configuration File:**

    Ensure that `settings.toml` is correctly configured with the necessary fields. Refer to the Settings Configuration section for details.

4. **Build the Rust Server:**

    ```bash
    cargo build --release
    ```

5. **Run the Server Locally:**

    ```bash
    cargo run --release
    ```

6. **Start the Client Application:**

    Navigate to the client directory and install dependencies.

    ```bash
    cd client
    npm install
    npm start
    ```

7. **Building and Running with Docker:**

    Ensure Docker is installed and running on your system.

    ```bash
    ./launch-docker.sh
    ```

### Security Verification

1. **Container Security Checklist:**
   - [ ] Containers running as non-root
   - [ ] Read-only filesystem enabled
   - [ ] Network properly isolated
   - [ ] Resource limits enforced
   - [ ] Health checks passing
   - [ ] Proper logging configured
   - [ ] Secure volume mounts
   - [ ] Cloudflare tunnel active

2. **File Permissions:**
```bash
# Verify directory permissions
ls -la /app/data

# Check configuration files
ls -la /etc/logseq-security/

# Verify log permissions
ls -la /var/log/logseq-security/
```

3. **Network Security:**
```bash
# Check network isolation
docker network inspect logseq-net

# Verify exposed ports
docker port logseqXR
```

### Security Maintenance

1. **Regular Updates:**
```bash
# Update security components
./setup-security.sh --update

# Rebuild with latest security patches
docker compose build --no-cache
```

2. **Monitoring:**
```bash
# View security logs
journalctl -u security-monitor

# Check container health
docker compose ps

# Monitor resource usage
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```

[Rest of the README remains unchanged...]
