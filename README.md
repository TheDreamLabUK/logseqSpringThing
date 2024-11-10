# WebXR Graph Visualization of Logseq Knowledge Graphs with RAGFlow Integration

![WebXR Graph Visualization](https://github.com/user-attachments/assets/39dd3101-d616-46d6-8d7a-1d02998eb8d9)

Inspired by Prof Rob Aspin's work 
https://github.com/trebornipsa

![P1080785_1728030359430_0](https://github.com/user-attachments/assets/3ecac4a3-95d7-4c75-a3b2-e93deee565d6)

## Project Overview

The **WebXR Graph Visualization** project transforms a LogSeq personal knowledge base into an interactive 3D graph, accessible in mixed reality environments. The system automates the parsing of Markdown files from a privately hosted GitHub repository, enhances the content using the **Perplexity AI API**, and integrates with **RAGFlow** for AI-powered question answering. Processed changes are submitted back to the source repository as pull requests (PRs).

**Key Features:**

Inspired by Prof Rob Aspin's work 
https://github.com/trebornipsa

Integrates Sonata rust wrapper for Piper
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

[Rest of the README content remains unchanged...]
=======
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
    ```

    **Note:** Ensure that sensitive information like API keys is **never** hardcoded and is managed securely.

3. **Update Configuration File:**

    Ensure that `settings.toml` is correctly configured with the necessary fields. Refer to the [Settings Configuration](#settings-configuration) section for details.

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

## Settings Configuration

The application relies on a `settings.toml` file for configuration. Below is an example configuration with explanations for each section.

```toml:settings.toml
# settings.toml

# Prompt for the AI assistant
prompt = """
You are an AI assistant building summaries of web links and text. You will visit any web links found in the text and integrate
a summary with web citations, aiming for up to two citations explicitly returned in context as raw web hyperlinks.
Ensure to return web links as citations separated by new lines.
You should aim to select one or more of these topics in this form appropriate to the created summary,
embedding the topic in Logseq double square brackets once in the returned text.
"""

# List of topics to embed in the summary
topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Rust Programming",
    "Web Development",
    "WebXR",
    "Three.js",
    "GPU Computing",
    "Graph Visualization",
    "Markdown Processing"
]

[perplexity]
perplexity_api_key = "your_perplexity_api_key"
perplexity_model = "llama-3.1-sonar-small-128k-online"
perplexity_api_base_url = "https://api.perplexity.ai/chat/completions"
perplexity_max_tokens = 4096
perplexity_temperature = 0.7
perplexity_top_p = 1.0
perplexity_presence_penalty = 0.0
perplexity_frequency_penalty = 0.0

[github]
github_access_token = "your_github_token"
github_owner = "your_github_owner"
github_repo = "your_github_repo"
github_directory = "your_github_directory"

[ragflow]
ragflow_api_key = "your_ragflow_api_key"
ragflow_api_base_url = "https://api.ragflow.com/v1"

[openai]
api_key = "your_openai_api_key"
base_url = "https://api.openai.com/v1"

# Default configurations (can be overridden by environment variables)
[default]
max_concurrent_requests = 5
max_retries = 3
retry_delay = 5
api_client_timeout = 30

[visualization]
node_color = "0x1A0B31"
edge_color = "0xff0000"
hologram_color = "0xFFD700"
node_size_scaling_factor = 1000
hologram_scale = 5
hologram_opacity = 0.1
edge_opacity = 0.3
label_font_size = 36
fog_density = 0.002
```

### Explanation of Configuration Sections

- **Prompt Section:**
  - Defines the system prompt for the AI assistant, guiding its behavior in summarizing content and embedding topics.

- **Topics:**
  - A list of topics that the AI assistant should focus on when generating summaries.

- **Perplexity:**
  - **perplexity_api_key:** Your API key for the Perplexity AI service.
  - **perplexity_model:** The model used by Perplexity for processing.
  - **perplexity_api_base_url:** The base URL for the Perplexity API.
  - **perplexity_max_tokens:** Maximum number of tokens to generate.
  - **perplexity_temperature, perplexity_top_p, perplexity_presence_penalty, perplexity_frequency_penalty:** Parameters controlling the randomness and creativity of the AI's responses.

- **GitHub:**
  - **github_access_token:** Your GitHub access token for repository interactions.
  - **github_owner:** Owner of the GitHub repository.
  - **github_repo:** Name of the GitHub repository.
  - **github_directory:** Directory within the repository to monitor and update.

- **RAGFlow:**
  - **ragflow_api_key:** API key for RAGFlow integration.
  - **ragflow_api_base_url:** Base URL for the RAGFlow API.

- **OpenAI:**
  - **api_key:** Your OpenAI API key for accessing OpenAI services.
  - **base_url:** Base URL for the OpenAI API.

- **Default:**
  - **max_concurrent_requests:** Maximum number of concurrent API requests.
  - **max_retries:** Number of retry attempts for failed requests.
  - **retry_delay:** Delay between retry attempts in seconds.
  - **api_client_timeout:** Timeout for API client requests in seconds.

- **Visualization:**
  - **node_color, edge_color, hologram_color:** Color codes for nodes, edges, and holograms in the visualization.
  - **node_size_scaling_factor:** Scaling factor for node sizes.
  - **hologram_scale:** Scale factor for hologram visuals.
  - **hologram_opacity, edge_opacity:** Opacity levels for holograms and edges.
  - **label_font_size:** Font size for node labels.
  - **fog_density:** Density of fog effects in the visualization.

## Development Status

The project is under active development with recent and ongoing enhancements:

**Recent Improvements:**

- **Enhanced Test Coverage:** Comprehensive tests for both server and client components, ensuring reliability and facilitating future development.
- **Integration Tests:** Established end-to-end workflow tests to validate the complete application flow.
- **Mocking Enhancements:** Improved mocking for API interactions within `PerplexityService` and `RAGFlowService`.
- **GPU Compute Integration:** Successfully integrated GPUCompute for accelerated graph layout calculations using WebGPU.

**Ongoing Focus Areas:**

- **Optimizing WebGPU Integration:** Enhancing server-side graph computations for better performance using WebGPU.
- **Finalizing Perplexity Integration:** Ensuring seamless processing and enhancement of Markdown files via the Perplexity AI API.
- **Expanding Unit Tests:** Continuously improving test coverage to encompass all new features and components.
- **Performance Enhancements:** Further optimizing the Rust-based server for scalability and efficiency.
- **RAGFlow Integration Refinement:** Enhancing the integration with RAGFlow for more robust AI-powered question answering capabilities.

## Testing

Our test suite adheres to best practices for Test-Driven Development (TDD), ensuring robust and reliable application behavior through comprehensive coverage of both success and error scenarios.

### Server-Side Tests (Rust)

Located in the `tests/server/` directory:

- `graph_service_test.rs`: Tests for graph building and management.
- `ragflow_service_test.rs`: Tests for RAGFlow integration.
- `perplexity_service_test.rs`: Tests for Perplexity API interactions.
- `file_service_test.rs`: Tests for GitHub file processing.

### Client-Side Tests (JavaScript)

Located in the `tests/client/` directory:

- `interface.test.js`: Tests for user interface components.
- `graphService.test.js`: Tests for client-side graph data management.
- `websocketService.test.js`: Tests for WebSocket communication.
- ... (additional client-side test files)

## Usage

### Running Locally

1. **Start the Rust Server:**

    ```bash
    cargo run --release
    ```

2. **Start the Client Application:**

    ```bash
    cd client
    npm install
    npm start
    ```

3. **Access the Application:**

    Navigate to `http://localhost:3000` in your web browser to interact with the 3D knowledge graph.

### Using Docker

1. **Build the Docker Image:**

    ```bash
    docker build -t webxr-graph-image .
    ```

2. **Run the Docker Container:**

    ```bash
    docker run -d \
      -p 8080:8080 \
      -e PERPLEXITY_API_KEY=your_perplexity_api_key \
      -e GITHUB_ACCESS_TOKEN=your_github_token \
      -e RAGFLOW_API_KEY=your_ragflow_api_key \
      -e RAGFLOW_API_BASE_URL=your_ragflow_base_url \
      -e OPENAI_API_KEY=your_openai_api_key \
      -e OPENAI_BASE_URL=https://api.openai.com/v1 \
      webxr-graph-image
    ```

    **Note:** Ensure that sensitive information like API keys are managed securely and not hardcoded within Docker images.

## Settings Configuration

The application relies on a `settings.toml` file for configuration. Below is an example configuration with explanations for each section.

```toml:settings.toml
# settings.toml

# Prompt for the AI assistant
prompt = """
You are an AI assistant building summaries of web links and text. You will visit any web links found in the text and integrate
a summary with web citations, aiming for up to two citations explicitly returned in context as raw web hyperlinks.
Ensure to return web links as citations separated by new lines.
You should aim to select one or more of these topics in this form appropriate to the created summary,
embedding the topic in Logseq double square brackets once in the returned text.
"""

# List of topics to embed in the summary
topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Rust Programming",
    "Web Development",
    "WebXR",
    "Three.js",
    "GPU Computing",
    "Graph Visualization",
    "Markdown Processing"
]

[perplexity]
perplexity_api_key = "your_perplexity_api_key"
perplexity_model = "llama-3.1-sonar-small-128k-online"
perplexity_api_base_url = "https://api.perplexity.ai/chat/completions"
perplexity_max_tokens = 4096
perplexity_temperature = 0.7
perplexity_top_p = 1.0
perplexity_presence_penalty = 0.0
perplexity_frequency_penalty = 0.0

[github]
github_access_token = "your_github_token"
github_owner = "your_github_owner"
github_repo = "your_github_repo"
github_directory = "your_github_directory"

[ragflow]
ragflow_api_key = "your_ragflow_api_key"
ragflow_api_base_url = "https://api.ragflow.com/v1"

[openai]
api_key = "your_openai_api_key"
base_url = "https://api.openai.com/v1"

# Default configurations (can be overridden by environment variables)
[default]
max_concurrent_requests = 5
max_retries = 3
retry_delay = 5
api_client_timeout = 30

[visualization]
node_color = "0x1A0B31"
edge_color = "0xff0000"
hologram_color = "0xFFD700"
node_size_scaling_factor = 1000
hologram_scale = 5
hologram_opacity = 0.1
edge_opacity = 0.3
label_font_size = 36
fog_density = 0.002
```

### Explanation of Configuration Sections

- **Prompt Section:**
  - Defines the system prompt for the AI assistant, guiding its behavior in summarizing content and embedding topics.

- **Topics:**
  - A list of topics that the AI assistant should focus on when generating summaries.

- **Perplexity:**
  - **perplexity_api_key:** Your API key for the Perplexity AI service.
  - **perplexity_model:** The model used by Perplexity for processing.
  - **perplexity_api_base_url:** The base URL for the Perplexity API.
  - **perplexity_max_tokens:** Maximum number of tokens to generate.
  - **perplexity_temperature, perplexity_top_p, perplexity_presence_penalty, perplexity_frequency_penalty:** Parameters controlling the randomness and creativity of the AI's responses.

- **GitHub:**
  - **github_access_token:** Your GitHub access token for repository interactions.
  - **github_owner:** Owner of the GitHub repository.
  - **github_repo:** Name of the GitHub repository.
  - **github_directory:** Directory within the repository to monitor and update.

- **RAGFlow:**
  - **ragflow_api_key:** API key for RAGFlow integration.
  - **ragflow_api_base_url:** Base URL for the RAGFlow API.

- **OpenAI:**
  - **api_key:** Your OpenAI API key for accessing OpenAI services.
  - **base_url:** Base URL for the OpenAI API.

- **Default:**
  - **max_concurrent_requests:** Maximum number of concurrent API requests.
  - **max_retries:** Number of retry attempts for failed requests.
  - **retry_delay:** Delay between retry attempts in seconds.
  - **api_client_timeout:** Timeout for API client requests in seconds.

- **Visualization:**
  - **node_color, edge_color, hologram_color:** Color codes for nodes, edges, and holograms in the visualization.
  - **node_size_scaling_factor:** Scaling factor for node sizes.
  - **hologram_scale:** Scale factor for hologram visuals.
  - **hologram_opacity, edge_opacity:** Opacity levels for holograms and edges.
  - **label_font_size:** Font size for node labels.
  - **fog_density:** Density of fog effects in the visualization.

## Development Status

The project is under active development with recent and ongoing enhancements:

**Recent Improvements:**

- **Enhanced Test Coverage:** Comprehensive tests for both server and client components, ensuring reliability and facilitating future development.
- **Integration Tests:** Established end-to-end workflow tests to validate the complete application flow.
- **Mocking Enhancements:** Improved mocking for API interactions within `PerplexityService` and `RAGFlowService`.
- **GPU Compute Integration:** Successfully integrated GPUCompute for accelerated graph layout calculations using WebGPU.

**Ongoing Focus Areas:**

- **Optimizing WebGPU Integration:** Enhancing server-side graph computations for better performance using WebGPU.
- **Finalizing Perplexity Integration:** Ensuring seamless processing and enhancement of Markdown files via the Perplexity AI API.
- **Expanding Unit Tests:** Continuously improving test coverage to encompass all new features and components.
- **Performance Enhancements:** Further optimizing the Rust-based server for scalability and efficiency.
- **RAGFlow Integration Refinement:** Enhancing the integration with RAGFlow for more robust AI-powered question answering capabilities.

## Testing

Our test suite adheres to best practices for Test-Driven Development (TDD), ensuring robust and reliable application behavior through comprehensive coverage of both success and error scenarios.

### Server-Side Tests (Rust)

Located in the `tests/server/` directory:

- `graph_service_test.rs`: Tests for graph building and management.
- `ragflow_service_test.rs`: Tests for RAGFlow integration.
- `perplexity_service_test.rs`: Tests for Perplexity API interactions.
- `file_service_test.rs`: Tests for GitHub file processing.

### Client-Side Tests (JavaScript)

Located in the `tests/client/` directory:

- `interface.test.js`: Tests for user interface components.
- `graphService.test.js`: Tests for client-side graph data management.
- `websocketService.test.js`: Tests for WebSocket communication.
- ... (additional client-side test files)

## Usage

### Running Locally

1. **Start the Rust Server:**

    ```bash
    cargo run --release
    ```

2. **Start the Client Application:**

    ```bash
    cd client
    npm install
    npm start
    ```

3. **Access the Application:**

    Navigate to `http://localhost:3000` in your web browser to interact with the 3D knowledge graph.

### Using Docker

1. **Build the Docker Image:**

    ```bash
    docker build -t webxr-graph-image .
    ```

2. **Run the Docker Container:**

    ```bash
    docker run -d \
      -p 8080:8080 \
      -e PERPLEXITY_API_KEY=your_perplexity_api_key \
      -e GITHUB_ACCESS_TOKEN=your_github_token \
      -e RAGFLOW_API_KEY=your_ragflow_api_key \
      -e RAGFLOW_API_BASE_URL=your_ragflow_base_url \
      -e OPENAI_API_KEY=your_openai_api_key \
      -e OPENAI_BASE_URL=https://api.openai.com/v1 \
      webxr-graph-image
    ```

    **Note:** Ensure that sensitive information like API keys are managed securely and not hardcoded within Docker images.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository:** Click the "Fork" button at the top right of this page to create your own fork.
2. **Clone Your Fork:**

    ```bash
    git clone https://github.com/yourusername/webxr-graph.git
    cd webxr-graph
    ```
    ```bash
    git clone https://github.com/yourusername/webxr-graph.git
    cd webxr-graph
    ```

3. **Create a New Branch:**

    ```bash
    git checkout -b feature/your-feature-name
    ```
    ```bash
    git checkout -b feature/your-feature-name
    ```

4. **Make Your Changes:** Implement your feature or fix.
5. **Commit Your Changes:**

    ```bash
    git commit -m "Add feature: your feature description"
    ```
    ```bash
    git commit -m "Add feature: your feature description"
    ```

6. **Push to Your Fork:**

    ```bash
    git push origin feature/your-feature-name
    ```
    ```bash
    git push origin feature/your-feature-name
    ```

7. **Create a Pull Request:** Navigate to your fork on GitHub and click the "Compare & pull request" button.

Please ensure that your contributions adhere to the project's coding standards and include relevant tests where applicable.

## License

This project is licensed under the [Creative Commons CC0 license](LICENSE).

## Acknowledgements

- **Perplexity AI:** For providing robust AI capabilities for content enhancement.
- **RAGFlow:** For enabling AI-powered question answering within the application.
- **LogSeq:** For facilitating powerful knowledge management through Markdown files.
- **Three.js & WebXR:** For enabling immersive 3D visualizations and mixed reality experiences.
- **Rust Community:** For creating a performant and reliable language ecosystem.
- **Docker:** For simplifying containerization and deployment processes.

---

## Troubleshooting Configuration Issues

If you encounter an error like:

```
Failed to load settings: missing field `api_key`
Error: Custom { kind: Other, error: "Failed to load settings: missing field `api_key`" }
```

It typically indicates a configuration mismatch. Here's how to resolve it:

1. **Verify `settings.toml` Structure:**

    Ensure that all `api_key` fields are nested under their respective sections. For example:

    ```toml
    [openai]
    api_key = "your_openai_api_key"
    base_url = "https://api.openai.com/v1"
    ```

2. **Check Environment Variables:**

    Ensure that your `.env` file or environment variables correctly override the necessary fields. Use the prefix and separators as defined in your configuration loader. For example:

    ```env
    APP_OPENAI__API_KEY=your_real_openai_api_key
    APP_RAGFLOW__RAGFLOW_API_KEY=your_real_ragflow_api_key
    ```

3. **Review `config.rs`:**

    Ensure that the `Settings` struct accurately reflects the structure of `settings.toml`. Each section should have its corresponding struct.

    ```rust
    #[derive(Debug, Deserialize, Clone)]
    pub struct Settings {
        pub perplexity: PerplexitySettings,
        pub github: GithubSettings,
        pub ragflow: RagFlowSettings,
        pub openai: OpenAISettings,
        pub visualization: VisualizationSettings,
        pub default: DefaultSettings,
    }
    ```

4. **Service Implementations:**

    Ensure that all services access `api_key` fields through their correct paths. For example, in `ragflow_service.rs`:

    ```rust
    pub struct RAGFlowService {
        client: Client,
        api_key: String,
        base_url: String,
        synthesizer: Arc<SonataSpeechSynthesizer>,
    }

    impl RAGFlowService {
        pub fn new(settings: &RagFlowSettings) -> Result<Self, RAGFlowError> {
            Ok(RAGFlowService {
                client: Client::new(),
                api_key: settings.ragflow_api_key.clone(),
                base_url: settings.ragflow_api_base_url.clone(),
                synthesizer: Arc::new(SonataSpeechSynthesizer::new(Path::new(&settings.voice_config_path))?),
            })
        }
    }
    ```

5. **Docker Configuration:**

    Ensure that `settings.toml` and `.env` are correctly included in the Docker image or passed as environment variables at runtime. Avoid hardcoding sensitive information within Dockerfiles.

    ```dockerfile
    # Stage 1: Build the Rust application
    FROM rust:1.70 AS builder

    WORKDIR /usr/src/app

    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        build-essential \
        libssl-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

    # Copy Cargo.toml and Cargo.lock
    COPY Cargo.toml Cargo.lock ./

    # Copy source code
    COPY src ./src

    # Build the application
    RUN cargo build --release

    # Stage 2: Create the final image
    FROM debian:buster-slim

    # Install runtime dependencies
    RUN apt-get update && apt-get install -y \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

    # Copy the compiled binary from builder
    COPY --from=builder /usr/src/app/target/release/webxr-graph /usr/local/bin/webxr-graph

    # Copy configuration files
    COPY settings.toml ./

    # Set environment variables (optional)
    ENV ROCKET_ADDRESS=0.0.0.0
    ENV ROCKET_PORT=8080

    # Expose port
    EXPOSE 8080

    # Set the entrypoint
    ENTRYPOINT ["webxr-graph"]
    ```

    **Note:** Consider using Docker secrets or environment variables for managing sensitive data instead of including them in the image.

6. **Logging and Debugging:**

    Add debugging statements in `config.rs` to verify loaded configurations, ensuring sensitive information is masked.

    ```rust
    impl Settings {
        pub fn new() -> Result<Self, ConfigError> {
            let settings = Config::builder()
                .add_source(File::with_name("settings.toml").required(true))
                .add_source(Environment::with_prefix("APP").separator("__").try_parsing(true))
                .build()?
                .try_deserialize()?;

            // Debugging: Print loaded settings (exclude sensitive fields)
            debug!("Loaded settings: {:?}", settings);

            Ok(settings)
        }
    }

    // Implement Debug for OpenAISettings to mask api_key
    impl fmt::Debug for OpenAISettings {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("OpenAISettings")
                .field("api_key", &"***REDACTED***")
                .field("base_url", &self.base_url)
                .finish()
        }
    }
    ```
