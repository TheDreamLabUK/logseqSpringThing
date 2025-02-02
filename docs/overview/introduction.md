# LogseqXR: Immersive WebXR Visualization for Logseq Knowledge Graphs

![image](https://github.com/user-attachments/assets/269a678d-88a5-42de-9d67-d73b64f4e520)

**Inspired by the innovative work of Prof. Rob Aspin:** [https://github.com/trebornipsa](https://github.com/trebornipsa)

![P1080785_1728030359430_0](https://github.com/user-attachments/assets/3ecac4a3-95d7-4c75-a3b2-e93deee565d6)

## Project Overview

LogseqXR revolutionizes the way you interact with your Logseq knowledge base. It's not just a visualization tool; it's a complete platform that transforms your notes into a living, breathing 3D graph, explorable in immersive AR/VR environments. This project leverages the power of **WebXR**, **Perplexity AI**, and **RAGFlow** to create a dynamic and interactive experience, allowing you to literally step into your knowledge graph and gain new insights through AI-powered interactions.

**What does this mean for you?**

Imagine walking through your Logseq notes as if they were a physical space. Each note becomes a tangible node, connected by edges that represent the relationships between your ideas. With LogseqXR, you can:

- **Visualize Complexity:** See the intricate connections within your knowledge base in a way that's impossible with a flat, 2D representation.
- **Interact Intuitively:** Move, manipulate, and explore your notes using natural hand gestures or controllers in AR/VR, or through a traditional mouse and keyboard interface.
- **Gain Deeper Understanding:** Leverage the power of AI to ask questions about your knowledge graph and receive contextually relevant answers, summaries, and insights.
- **Collaborate in Real-Time:** Future versions will allow multiple users to explore and interact with the same knowledge graph simultaneously, fostering collaborative knowledge building.
- **Seamlessly Update:** Changes made within the visualization can be automatically submitted back to your source GitHub repository as pull requests, ensuring your Logseq data stays synchronized.

## Key Features

LogseqXR is packed with features designed to enhance your knowledge exploration experience:

### **1. Immersive WebXR 3D Visualization:**

- **AR/VR Environments:** LogseqXR isn't just about viewing your graph on a screen. It's about stepping into it. With support for WebXR, you can experience your knowledge graph in augmented reality (AR) or virtual reality (VR) using compatible devices like the Oculus Quest.
- **Node Interaction and Manipulation:**
  - **Click, Drag, and Reposition:** Interact with nodes directly. Click to select, drag to move, and reposition nodes to explore different perspectives on your data.
  - **Intuitive Navigation:** Use hand tracking or controllers to navigate through the 3D space, zoom in and out, and focus on specific areas of interest.
- **Dynamic Force-Directed Layout:**
  - **Real-time Recalculation:** The graph layout is not static. It's powered by a force-directed algorithm that constantly recalculates node positions based on their connections and user interactions. This creates a dynamic and organic visualization that adapts to changes in real-time.
  - **Customizable Physics:** Fine-tune the physics parameters (like spring strength, repulsion, and damping) to control the layout behavior and tailor it to your preferences.
- **Custom Shaders for Visual Effects:**
  - **Holographic Displays:** Nodes and edges can be rendered with a futuristic holographic effect, enhancing the immersive experience.
  - **Lighting Effects:** Dynamic lighting adds depth and realism to the visualization, making it easier to perceive the 3D structure of the graph.
- **Fisheye Distortion for Focus + Context Visualization:**
  - **Focus on Details:** Apply a fisheye distortion effect to magnify specific areas of the graph while still maintaining an overview of the surrounding context. This allows you to focus on details without losing sight of the bigger picture.

### **2. Voice Interaction System**

The system provides flexible voice interaction capabilities with two options:

#### Current Implementation:
- **OpenAI Voice Integration:**
  - Real-time voice-to-voice communication via WebSocket API
  - High-quality text-to-speech synthesis
  - Natural language understanding
  - Low-latency streaming responses

#### Planned Local Implementation:
- **GPU-Accelerated Voice Processing:**
  - Kororo for local text-to-speech synthesis
  - Whisper for local speech-to-text processing
  - Full GPU acceleration for real-time performance
  - Complete privacy with all processing done locally
  - Zero dependency on external APIs

Both systems support:
- Real-time voice interaction
- Natural conversation flow
- Multi-language support
- WebSocket streaming for low-latency responses
- Seamless integration with the 3D visualization

### **3. AI-Powered Knowledge Enhancement (In Development)**

> **Note:** The Perplexity AI integration is currently under development. The following features describe the planned functionality that will be available in upcoming releases.

- **Automated Knowledge Base Updates:**
  - Perplexity AI analyzes your Markdown files to identify outdated information and suggest updates
  - Automatically generates pull requests to keep your knowledge base current
  - Maintains the integrity of your note structure while enhancing content

- **Intelligent Content Analysis:**
  - Identifies key topics and concepts across your knowledge base
  - Suggests new connections between related notes
  - Highlights potential gaps in your knowledge graph

- **GitHub Integration:**
  - Seamlessly submits updates through pull requests
  - Maintains full version control of all AI-suggested changes
  - Allows for easy review and approval of updates

- **Content Enhancement:**
  - Expands abbreviated notes with detailed explanations
  - Adds relevant citations and references
  - Updates technical information to reflect current state of the art

### **4. Real-time Updates:**

- **WebSocket-Based Communication:** LogseqXR uses WebSockets for real-time, bi-directional communication between the client and server. This ensures that any changes made to the graph, either locally or remotely, are instantly reflected in the visualization.
- **Optimized Binary Protocol:**
  - **Efficient Data Transfer:** To minimize bandwidth usage and maximize performance, LogseqXR uses a custom binary protocol for transmitting node position and velocity updates.
  - **Fixed-Size Format:** Each node update uses a compact 28-byte format (4-byte header + 24 bytes data), making it extremely efficient to transmit over the network.
  - **Initial Layout Flag:** The 4-byte header includes a flag to indicate whether the message contains the initial layout data or just incremental updates.
- **Automatic Graph Layout Recalculation:** When the graph structure changes (e.g., new nodes or edges are added), the force-directed layout algorithm automatically recalculates the optimal node positions to maintain a clear and organized visualization.
- **Live Preview of Changes:** Any changes made to the underlying Logseq knowledge base are immediately reflected in the 3D visualization, providing a live preview of your evolving knowledge graph.

### **5. GPU Acceleration:**

- **WebGPU Compute Shaders for Layout Calculation:** LogseqXR leverages the power of the GPU (Graphics Processing Unit) to perform complex calculations for the force-directed layout algorithm. WebGPU compute shaders, written in WGSL (WebGPU Shading Language), enable parallel processing of node positions and velocities, resulting in significantly faster layout calculations compared to CPU-based approaches.
- **Efficient Force-Directed Algorithms:** The force-directed layout algorithm is optimized for GPU execution, taking advantage of parallel processing to handle large graphs with thousands of nodes and edges.
- **Fallback to CPU Computation:** If a compatible GPU is not available, LogseqXR gracefully falls back to CPU-based computation, ensuring that the visualization remains functional on a wide range of devices.
- **Custom WGSL Shaders for Visual Effects:** In addition to layout calculations, WebGPU is used to power advanced visual effects like the holographic display and fisheye distortion, enhancing the overall visual appeal and immersiveness of the visualization.

For more detailed information about specific features and components, please refer to the following documentation sections:

- [Technical Architecture](../technical/architecture.md)
- [Development Setup](../development/setup.md)
- [API Documentation](../api/rest.md)
- [Deployment Guide](../deployment/docker.md)