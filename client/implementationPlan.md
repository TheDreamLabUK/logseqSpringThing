# Implementation Plan

You can find our old implmentation in /featureclient but it was over complex and didn't work. We are working through creating files in /client in order to completely replace the project client. You can take a look at /featureclient/structure.md if you get stuck, letting it guide you into the old codebase for clues on our original intent. 

This new implmentation should be best in class "mixed reality first" app structure, leaning heavily into the meta quest 3. I think vue has been causing us problems so we should try to get back to simple ts. The desktop interface is still important. We need a way to change all those visualisation settings which were in the vue panel and still exist in settings.toml, using a save settings button to write back to settings.toml on the server. This will allow us also to clean up the meta quest AR interface, removing any control panel from that side. 

When you are confident that you have written a file into this new empty /client directory structure you can mark it with a tick here in the implmentationplan.md file, and move onto the next one.

High-Level Goals
Mixed Reality First, Desktop Second:
Develop a "mixed reality first" architecture that provides an optimal experience on the Meta Quest 3. Desktop remains fully functional and provides additional UI for managing settings. Both share a unified codebase as much as possible.

Clean, Modern Architecture:
Use TypeScript, Three.js, and other best-practice libraries as needed (e.g., state management, lightweight UI libraries). No Vue.js. A simple TypeScript-based approach to UI and state management will replace the old Vue-based panel.

Unified Settings Management:
Settings are stored on the server in settings.toml. The desktop interface allows users to adjust these settings and save them back to the server. The Quest version reads and applies these same settings. Avoid duplication by using a unified data model and minimal branching logic.

Vector3 Standardization:
Migrate all position/velocity data to a Vector3 (or equivalent [f32; 3] arrays in Rust) representation. Streamline GPU and WebSocket protocols to match this format once and eliminate conversion overhead. All binary data is now handled through Float32Arrays with proper type safety and validation.

Sensible Data Flows and Networking:
Rely on a stable WebSocket channel for real-time updates. Desktop and Quest clients share the same binary and JSON protocols. Nginx, Cloudflare tunnels, and Docker networks remain as currently configured, but we keep the plan free of redundant details.

Architecture Overview
```
client/
  ├─ core/              # Core types, constants, utilities ✓
  ├─ state/             # Centralized state (settings, graph data) ✓
  ├─ rendering/         # Three.js scene, nodes/edges, text rendering ✓
  ├─ xr/                # XR integration (Quest 3 focus, extends scene) ✓
  ├─ platform/          # Platform abstraction (detect Quest vs Desktop) ✓
  ├─ websocket/         # WebSocket service and message handling ✓
  ├─ types/             # TypeScript declarations for Three.js and WebXR ✓
  ├─ ui/                # Minimal UI components (desktop settings panel)
  └─ main.ts            # Application entry point (initializes everything) ✓
```

Key Principles:

Unified Code Paths:
Keep the code paths for desktop and XR similar, differing mainly in input and UI layers. For example, both platforms use the same SceneManager and GraphDataManager, but desktop also spins up a settings panel, while XR relies on in-world interactions or no panel at all.

Minimal Frameworks:
Vanilla TS, Three.js, and potentially a small state or UI library (e.g., Zustand or a lightweight reactive store) to handle global state and reduce complexity.

No Vue.js:
The old Vue-based panel is replaced with a simple HTML/TS UI component for the desktop. This UI retrieves and updates settings.toml via API calls.

Core Components
1. Core (types, constants, utils) ✓
types.ts: ✓
Define Node, Edge, GraphData, VisualizationSettings, and other core interfaces. Make sure these interfaces align with the server's Vector3-based data model.
For positions/velocities, use a simple [number, number, number] tuple or a small Vector3 class to unify all calculations.

constants.ts: ✓
Store shared numeric constants (e.g., scales for position/velocity, default node sizes).

utils.ts: ✓
General-purpose helpers: math functions (e.g., for Vector3 operations), throttlers for update frequency, and simple logging helpers.

2. State Management ✓
state/settings.ts: ✓
A state module that manages VisualizationSettings.

On startup, load settings from the server (JSON endpoint converting settings.toml to JSON).
Provide methods to update local settings and trigger a save operation back to settings.toml.
Make settings accessible to both rendering and XR modules.

state/graphData.ts: ✓
A centralized store for the graph's node/edge data.

Handles incoming WebSocket data (initial graph, incremental updates).
Normalizes data into a Vector3-aligned format.
Exposes a method for the rendering system to apply node positions efficiently.

3. WebSocket Communication ✓
websocketService.ts: ✓
A single WebSocket service that:
Connects upon startup.
Receives initial GraphData and subsequent incremental updates (binary for positions, JSON for metadata).
Dispatches events or uses a callback to graphData.ts to update state directly.
Is protocol-aligned with the Vector3 standardization (24-byte structures for position updates, etc.).

4. Rendering ✓
scene.ts: ✓
Sets up a Three.js scene, camera, lights, and renderer.
Adapts camera setup if XR is active. Maintains a unified scene whether viewed on desktop or Quest 3.
Desktop: OrbitControls enabled.
XR: Controlled via headset/hand-tracking.

nodes.ts: ✓
Manages instanced meshes for nodes and possibly lines for edges.
Apply incoming position data directly to instance matrices.
Update on each tick if data changes.
Keep logic minimal and rely on graphData.ts for raw data access.
Implements efficient batched edge updates.

textRenderer.ts: ✓
Handles text labels using SDF-based text rendering.
Shared between desktop and XR, but the display may differ (e.g., fewer labels in XR mode to reduce clutter).

5. XR Integration ✓
xrSessionManager.ts: ✓
Initializes an XR session on supported devices (Quest 3).
Adapts the camera and rendering loop to XR.
Maintains platform-agnostic logic, just triggers XR mode if available.
Implements spatial awareness features:
- 10x10 grid helper for orientation
- Semi-transparent ground plane for depth perception
- Hit test visualization with ring marker
- Environment-aware lighting with intensity matching
- Type-safe feature detection and handling

xrInteraction.ts: ✓
Handles XR-specific input (hand controllers, gestures).
Shared logic with desktop interaction where possible—both use raycasting to select nodes, but input events differ.
Quest 3 might trigger node highlighting or selection differently, but the underlying data handling remains the same.
Implements efficient hit testing with proper type safety.

6. Platform Abstraction ✓
platformManager.ts: ✓
Detects if running on Quest (WebXR capable) or Desktop.
Hooks into main.ts to decide whether to start an XR session or show a desktop UI panel.
Keeps platform-conditional logic localized here.

7. UI (Desktop Settings Interface)
ui/settingsPanel.ts:
A simple HTML+TS panel for desktop:
Fetches current settings on load.
Allows editing (sliders, text inputs) of VisualizationSettings.
On "Save" click, sends updated settings back to the server, updating settings.toml.
The Quest interface reads the updated settings but does not offer in-headset controls for them.

8. Application Entry ✓
main.ts: ✓
The central entry point that:
Initializes platformManager to detect platform.
Initializes websocketService and waits for initial graph data.
Loads settings from the server and updates SettingsManager.
Creates the SceneManager and NodeManager, applying settings and initial graph data.
If desktop, renders the settingsPanel UI.
If Quest, starts the XR session.
Enters the render loop.

9. Type System ✓
types/three.d.ts: ✓
Comprehensive TypeScript declarations for Three.js.
Proper type definitions for geometries, materials, and transforms.
Support for matrix operations and Vector3 calculations.

types/webxr.d.ts: ✓
Complete WebXR API type declarations.
Type guards for optional WebXR features.
Proper handling of hit testing and light estimation.

Vector3 & Protocol Consolidation
Single Source of Truth:
All position and velocity data are represented as Vector3-like arrays ([number, number, number]) on the client, and [f32; 3] on the server side. The WebSocket binary protocol uses a 24-byte (6 floats) structure for each position/velocity update. On the GPU side, CUDA kernels and WGSL shaders also align with vec3 data formats, reducing conversions and overhead.

Data Flow: ✓

Server → Client: ✓
Binary WebSocket frames contain a packed array of float32 triples for position and velocity.
Client parses these directly into Float32Arrays and updates node instances.
Implements proper validation and type checking.

GPU Integration: ✓
The client's GPU compute (if used) and Three.js transforms rely on the same Vector3 format.
Shaders, if needed, use vec3<f32> with no conversions required.

Result: ✓
A consistent Vector3-oriented pipeline from server logic to client rendering reduces complexity and improves performance.

Settings Management Flow (Partial)
Startup: ✓
Client fetches settings.toml via a server endpoint that returns JSON ✓.
SettingsManager merges these settings with defaults and applies them to the rendering system ✓.

Desktop Editing:
The settingsPanel UI lets the user modify visualization settings (e.g., node size, color schemes).
Clicking "Save" sends a POST request with the updated settings to the server, rewriting settings.toml.

Quest Reading: ✓
On Quest, no separate panel is displayed. The XR experience uses the settings loaded at startup or after a refresh. Changes made on the desktop propagate automatically at next startup or when re-fetching settings.

Unified Handling: ✓
The SettingsManager notifies rendering components about any setting changes, ensuring both desktop and XR views stay consistent.

Implementation & Integration Steps
Phase 1: Skeleton & Basic Connectivity ✓
Set up main.ts ✓, platformManager.ts ✓, websocketService.ts ✓, SettingsManager ✓, and SceneManager stubs ✓.
Connect to WebSocket, fetch initial data and settings, log them ✓.

Phase 2: Rendering & Data Flow ✓
Implement NodeManager and basic Three.js scene ✓.
Apply binary data updates to node positions, confirm rendering works on desktop ✓.
Add efficient batched edge updates ✓.
Add proper type safety for all operations ✓.

Phase 3: Settings Integration (In Progress)
Implement settingsPanel.ts for desktop.
Wire up SettingsManager to scene and nodes ✓, confirm changes apply when user saves.

Phase 4: XR Integration ✓
Implement xrSessionManager.ts ✓ and xrInteraction.ts ✓.
Add spatial awareness features (grid, ground plane, hit testing) ✓.
Add environment-aware lighting ✓.
Add proper type guards and safety ✓.
Test on Quest 3 to ensure scene and controls work similarly to desktop ✓.

Phase 5: Polish & Optimization ✓
Optimize performance (instancing, memory usage) ✓.
Add batched updates and efficient matrix operations ✓.
Add proper type safety throughout the codebase ✓.
Ensure stable Vector3-based pipeline and minimal code duplication ✓.

[Rest of Docker & Network Architecture section remains unchanged]

Docker & Network Architecture
===========================

Container Structure
-----------------
1. Multi-Stage Build:
- Stage 1: Frontend Build (node:20-slim)
  * Builds TypeScript/client code using pnpm
  * Outputs static files to /app/data/public/dist

- Stage 2: Rust Dependencies (nvidia/cuda:12.2.0-devel-ubuntu22.04)
  * Caches Rust dependencies
  * Compiles CUDA kernels to PTX

- Stage 3: Rust Application
  * Builds the main Rust application
  * Handles WebSocket and HTTP endpoints

- Stage 4: Python Dependencies
  * Sets up Python environment for TTS
  * Installs GPU-enabled packages

- Stage 5: Final Runtime (nvidia/cuda:12.2.0-devel-ubuntu22.04)
  * Combines all built artifacts
  * Runs nginx and application server

Network Architecture
------------------
1. External Access Chain:
   Client → Cloudflare → cloudflared tunnel → nginx → Rust backend

2. Docker Network (docker_ragflow):
   * Subnet: 172.19.0.0/16
   * Container Aliases:
     - logseq-xr-webxr
     - webxr-client
     - cloudflared

3. Port Mapping:
   * External: 4000
   * Internal: 3000
   * WebSocket endpoint: /wss

Security & Performance
--------------------
1. Container Security:
   * Read-only filesystem where possible
   * Explicit capability limitations
   * No privilege escalation
   * Regular security audits via launch script

2. Resource Management:
   * CPU: Limited to 16 cores
   * Memory: 64GB limit
   * GPU: Explicit NVIDIA GPU 0 allocation
   * tmpfs for high-speed temporary storage

3. Health Monitoring:
   * 30s health check intervals
   * Automatic container restart
   * JSON log format with rotation
   * Resource usage tracking

Deployment Flow
-------------
1. Launch Process:
   * Environment validation
   * Security checks (pnpm, TypeScript, Rust)
   * Container cleanup
   * Service startup
   * Health verification
   * Cloudflare tunnel establishment

2. Volume Management:
   * /app/data/markdown: Persistent markdown storage
   * /app/data/piper: TTS model storage
   * /app/client: Development mount
   * /tmp: High-speed tmpfs

3. Configuration:
   * settings.toml: Main configuration
   * nginx.conf: Web server setup
   * config.yml: Cloudflare tunnel config
   * .env: Environment variables

This architecture ensures:
- Secure external access via Cloudflare
- Efficient container-to-container communication
- GPU acceleration for compute tasks
- Persistent data storage
- Development flexibility
