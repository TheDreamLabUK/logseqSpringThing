# Utilities Architecture

## Overview
The utilities layer provides common functionality, helper methods, and shared tools across the application.

## GPU Compute

### GPUCompute Structure
The `GPUCompute` struct manages CUDA resources and orchestrates GPU-accelerated graph physics calculations.

```rust
// In src/utils/gpu_compute.rs
// Note: Cuda* types are from the `cudarc` crate.
// BinaryNodeData is defined in `crate::utils::socket_flow_messages`.
// SimulationParams is defined in `crate::models::simulation_params`.
pub struct GPUCompute {
    // pub device: Arc<CudaDevice>, // Handle to the CUDA-capable GPU device
    // pub force_kernel: CudaFunction, // Compiled CUDA kernel for force calculation (e.g., from compute_forces.ptx)
    // pub node_data_gpu: CudaSlice<BinaryNodeData>, // Buffer for node data on the GPU
    // pub edge_data_gpu: Option<CudaSlice<GPUEdgeData>>, // Buffer for edge data on the GPU
    // pub num_nodes: u32,
    // pub num_edges: u32,
    // pub node_id_to_idx: HashMap<String, u32>, // Maps node ID (String) to its index in the GPU buffer
    // pub simulation_params_gpu: CudaSlice<GPUSimulationParams>, // GPU-side simulation parameters
    // pub iteration_count: Arc<AtomicU32>, // Tracks the number of simulation iterations performed
    // pub ptx_path: PathBuf, // Path to the compiled PTX file
}

impl GPUCompute {
    // pub fn new(initial_nodes: &[Node], initial_edges: &[Edge], sim_params: &SimulationParams, ptx_path: PathBuf) -> Result<Self, GpuError>;
    // pub fn update_simulation_params(&mut self, sim_params: &SimulationParams) -> Result<(), GpuError>;
    // pub fn update_node_data(&mut self, nodes: &[Node]) -> Result<(), GpuError>;
    // pub fn run_simulation_step(&mut self) -> Result<(), GpuError>;
    // pub fn get_node_positions(&self) -> Result<Vec<BinaryNodeData>, GpuError>; // Fetches updated positions from GPU
    // pub fn test_gpu_communication() -> Result<(), GpuError>;
}
```
- **Resource Management**: Handles CUDA device, context, stream, module, and kernel loading from a PTX file (e.g., `compute_forces.ptx` compiled from `compute_forces.cu`).
- **Data Transfer**: Manages copying `BinaryNodeData` (for nodes) and potentially `GPUEdgeData` (a GPU-specific edge representation) to and from GPU memory (`CudaSlice`).
- **Kernel Execution**: Launches the CUDA kernel for force calculation with appropriate parameters.
- **State Tracking**: Keeps track of the number of nodes, edges, and simulation iterations.

### Simulation Parameters ([`src/models/simulation_params.rs`](../../src/models/simulation_params.rs))
The `SimulationParams` struct defines the parameters controlling the physics simulation. These are used by both CPU and GPU computation paths. A separate C-compatible version (`GPUSimulationParams`) might be used for direct transfer to CUDA kernels if the main `SimulationParams` struct is not `#[repr(C)]`.

```rust
// In src/models/simulation_params.rs
#[derive(Default, Serialize, Deserialize, Clone, Debug, Copy)] // Ensure Copy if used by GPUSimulationParams directly
#[serde(rename_all = "camelCase")]
pub struct SimulationParams {
    pub iterations: u32,
    pub time_step: f32,
    pub spring_strength: f32,
    pub repulsion_strength: f32, // Note: field name consistency (repulsion vs repulsion_strength)
    pub max_repulsion_distance: f32,
    pub mass_scale: f32,
    pub damping: f32,
    pub boundary_damping: f32,
    // enable_bounds, gravity_strength, center_attraction_strength are typically part of
    // AppFullSettings.visualisation.physics and influence these params.
    // SimulationMode and SimulationPhase enums might also be defined here if used.
}

// Potentially a separate struct for GPU if SimulationParams is not repr(C)
// #[repr(C)]
// #[derive(Default, Clone, Copy, Debug, Pod, Zeroable)] // For bytemuck if used
// pub struct GPUSimulationParams { /* fields matching kernel expectations */ }
```
- These parameters are crucial for tuning the behavior and performance of the graph layout algorithm.
- The actual parameters used by `GPUCompute` might be a subset or a transformed version of those found in `AppFullSettings.visualisation.physics`.

## Logging

### Configuration
```rust
// In src/utils/logging.rs
pub struct LogConfig {
    pub level: String, // e.g., "info", "debug", "warn", "error", "trace"
    pub format: String, // e.g., "json", "text"
    // pub file_path: Option<String>, // Optional: path for file logging
    // pub file_level: Option<String>, // Optional: different level for file
    // pub rotation_size: Option<u64>, // Optional: log file rotation size
    // pub rotation_count: Option<usize>, // Optional: number of rotated files to keep
}

// Function to initialize logging based on LogConfig
// pub fn init_logging_with_config(config: &LogConfig) -> Result<(), Box<dyn std::error::Error>>;
```
- Configures log levels (e.g., from `AppFullSettings.system.debug.log_level`).
- Sets output formatting (e.g., from `AppFullSettings.system.debug.log_format`).
- May handle file rotation if file logging is enabled.
- Uses `tracing` and `tracing_subscriber` crates.

### Usage Patterns
```rust
info!("Starting operation: {}", operation_name);
debug!("Processing data: {:?}", data);
error!("Operation failed: {}", error);
```

## WebSocket Messages

The application uses WebSockets for real-time communication between the server and clients. This includes sending control messages and streaming graph data updates.

### JSON Control Messages ([`src/utils/socket_flow_messages.rs`](../../src/utils/socket_flow_messages.rs))
The `Message` enum defines JSON-based control messages:
```rust
// In src/utils/socket_flow_messages.rs
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "payload", rename_all = "camelCase")]
pub enum Message {
    Ping, // Client sends Ping, Server responds with Pong
    Pong, // Server sends Pong
    EnableRandomization { enabled: bool }, // Client requests to toggle node position randomization (server acknowledges)
    RequestInitialData, // Client requests initial graph data / start updates
    // Server might send:
    // ConnectionEstablished { timestamp: u64 },
    // UpdatesStarted { timestamp: u64 },
    // Loading { message: String },
}
```
- These are for signaling, heartbeating, and simple state changes.

### Binary Data Transmission ([`src/utils/socket_flow_messages.rs`](../../src/utils/socket_flow_messages.rs))
- **`BinaryNodeData`**: As defined in the Graph Types section of `docs/server/types.md`. This struct (`position`, `velocity`, `mass`, `flags`, `padding`) is used server-side for physics.
- For WebSocket transmission to the client, a more compact version is typically used: Node ID (`u16`), Position (3x `f32`), Velocity (3x `f32`) = 26 bytes per node. This is often packed into a larger `Vec<u8>` or `Bytes` object for sending.
- The `socket_flow_handler.rs` and `ClientManager` handle sending these binary updates, which are generated by `GraphService`.

### Flow Control and Handling
- The main WebSocket connection logic is in [`src/handlers/socket_flow_handler.rs`](../../src/handlers/socket_flow_handler.rs).
- `ClientManager` (also often in `socket_flow_handler.rs` or a related module) tracks connected clients and broadcasts messages.

## Security

Security in the application primarily revolves around Nostr-based authentication and session management, facilitated by the `NostrService` and helper functions in `src/utils/auth.rs`.

### Authentication and Authorization Utilities ([`src/utils/auth.rs`](../../src/utils/auth.rs))
This module provides helper functions for verifying access to protected API endpoints.

-   **`verify_access(req: &HttpRequest, nostr_service: &NostrService, required_level: AccessLevel) -> Result<String, HttpResponse>`**:
    -   Extracts `X-Nostr-Pubkey` and `X-Nostr-Token` from request headers.
    -   Calls `NostrService::validate_session` to check token validity and expiry.
    -   Checks if the user meets the `required_level` (`Authenticated` or `PowerUser`).
    -   Returns the validated `pubkey` on success or an appropriate `HttpResponse` error.
-   **`AccessLevel` Enum**: Defines `Authenticated` and `PowerUser` levels.
-   These utilities are used as guards in Actix request handlers.

### Nostr Event Verification & Session Management
- This is primarily handled by [`NostrService`](../../src/services/nostr_service.rs).
- `NostrService::verify_auth_event` validates incoming Nostr authentication events (kind 22242).
- `NostrService` manages session tokens (generation, storage in memory with expiry, validation). User profiles, including API keys, are often stored in `ProtectedSettings` and accessed via `NostrService`.

### Encryption
- **Transport Layer Security**: HTTPS and WSS are used for encrypting data in transit.
- **Data at Rest**: Sensitive data like API keys in `ProtectedSettings` (stored in `protected_settings.json`) rely on file system permissions and environment security. No application-level encryption utilities for general data at rest are typically found in `src/utils/`.
- **Nostr Event Signatures**: Ensure authenticity and integrity of Nostr events. Content encryption depends on client-side NIP implementations.

## Helper Functions

Generic helper functions for tasks like string manipulation or common data transformations might exist but are not typically centralized in a single "kitchen sink" `utils.rs` file. They are often co-located with the modules that use them or organized into more specific utility modules if widely needed.
- File system operations are primarily handled by [`FileService`](../../src/services/file_service.rs).
- Specific data conversion or formatting utilities might be found within individual service or model files.
- The plan's mention of `sanitize_filename`, `generate_slug`, etc., as not being present is accurate for a generic `utils` module; such specific helpers would be within relevant services.

## Error Handling

### Custom Errors
```rust
pub enum UtilError {
    IO(std::io::Error),
    Format(String),
    Validation(String),
}
```
- Error types
- Error conversion
- Error context

### Recovery Strategies
- Retry logic
- Fallback mechanisms
- Error reporting