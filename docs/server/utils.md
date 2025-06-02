# Utilities Architecture

## Overview
The utilities layer provides common functionality, helper methods, and shared tools across the application.

## GPU Compute

### GPUCompute Structure
The `GPUCompute` struct manages CUDA resources and orchestrates GPU-accelerated graph physics calculations.

```rust
// From src/utils/gpu_compute.rs
// Note: Cuda* types are from the `cudarc` crate.
// BinaryNodeData is defined in `crate::utils::socket_flow_messages`.
// SimulationParams is defined in `crate::models::simulation_params`.
pub struct GPUCompute {
    pub device: Arc<CudaDevice>, // Handle to the CUDA-capable GPU device
    pub force_kernel: CudaFunction, // Compiled CUDA kernel for force calculation
    pub node_data: CudaSlice<BinaryNodeData>, // Buffer for node data on the GPU
    pub num_nodes: u32, // Number of nodes currently managed by the GPU
    pub node_indices: HashMap<String, usize>, // Maps node ID (String) to its index in the GPU buffer
    pub simulation_params: SimulationParams, // Current simulation parameters
    pub iteration_count: u32, // Tracks the number of simulation iterations performed
}

impl GPUCompute {
    pub async fn new(graph: &GraphData) -> Result<Arc<RwLock<Self>>, std::io::Error>;
    pub async fn test_gpu() -> Result<(), std::io::Error>;
    // ... other methods like update_graph_data, update_simulation_params, compute_forces, step ...
}
```
- **Resource Management**: Handles CUDA device, context, stream, module, and kernel loading.
- **Data Transfer**: Manages copying `BinaryNodeData` and `GPUEdgeData` to and from GPU memory (`CudaSlice`).
- **Kernel Execution**: Launches the CUDA kernel (`compute_forces_kernel`) with appropriate parameters.
- **State Tracking**: Keeps track of the number of nodes and simulation iterations.

### Simulation Parameters
The `SimulationParams` struct defines the parameters controlling the physics simulation. These are used by both CPU and GPU computation paths. The `GPUSimulationParams` struct is a C-compatible version for direct use in CUDA kernels.

```rust
// From src/models/simulation_params.rs

/// Defines the operational mode of the simulation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationMode {
    Remote,  // GPU-accelerated remote computation (default)
    GPU,     // Local GPU computation (deprecated)
    Local,   // CPU-based computation (disabled)
}
// Default: SimulationMode::Remote

/// Defines the current phase of the simulation, allowing for different parameter sets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationPhase {
    Initial,    // Heavy computation for initial layout
    Dynamic,    // Lighter computation for dynamic updates
    Finalize,   // Final positioning and cleanup
}
// Default: SimulationPhase::Initial

/// Parameters for the physics simulation.
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SimulationParams {
    // Core iteration parameters
    pub iterations: u32,           // e.g., 100
    pub time_step: f32,            // e.g., 0.2
    
    // Force parameters
    pub spring_strength: f32,      // e.g., 0.5
    pub repulsion: f32,            // e.g., 100.0
    pub max_repulsion_distance: f32, // e.g., 500.0
    
    // Mass and damping
    pub mass_scale: f32,           // e.g., 1.0
    pub damping: f32,              // e.g., 0.5
    pub boundary_damping: f32,     // e.g., 0.9
    
    // Boundary control
    pub viewport_bounds: f32,      // e.g., 1000.0
    pub enable_bounds: bool,       // e.g., true
    
    // Simulation state
    pub phase: SimulationPhase,
    pub mode: SimulationMode,
}

/// GPU-compatible version of simulation parameters.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)] // Pod, Zeroable from bytemuck
pub struct GPUSimulationParams {
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion: f32,
    pub damping: f32,
    pub max_repulsion_distance: f32,
    pub viewport_bounds: f32,
    pub mass_scale: f32,
    pub boundary_damping: f32,
}
```
- These parameters are crucial for tuning the behavior and performance of the graph layout algorithm.
- `SimulationPhase` allows for different sets of parameters (e.g., more iterations initially, fewer for dynamic updates).

## Logging

### Configuration
```rust
pub struct LogConfig {
    pub console_level: String,
    pub file_level: String,
}

pub fn init_logging_with_config(config: LogConfig) -> Result<(), Error>
```
- Log levels
- Output formatting
- File rotation

### Usage Patterns
```rust
info!("Starting operation: {}", operation_name);
debug!("Processing data: {:?}", data);
error!("Operation failed: {}", error);
```

## WebSocket Messages

The application uses WebSockets for real-time communication between the server and clients. This includes sending control messages and streaming graph data updates.

### Control Message Types (`src/utils/socket_flow_messages.rs`)
The following `Message` enum defines JSON-based control messages exchanged over WebSocket:

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")] // Uses a "type" field to distinguish message variants
pub enum Message {
    #[serde(rename = "ping")]
    Ping { timestamp: u64 }, // Client or server sends a ping with a timestamp
    
    #[serde(rename = "pong")]
    Pong { timestamp: u64 }, // Response to a ping, echoing the timestamp
    
    #[serde(rename = "enableRandomization")]
    EnableRandomization { enabled: bool }, // Client requests to toggle node position randomization
}
```
- These messages are typically small and used for signaling or simple state changes.

### Graph Data Transmission
- **Node Positions**: Updates to node positions are usually sent via a separate binary WebSocket stream for efficiency. This stream transmits an array of `BinaryNodeData` structures (defined in `crate::utils::socket_flow_messages`).
- **Initial Graph Structure**: The initial graph structure (nodes, edges, metadata) might be sent via a REST API endpoint or a larger initial WebSocket message, rather than through the continuous `Message` enum stream.

### Flow Control and Handling (`src/handlers/socket_flow_handler.rs`)
The main WebSocket connection and message handling logic resides in `socket_flow_handler.rs`. This handler manages:
```rust
// Simplified signature from src/handlers/socket_flow_handler.rs
pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>, // Shared application state
    client_manager: web::Data<ClientManager>, // Manages connected clients
) -> Result<HttpResponse, Error>;
```
- **Client Management**: Tracking connected clients, broadcasting messages.
- **Message Parsing**: Deserializing incoming JSON control messages and handling binary data.
- **State Synchronization**: Coordinating graph state updates with the `GraphService`.
- **Heartbeating**: Using Ping/Pong messages to maintain connection health.

## Security

Security in the application primarily revolves around Nostr-based authentication and session management, facilitated by the `NostrService` and helper functions in `src/utils/auth.rs`.

### Authentication and Authorization (`src/services/nostr_service.rs`, `src/utils/auth.rs`)

1.  **Nostr Event Verification**:
    *   Clients initiate authentication by sending a signed Nostr event (typically kind 22242 for authentication).
    *   The `NostrService::verify_auth_event` method validates this event's signature using the `nostr-sdk`.

2.  **Session Token Management**:
    *   Upon successful Nostr event verification, `NostrService` generates a unique session token (UUID v4) for the user.
    *   This token is stored in memory associated with the user's `pubkey` and has a configurable expiry time (e.g., 1 hour, from `AUTH_TOKEN_EXPIRY` env var).
    *   The `NostrUser` struct stores this `session_token` and `last_seen` timestamp.
    *   Methods like `validate_session`, `refresh_session`, and `logout` manage the lifecycle of these tokens.

3.  **Access Control with HTTP Headers**:
    *   Authenticated API requests are expected to include `X-Nostr-Pubkey` and `X-Nostr-Token` headers.
    *   The `src/utils/auth.rs` module provides helper functions:
        *   `verify_access(req, nostr_service, required_level)`: Core function to check headers against `NostrService::validate_session` and the required `AccessLevel`.
        *   `verify_authenticated(req, nostr_service)`: Ensures a valid, active session.
        *   `verify_power_user(req, nostr_service)`: Ensures a valid session for a user designated as a "power user" (configured via `POWER_USER_PUBKEYS` env var).
    *   `AccessLevel` enum (`Authenticated`, `PowerUser`) defines the required authorization.

```rust
// Example from src/utils/auth.rs
pub enum AccessLevel {
    Authenticated,
    PowerUser,
}

pub async fn verify_access(
    req: &HttpRequest,
    nostr_service: &NostrService,
    required_level: AccessLevel,
) -> Result<String, HttpResponse>; // Returns pubkey on success
```

### Encryption
- **Transport Layer Security**: The application relies on HTTPS and WSS (WebSocket Secure) for encrypting data in transit.
- **Data at Rest**: There are no generic application-level encryption utilities like `encrypt_data` defined in the `utils` module for general data encryption at rest. Sensitive data like API keys within `ProtectedSettings` would be protected by file system permissions and the security of the environment where the `protected_settings.json` file is stored.
- **Nostr Event Signatures**: Nostr events themselves are signed, ensuring authenticity and integrity, but their content is not typically encrypted by default unless specific NIPs (Nostr Implementation Possibilities) for encryption are used by clients.

## Helper Functions

Generic helper functions for string manipulation or file operations are not centralized in a single `utils` module. Specific functionalities are typically implemented within the services or modules that require them. For example:
- File system interactions are primarily handled by `FileService` ([`src/services/file_service.rs`](src/services/file_service.rs)).
- String formatting or specific transformations are usually done ad-hoc where needed.

(The previously listed generic helper functions like `sanitize_filename`, `generate_slug`, `ensure_directory`, and `atomic_write` are not present as general utilities in `src/utils/`.)

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