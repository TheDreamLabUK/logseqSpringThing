# Utilities Architecture

## Overview
The utilities layer provides common functionality, helper methods, and shared tools across the application.

## GPU Compute

### Initialization
```rust
pub struct GPUCompute {
    compute_device: Arc<RwLock<Option<ComputeDevice>>>,
    simulation_params: Arc<RwLock<SimulationParams>>,
}

impl GPUCompute {
    pub async fn new(graph_data: &GraphData) -> Result<Self, Error>
    pub async fn test_gpu_at_startup(gpu_compute: Option<Arc<RwLock<GPUCompute>>>)
}
```
- Device detection
- Resource allocation
- Capability testing

### Simulation Parameters
```rust
pub struct SimulationParams {
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion: f32,
    pub damping: f32,
    pub time_step: f32,
    pub phase: SimulationPhase,
    pub mode: SimulationMode,
}
```

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

### Message Types
```rust
pub enum SocketMessage {
    GraphUpdate(GraphData),
    StateUpdate(StateData),
    Error(ErrorData),
}
```
- Binary messages
- Text messages
- Control frames

### Flow Control
```rust
pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error>
```
- Message queuing
- Rate limiting
- Connection management

## Security

### Token Management
```rust
pub fn generate_token() -> String
pub fn validate_token(token: &str) -> Result<Claims, TokenError>
```
- Token generation
- Validation
- Expiration

### Encryption
```rust
pub fn encrypt_data(data: &[u8], key: &[u8]) -> Result<Vec<u8>, CryptoError>
pub fn decrypt_data(encrypted: &[u8], key: &[u8]) -> Result<Vec<u8>, CryptoError>
```
- Data encryption
- Key management
- Secure storage

## Helper Functions

### String Manipulation
```rust
pub fn sanitize_filename(name: &str) -> String
pub fn generate_slug(title: &str) -> String
```
- Text formatting
- Sanitization
- Normalization

### File Operations
```rust
pub async fn ensure_directory(path: &Path) -> Result<(), Error>
pub async fn atomic_write(path: &Path, content: &[u8]) -> Result<(), Error>
```
- Safe writes
- Directory management
- Path handling

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