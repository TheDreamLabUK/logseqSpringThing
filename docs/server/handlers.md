# Request Handlers Architecture

## Overview
The handler layer manages HTTP and WebSocket endpoints, providing API interfaces for client interactions.

## Core Handlers

### API Handler
```rust
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .service(web::resource("/graph").route(web::get().to(graph_handler)))
            .service(web::resource("/files/upload").route(web::post().to(file_upload_handler)))
            .service(web::resource("/user-settings").route(web::get().to(get_user_settings_handler)))
            .service(web::resource("/visualisation/settings/{category}").route(web::get().to(get_visualisation_settings_handler)))
            .service(web::resource("/ragflow/chat").route(web::post().to(ragflow_chat_handler)))
            // ... other API routes
    );
}
```
- REST API endpoints for graph data, file uploads, user settings, visualization settings, and AI services.
- Request validation and deserialization.
- Response serialization and formatting.

### WebSocket Handler
```rust
pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error>
```
- Handles WebSocket connections for real-time communication.
- Manages graph data updates and broadcasts them to connected clients.
- Processes client-sent messages, including position updates, control commands, and audio streams for STT.

### Health Handler
```rust
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/health") // Corrected path
            .route("", web::get().to(health_check))
            .route("/ready", web::get().to(readiness_check))
    );
}
```
- Provides endpoints for system health monitoring (`/api/health`) and readiness checks (`/api/health/ready`).
- Reports the status of core services and dependencies.

### Pages Handler
```rust
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/pages") // Corrected path
            .route("/index.html", web::get().to(index_html_handler)) // Example, actual routes may vary
            // ... other static page routes
    );
}
```
- Serves static frontend assets and HTML pages, typically mounted under `/api/pages`.
- Manages routing for client-side application entry points if applicable through this path.

## Middleware Integration

### CORS Configuration
```rust
let cors = Cors::default()
    .allow_any_origin()
    .allow_any_method()
    .allow_any_header()
    .max_age(3600)
    .supports_credentials();
```

### Compression
```rust
.wrap(middleware::Compress::default())
```

### Logging
```rust
.wrap(middleware::Logger::default())
```

## Error Handling

### Request Validation
- Input sanitization
- Parameter validation
- Type checking

### Response Formatting
- Error standardization
- Status codes
- Error messages

## Security

### Authentication
- Token validation
- Session management
- Access control

### Authorization
- Role-based access
- Permission checking
- Resource protection