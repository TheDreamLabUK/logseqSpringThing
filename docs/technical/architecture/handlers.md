# Request Handlers Architecture

## Overview
The handler layer manages HTTP and WebSocket endpoints, providing API interfaces for client interactions.

## Core Handlers

### API Handler
```rust
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            // Route configurations
    );
}
```
- REST API endpoints
- Request validation
- Response formatting

### WebSocket Handler
```rust
pub async fn socket_flow_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error>
```
- Real-time communication
- Graph updates
- Client state management

### Health Handler
```rust
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/health")
            .route("", web::get().to(health_check))
            .route("/ready", web::get().to(readiness_check))
    );
}
```
- System health monitoring
- Readiness checks
- Dependency status

### Pages Handler
```rust
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/pages")
            // Static and dynamic page routes
    );
}
```
- Static content serving
- Dynamic page generation
- Asset management

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