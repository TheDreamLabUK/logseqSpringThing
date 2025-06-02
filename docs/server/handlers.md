# Request Handlers Architecture

## Overview
The handler layer manages HTTP and WebSocket endpoints, providing API interfaces for client interactions.

## Core Handlers

### API Handler ([`src/handlers/api_handler/mod.rs`](../../src/handlers/api_handler/mod.rs))
This module configures the main REST API routes under the base scope `/api`.

```rust
// In src/handlers/api_handler/mod.rs
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            // Files API routes (e.g., /process, /get_content/{filename})
            .configure(files::config)
            // Graph API routes (e.g., /data, /data/paginated, /update, /refresh)
            .configure(graph::config)
            // Visualisation settings routes (e.g., /settings/{category}, /get_settings/{category})
            .configure(visualisation::config)
            // Nostr authentication routes (e.g., /auth/nostr, /auth/nostr/verify, /auth/nostr/api-keys)
            .service(web::scope("/auth/nostr").configure(crate::handlers::nostr_handler::config))
            // User settings routes (e.g., /user-settings, /user-settings/sync)
            .service(web::scope("/user-settings").configure(crate::handlers::settings_handler::config_public))
            // RAGFlow chat route
            .service(web::scope("/ragflow").configure(crate::handlers::ragflow_handler::config))
            // Health check routes
            .service(web::scope("/health").configure(crate::handlers::health_handler::config))
            // Static pages/assets (if served via /api/pages)
            .service(web::scope("/pages").configure(crate::handlers::pages_handler::config))
            // Potentially other top-level API groups
    );
}
```
-   Organizes REST API endpoints into logical sub-scopes:
    -   `/api/files` (from `src/handlers/api_handler/files/mod.rs`)
    -   `/api/graph` (from `src/handlers/api_handler/graph/mod.rs`)
    -   `/api/visualisation` (from `src/handlers/api_handler/visualisation/mod.rs`)
    -   `/api/auth/nostr` (from `src/handlers/nostr_handler.rs`)
    -   `/api/user-settings` (from `src/handlers/settings_handler.rs`)
    -   `/api/ragflow` (from `src/handlers/ragflow_handler.rs`)
-   Handles request validation, deserialization, calls appropriate services, and serializes responses.

### WebSocket Handler ([`src/handlers/socket_flow_handler.rs`](../../src/handlers/socket_flow_handler.rs))
Manages WebSocket connections for real-time graph data updates.
-   **Path:** `/wss` (or as configured in `main.rs` or `docker-compose.yml` via Nginx proxy)
-   **Function:** `socket_flow_handler(req: HttpRequest, stream: web::Payload, srv: web::Data<Arc<ClientManager>>)` - Note: It typically takes `ClientManager` directly or via `AppState`. The plan mentions `ClientManager` is a static instance, so `AppState` might provide access or it's accessed directly.
-   Handles client connections, disconnections, and messages (`ping`, `requestInitialData`).
-   Uses the static `APP_CLIENT_MANAGER` to register clients and receive graph updates from `GraphService` to broadcast.

### Health Handler ([`src/handlers/health_handler.rs`](../../src/handlers/health_handler.rs))
Provides endpoints for system health monitoring.
-   **Base Path:** `/api/health`
-   **Endpoints:**
    -   `/api/health`: General health check.
    -   `/api/health/physics`: Specific check for the physics simulation status (via `check_physics_simulation` handler).
-   Reports the status of core services and dependencies.

### Pages Handler ([`src/handlers/pages_handler.rs`](../../src/handlers/pages_handler.rs))
Serves static frontend assets and the main `index.html` page.
-   **Base Path:** `/api/pages` (or could be `/` if Nginx proxies static assets differently).
-   Handles routing for client-side application entry points.
-   The exact routes depend on how static file serving is configured in Actix (e.g., using `actix_files`).

### Speech WebSocket Handler ([`src/handlers/speech_socket_handler.rs`](../../src/handlers/speech_socket_handler.rs))
Manages WebSocket connections specifically for speech-related functionalities (STT/TTS).
-   **Path:** `/speech` (or as configured)
-   Interacts with `SpeechService` to process audio streams and broadcast responses.

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