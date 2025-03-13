# Service Layer Architecture

## Overview
The service layer provides core business logic and data management capabilities, implementing key features like file handling, graph operations, and external service integration.

## Core Services

### File Service (`src/services/file_service.rs`)
- Handles file system operations
- Manages file metadata
- Implements caching strategies
- Coordinates with GitHub integration

### Graph Service (`src/services/graph_service.rs`)
- Core graph data structure management
- Node and edge operations
- Layout calculations
- Graph state persistence

### GitHub Integration (`src/services/github/`)
```rust
// Key components:
mod api;        // API client implementation
mod config;     // GitHub configuration
mod content;    // Content management
mod pr;         // Pull request handling
mod types;      // Type definitions
```

### Perplexity Service (`src/services/perplexity_service.rs`)
- AI integration for graph analysis
- Query processing
- Response handling

### RAGFlow Service (`src/services/ragflow_service.rs`)
- Retrieval-augmented generation
- Document processing
- Knowledge graph integration

### Nostr Service (`src/services/nostr_service.rs`)
- Authentication management
- User session handling
- Nostr protocol integration

## Service Interaction Patterns

### 1. Dependency Injection
Services are initialized through the `AppState` structure:
```rust
pub struct AppState {
    pub graph_service: GraphService,
    pub github_client: Arc<GitHubClient>,
    // ... other services
}
```

### 2. Concurrency Management
- Services use `Arc<RwLock<T>>` for shared state
- Async operations with Tokio
- Thread-safe data access patterns

### 3. Error Handling
- Consistent error propagation
- Service-specific error types
- Graceful degradation strategies

## Configuration Management
- Environment-based configuration
- Feature flags
- Service-specific settings

## Performance Considerations
- Connection pooling
- Caching strategies
- Rate limiting
- Resource cleanup