LogseqXR Networking and Data Flow Briefing
This document outlines the networking architecture and data flow for the LogseqXR application, clarifying the roles of REST, WebSockets.

1. Overall Architecture

The application follows a client-server model, with the server responsible for data storage, processing, and settings management, while the client handles visualization and user interaction. Communication occurs through REST API calls for initial setup and settings management, and WebSockets for real-time position updates. 

2. Server-Side (Rust)

Data Storage: Graph data (nodes, edges, metadata) is stored on the server, in a file system. Settings are stored in settings.toml and are updated in real-time in groups of settings only when those settings are updated. Settings can also be overridden by environment variables.

REST API (actix-web): The server exposes a REST API for:

Graph Data: 
- GET /api/graph/data (full graph)
- GET /api/graph/paginated?page={page}&pageSize={pageSize} (paginated graph)
- POST /api/graph/update (update graph data)

Settings: (Temporarily Disabled)
The following settings endpoints are currently disabled as settings are managed locally on the client:
- GET /api/settings (get all settings)
- GET /api/settings/{category} (get all settings for a category)
- GET /api/settings/{category}/{setting} (get individual setting)
- PUT /api/settings/{category}/{setting} (update individual setting)

Note: Settings are currently managed locally in the client's SettingsStore using default values. Server synchronization will be re-enabled in a future update.

Categories include:
- system.network
- system.websocket
- system.security
- system.debug
- visualization.animations
- visualization.ar
- visualization.audio
- visualization.bloom
- visualization.edges
- visualization.hologram
- visualization.labels
- visualization.nodes
- visualization.physics
- visualization.rendering

Other API endpoints: 
- /api/files/fetch: Fetch file contents from the repository
- /api/chat/*: AI chat endpoints for RAGFlow integration
- /api/perplexity: Perplexity AI integration endpoints

WebSocket Handling (actix-web-actors): 
- Binary Protocol (/wss endpoint): 
  - Uses a binary protocol for efficient real-time position and velocity updates
  - Optimized format with 6 floats per node (position + velocity)
  - Supports compression for large graph updates
  - Includes heartbeat mechanism for connection health monitoring

3. Client-Side (TypeScript/Three.js)

The client maintains several key connections:

REST API Communication:
- Initial Graph Data: Retrieving the initial graph data using pagination
- Settings: Loading category settings, getting/updating individual settings
- File Content: Fetching markdown and other file contents as needed
- AI Integration: Communicating with RAGFlow and Perplexity services

WebSocket Connection:
- Establishes WebSocket connection for real-time updates
- Handles binary protocol for position/velocity updates
- Implements reconnection logic with exponential backoff
- Processes compressed data for large graph updates

4. Development and Testing

The application includes several scripts for testing network functionality:
- `scripts/dev.sh`: Main development script with commands for:
  - Starting/stopping containers
  - Testing endpoints
  - Viewing logs
  - Rebuilding services
- `scripts/test-api.sh`: Tests individual API endpoints
- `scripts/test_all_endpoints.sh`: Comprehensive API endpoint testing

5. Security Considerations

- All WebSocket connections use WSS (WebSocket Secure)
- API endpoints require proper authentication headers
- Rate limiting is implemented on sensitive endpoints
- Environment variables are used for sensitive configuration
- CORS is properly configured for development and production

6. Performance Optimizations

Network optimizations include:
- Binary protocol for WebSocket updates
- Compression for large data transfers
- Pagination for initial graph loading
- Efficient settings updates (only changed values)
- Connection pooling for database operations
- Caching of frequently accessed data

7. Error Handling

The system implements robust error handling:
- Automatic WebSocket reconnection
- Graceful degradation on connection loss
- Clear error messages for API failures
- Logging of network-related issues
- Recovery mechanisms for interrupted operations