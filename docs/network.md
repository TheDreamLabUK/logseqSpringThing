LogseqXR Networking and Data Flow Briefing
This document outlines the networking architecture and data flow for the LogseqXR application, clarifying the roles of REST, WebSockets, and RAGFlow integration.

1. Overall Architecture

The application follows a client-server model, with the server responsible for data storage, processing, and settings management, while the client handles visualization and user interaction. Communication occurs through REST API calls for initial setup and settings management, and WebSockets for real-time position updates. 

2. Server-Side (Rust)

Data Storage: Graph data (nodes, edges, metadata) is stored on the server, in a file system. Settings are stored in settings.toml and are updated in real-time in groups of settings only when those settings are updated.

REST API (actix-web): The server exposes a REST API for:

Graph Data: /api/graph/data (full graph) and /api/graph/data/paginated (paginated graph).

Settings: 
- GET /api/visualization/settings/{category} (get all settings for a category)
- GET /api/visualization/settings/{category}/{setting} (get individual setting)
- PUT /api/visualization/settings/{category}/{setting} (update individual setting)

Other API endpoints: /api/files/fetch, /api/chat/*, /api/perplexity.

WebSocket Handling (actix-web-actors): 
- Binary Protocol (/wss endpoint): 
  - Uses a binary protocol for efficient real-time position and velocity updates
  - Optimized format with 6 floats per node (position + velocity)
- WebSocket Control API (/api/visualization/settings/):
  - REST-based control plane for WebSocket configuration
  - Manages settings, heartbeat intervals
  - Allows runtime updates to WebSocket behavior without connection disruption
  - Separates control logic from high-frequency data updates
- Connection Management:
  - Message queuing with configurable queue size
  - Configurable update rate (framerate)
  - Robust reconnection logic with configurable attempts and delays
  - Connection status tracking and notifications
- Heartbeat:
  - Configurable ping/pong intervals
  - Timestamp-based health monitoring
  - Automatic reconnection on timeout
- Error Handling:
  - Comprehensive error types and status codes
  - Detailed error reporting and logging
  - Graceful failure recovery

RAGFlow Integration:
- Network Integration: Joins the RAGFlow Docker network (docker_ragflow)
- Service Discovery: Uses Docker network aliases for service communication
- Optional Connectivity: Gracefully handles RAGFlow availability
- Health Checks: Monitors RAGFlow service health without direct dependencies

Security:
- Handled by cloudflared tunnel and docker

Port Configuration:
- Nginx Frontend: Listens on port 4000 for external connections
- Rust Backend: Runs on port 3001 internally (configurable via PORT env var)
- Nginx Proxy Configuration:
  - WebSocket Binary Protocol (/wss):
    - Disabled buffering and caching for real-time communication
    - Extended timeouts: 3600s read/send, 75s connect
    - Proper connection upgrade handling
    - Optimized for binary data streaming
  - WebSocket Control API (/api/visualization/settings):
    - Standard API timeouts: 60s read/send/connect
    - Enabled buffering for REST responses
    - Handles WebSocket configuration updates
  - API Endpoints (/api):
    - Enabled buffering with 128k buffer size
    - 60s timeouts for read/send/connect
    - Enhanced proxy buffers (4 x 256k)
  - Graph Endpoints (/graph):
    - 30s connect timeout matching heartbeat interval
    - No-store cache control
- Health Checks: 
  - Regular HTTP and WebSocket endpoint monitoring
  - 10-second interval checks with 5-second timeout
  - 5 retries with 10-second start period

3. Client-Side (TypeScript)

Initialization:
- The client loads initial graph data from /api/graph/data/paginated using pagination
- The client loads all visualization settings from /api/visualization/settings/{category}
- WebSocket initialization follows a two-step process:
  1. Control Setup (/api/visualization/settings/websocket):
     - Load WebSocket configuration settings
     - Set up error handling and reconnection policies
  2. Binary Connection (/wss):
     - Establish WebSocket connection for real-time updates
     - Use binary protocol for position/velocity data
     - Handle heartbeat and connection lifecycle

REST API Interaction:
- Initial Graph Data: Retrieving the initial graph data using pagination
- Settings: Loading category settings, getting/updating individual settings

WebSocket Connection and it's REST management system: 
- Establishes compressed WebSocket connection for real-time updates
- Implements reconnection logic with configurable attempts (default: 3)
- Configurable settings for:
  - Heartbeat interval (default: 15s)
  - Heartbeat timeout (default: 60s)
  - Reconnect delay (default: 5s)
- Message queuing with size limits
- Binary message handling with version verification
- Comprehensive error handling and status notifications

4. Docker Networking

The application uses Docker networking for service communication:

RAGFlow Integration:
```yaml
networks:
  ragflow:
    external: true
    name: docker_ragflow  # RAGFlow's network from docker network ls
```

Service Configuration:
```yaml
services:
  webxr:
    networks:
      ragflow:
        aliases:
          - logseq-xr-webxr
          - webxr-client
    deploy:
      resources:
        limits:
          cpus: '16.0'
          memory: 64G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```

Cloudflare Tunnel:
The application uses Cloudflare's tunnel service for secure external access:
- Runs as a separate container (cloudflared-tunnel)
- Environment Configuration:
  - TUNNEL_METRICS: Exposed on 0.0.0.0:2000
  - TUNNEL_DNS_UPSTREAM: Uses 1.1.1.1 and 1.0.0.1
  - TUNNEL_TRANSPORT_PROTOCOL: Uses HTTP/2
  - TUNNEL_WEBSOCKET_ENABLE: Enabled for WebSocket support
  - TUNNEL_WEBSOCKET_HEARTBEAT_INTERVAL: 30s
  - TUNNEL_WEBSOCKET_TIMEOUT: 3600s
  - TUNNEL_RETRIES: 5 attempts
  - TUNNEL_GRACE_PERIOD: 30s
- Provides secure tunneling without exposing ports directly
- Configuration managed through config.yml with ingress rules

Health Check System:
- Container Health: Docker healthcheck monitors service availability
- Backend Health: Rust service monitors internal state and dependencies
- Frontend Health: Nginx monitors backend connectivity
- RAGFlow Health: Periodic checks for RAGFlow service availability
- Metrics: Health status exposed through container metrics

Clear Protocol Definition:
Binary format details (24 bytes per node)
Exact message types (binary updates, ping/pong)
Simplified Configuration:
Clear separation between REST and WebSocket responsibilities
Performance Focus:
Direct binary transmission
No JSON overhead
Efficient TypedArray usage
Clear Client Flow:
Step-by-step initialization process
Explicit data flow patterns
Error handling and performance considerations