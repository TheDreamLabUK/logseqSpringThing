# WebSocket Connection and Communication in Logseq XR
## WebSocket Connection Architecture

The application uses a two-component architecture for WebSocket communication:

1. **WebSocketService** (client/websocket/websocketService.ts):
   - Handles the raw WebSocket connection to the server
   - Manages binary protocol encoding/decoding
   - Implements reconnection logic
   - Uses socket-flow library on the server side

2. **GraphDataManager** (client/state/graphData.ts):
   - Manages graph data state
   - Processes node and edge updates
   - Communicates with WebSocketService for binary updates

### Connection Flow

The proper connection sequence must follow this order:

1. Initialize GraphVisualization in index.ts
2. Connect WebSocketService to server with websocketService.connect()
3. Set up binary message handler with websocketService.onBinaryMessage()
4. **Create an adapter** that connects WebSocketService to GraphDataManager
5. Register the adapter with GraphDataManager using setWebSocketService()
6. Enable binary updates in GraphDataManager with graphDataManager.setBinaryUpdatesEnabled(true)

This sequence ensures proper communication between the two components, which use different interfaces.

This sequence ensures that GraphDataManager knows when it can start sending binary updates through WebSocketService.

### Common Issues

#### "WebSocket service not configured" Error

If you see log messages like:
```
[GraphDataManager] Binary updates enabled but WebSocket service not yet configured
[GraphDataManager] WebSocket service still not configured (attempt X/30)
```

This indicates that GraphDataManager is trying to send binary updates before WebSocketService is properly initialized. The fix is to ensure proper initialization sequence:

1. First connect the WebSocketService
2. Then enable binary updates in GraphDataManager

Do not try to directly connect these components through their internal interfaces, as they use different communication patterns. Instead, use their public API methods as shown in the connection flow.

### Component Bridging with Adapter Pattern

GraphDataManager and WebSocketService use different interfaces, which can cause errors if not connected properly. The solution is to use an adapter:

```typescript
// Create an adapter that implements the InternalWebSocketService interface
const webSocketAdapter = {
    send: (data: ArrayBuffer) => {
        websocketService.sendRawBinaryData(data);
    }
};

// Register the adapter with GraphDataManager
graphDataManager.setWebSocketService(webSocketAdapter);
```

This document describes the WebSocket connection and communication process in the Logseq XR project, covering the client-server interaction, data formats, compression, heartbeats, and configuration.

## 1. Connection Establishment

*   **Client-Side Initiation:** The `WebSocketService` in `client/websocket/websocketService.ts` manages the WebSocket connection. The `connect()` method must be called explicitly to initiate the connection.
*   **URL Construction:** The `buildWsUrl()` function in `client/core/api.ts` constructs the WebSocket URL:
    *   **Protocol:** `wss:` for HTTPS, `ws:` for HTTP (determined by `window.location.protocol`).
    *   **Host:** `window.location.hostname`.
    *   **Port:** `:4000` in development (non-production), empty (default port) in production.
    *   **Path:** `/wss`.
    *   **Final URL:** `${protocol}//${host}${port}/wss`
*   **Development Proxy:** In development (using `vite`), the `vite.config.ts` file configures a proxy:
    *   Requests to `/wss` are proxied to `ws://localhost:4000`.
*   **Docker Environment:**
    *   The `Dockerfile` exposes port 4000.
    *   The `docker-compose.yml` file maps container port 4000 to host port 4000.
    *   `nginx` (configured in `nginx.conf`) listens on port 4000 inside the container.
    *   The `scripts/launch-docker.sh` script builds and starts the containers, including a readiness check that uses `websocat` to test the WebSocket connection to `ws://localhost:4000/wss`.
*   **Cloudflared:** When using Cloudflared (`docker-compose.yml`):
    *   Cloudflared forwards traffic to the `webxr` container's `nginx` instance on port 4000, using the container name (`logseq-xr-webxr`) as the hostname.
*   **Server-Side Handling:**
    *   The `socket_flow_handler` function in `src/handlers/socket_flow_handler.rs` handles WebSocket connections (using Actix Web).
    *   It checks for the `Upgrade` header to verify it's a WebSocket request.
    *   It creates a `SocketFlowServer` instance for each connection.
    *   The `ws::start()` function starts the WebSocket actor.
* **Nginx Proxy:** The `nginx.conf` file configures nginx to proxy websocket connections at `/wss` to the rust backend, which is listening on `127.0.0.1:3001` inside the container.

## 2. Message Handling

*   **Client-Side (`client/websocket/websocketService.ts`):**
    *   **`onopen`:** Sends a `requestInitialData` message (JSON) to the server.
    *   **`onmessage`:**
        *   **Text Messages:** Parses JSON messages. Handles `connection_established`, `loading`, and `updatesStarted`.
        *   **Binary Messages:**
            *   Decompresses using `pako.inflate()` (zlib) if necessary.
            *   Decodes the binary data according to the custom protocol (see "Binary Protocol").
            *   Calls the `binaryMessageCallback` with the decoded node data.
    *   **`sendMessage()`:** Sends text messages (JSON).
    *   **`sendNodeUpdates()`:** Sends binary messages for node updates (limited to 2 nodes per update). Compresses if needed.
*   **Server-Side (`src/handlers/socket_flow_handler.rs`):**
    *   **`started`:** Sends a `connection_established` message (JSON) followed by a `loading` message.
    *   **`handle`:**
        *   **`Ping`:** Responds with a `Pong` message (JSON).
        *   **`Text`:**
            *   Parses JSON.
            *   Handles `ping` messages (responds with `pong`).
            *   Handles `requestInitialData`:
                *   Immediately starts a timer to send binary position updates periodically (interval based on settings, default 30 Hz).
                *   Sends an `updatesStarted` message (JSON) to signal that updates have begun.
        *   **`Binary`:**
            *   Decodes using `binary_protocol::decode_node_data()`.
            *   Handles `MessageType::PositionVelocityUpdate` (for up to 2 nodes): Updates node positions and velocities in the graph data.
        *   **`Close`:** Handles client close requests.

## 3. Binary Protocol (`src/utils/binary_protocol.rs`)

*   **`MessageType`:** `PositionVelocityUpdate` (0x01).
*   **`BinaryNodeData`:**
    *   `id`: `u16` (2 bytes)
    *   `position`: `Vec3Data` (12 bytes: x, y, z as `f32` properties)
    *   `velocity`: `Vec3Data` (12 bytes: x, y, z as `f32` properties)
    *   Total: 26 bytes per node.
*   **Encoding:**
    *   Node Data (stream of nodes without header):
        *   Node ID (`u16` Little Endian).
        *   Position (x, y, z as `f32` Little Endian).
        *   Velocity (x, y, z as `f32` Little Endian).
*   **Decoding:**
    *   Validates message size is a multiple of 26 bytes.
    *   Reads node data for each 26-byte segment.
* **Byte Order:** Little Endian.

## 4. Data Alignment and Case Handling

*   **Client (TypeScript):** Uses `camelCase` for variables and interfaces.
*   **Server (Rust):**
    *   `socket_flow_messages::BinaryNodeData`: Uses Vec3Data struct with x,y,z properties for position and velocity
    *   `types::vec3::Vec3Data`: Structured vector representation with x,y,z fields
    *   `models::node::Node`: Uses `camelCase` for fields (due to Serde's `rename_all` attribute)
    * API calls use `burger-case`.
*   **Data Transfer:** The binary protocol ensures data alignment between the client and server. The `BinaryNodeData` struct in `socket_flow_messages.rs` mirrors the structure sent over the WebSocket.

## 5. Compression

*   **Client:** Uses `pako` library for zlib compression/decompression.
    *   Compresses binary messages if they are larger than `COMPRESSION_THRESHOLD` (1024 bytes).
    *   Attempts to decompress incoming binary messages, falling back to original data if decompression fails.
*   **Server:** Uses `flate2` crate for zlib compression/decompression.
    *   `maybe_compress()`: Compresses if enabled in settings and data size exceeds the threshold.
    *   `maybe_decompress()`: Decompresses if enabled in settings.

## 6. Heartbeat

*   **Server:** Expects `ping` messages from the client. `src/utils/socket_flow_constants.rs` defines:
    *   `HEARTBEAT_INTERVAL`: 30 seconds.
    *   `CLIENT_TIMEOUT`: 60 seconds (double the heartbeat interval).
*   **Client:** The client-side code doesn't have explicit heartbeat sending logic, but the server expects pings, and the `docker-compose.yml` healthcheck sends a ping. The `cloudflared` configuration also sets `TUNNEL_WEBSOCKET_HEARTBEAT_INTERVAL` to 30s.
* **Nginx:** `nginx.conf` has timeouts configured:
    * `proxy_read_timeout`: 3600s
    * `proxy_send_timeout`: 3600s
    * `proxy_connect_timeout`: 75s

## 7. Throttling/Update Rate

*   **Server:** Sends position updates at a rate determined by the `binary_update_rate` setting (defaulting to 30 Hz), controlled by a timer in `socket_flow_handler.rs`. The constant `POSITION_UPDATE_RATE` in `socket_flow_constants.rs` is 5 Hz, but the actual update rate is controlled by the settings.
*   **Client:**  The client receives updates as they are sent by the server. There's no explicit throttling on the client side, other than limiting user-initiated updates to 2 nodes per message.
*   **Initial Delay:** To ensure GPU computations have time to run, the server now has a 500ms delay before starting to accept client connections.

## 8. Order of Operations

1.  Client initiates a WebSocket connection to `/wss`.
2.  In development, Vite proxies the connection to `ws://localhost:4000`.
3.  In Docker, the connection goes to port 4000 on the host, which is mapped to port 4000 of the `webxr` container.
4.  Before accepting connections, the server has a brief delay (500ms) to allow the GPU to compute initial node positions.
5.  `nginx` (inside the container) receives the connection on port 4000.
6.  `nginx` proxies the WebSocket connection to the Rust backend on `127.0.0.1:3001`.
7.  The `socket_flow_handler` in the Rust backend handles the connection.
8.  The server sends a `connection_established` message (JSON).
9.  The server sends a `loading` message to signal that the client should display a loading indicator.
10. The client displays a loading indicator and sends a `requestInitialData` message (JSON).
11. The server starts sending binary position updates at the configured interval.
12. The client creates an adapter to connect WebSocketService with GraphDataManager.
13. The client registers the adapter using GraphDataManager.setWebSocketService().
14. The server sends an `updatesStarted` message to signal that updates have begun.
13. The client hides the loading indicator upon receiving the `updatesStarted` message.
14. The client receives and processes the binary data, updating the visualization.
15. The server and client exchange `ping` and `pong` messages for connection health (although the client-side pinging is primarily handled by the `docker-compose` healthcheck and potentially Cloudflared).
16. User interactions on the client can trigger sending binary node updates (limited to 2 nodes) to the server.

## 9. Loading State and User Feedback

*   **Server-Side Loading Message:** After the connection is established, the server sends a `loading` message to indicate data is being prepared.
*   **Client-Side Loading Indicator:** Upon receiving the `loading` message, the client displays a loading indicator (in `VisualizationController.ts`).
*   **Updates Started Signal:** Once the server is ready to send position updates, it sends an `updatesStarted` message.
*   **Loading Complete:** Upon receiving the `updatesStarted` message, the client hides the loading indicator and begins displaying the graph.
*   This provides visual feedback during the initialization process and ensures users don't see poorly-distributed node layouts.

## 10. Recent Protocol Optimizations

*   **Size Reduction:** 
    * Node ID changed from u32 (4 bytes) to u16 (2 bytes)
    * Node data size reduced from 28 to 26 bytes per node (~7% reduction)

*   **Format Simplification:**
    * Removed message headers (no version number, sequence number, timestamp)
    * Binary data is now a simple array of node updates

*   **Type Consistency:**
    * Consistent use of structured Vec3Data/THREE.Vector3 objects throughout the pipeline
    * Helper functions handle GPU compatibility where array formats are needed