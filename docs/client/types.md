# Client TypeScript Types and Interfaces

This document summarizes key TypeScript types and interfaces used throughout the LogseqXR client application. It focuses on data structures related to graph management, settings, and core application entities.

## Core Graph Data Structures

These types define the fundamental elements of the knowledge graph.

### `Node`

Represents a node in the graph, typically corresponding to a file or a concept.

```typescript
// As used by client/src/features/graph/managers/graphDataManager.ts
// and derived from server-side models/node.rs::Node
export interface Node {
  id: string; // Usually a numeric string, but can be any unique identifier
  metadata_id: string; // Often the filename or a persistent ID for metadata lookup
  label: string; // The display name of the node
  data: BinaryNodeData; // Contains dynamic physics data like position, velocity
  metadata: Record<string, any>; // Arbitrary metadata associated with the node
  file_size?: number; // Optional: size of the associated file
  // ... other properties like color, shape, etc., can be added based on settings
}

// From client/src/types/binaryProtocol.ts (simplified)
// Matches utils/socket_flow_messages.rs::BinaryNodeData on the server
export interface BinaryNodeData {
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  mass: number;
  flags: number; // For bitmasking various states
  // ... other physics-related properties
}
```

### `Edge`

Represents a link or relationship between two nodes.

```typescript
// As used by client/src/features/graph/managers/graphDataManager.ts
// and derived from server-side models/edge.rs::Edge
export interface Edge {
  source: string; // ID of the source node
  target: string; // ID of the target node
  weight?: number; // Strength or importance of the link
  edge_type?: string; // Type of relationship (e.g., "hyperlink", "reference")
  metadata?: Record<string, any>; // Arbitrary metadata for the edge
  // ... other visual properties like color, width
}
```

### `GraphData`

The primary container for all nodes and edges that constitute the graph.

```typescript
// As used by client/src/features/graph/managers/graphDataManager.ts
// and derived from server-side models/graph.rs::GraphData
export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>; // Global metadata about the graph
  // id_to_metadata is a server-side concept, client usually processes nodes directly
}
```

## Settings Interfaces

These interfaces define the structure of the application settings, managed by `SettingsStore`.

### Main `Settings` Interface

The root interface for all application settings, located in `client/src/features/settings/config/settings.ts`.

```typescript
// From client/src/features/settings/config/settings.ts (Key categories shown)
export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings;
  auth: AuthSettings;
  ragflow?: RAGFlowSettings;
  perplexity?: PerplexitySettings;
  openai?: OpenAISettings;
  kokoro?: KokoroSettings;
  whisper?: WhisperSettings; // Added based on plan
}

// Example sub-interfaces (refer to settings.ts for full definitions)
export interface VisualisationSettings {
  nodes: NodeSettings;
  edges: EdgeSettings;
  physics: PhysicsSettings;
  rendering: RenderingSettings;
  // ... and more (animations, labels, bloom, hologram, camera)
}

export interface NodeSettings {
  nodeSize: number;
  baseColor: string;
  opacity: number;
  // ...
}

export interface SystemSettings {
  websocket: WebSocketSettings;
  debug: DebugSettings;
  persistSettings: boolean;
  customBackendUrl?: string;
}

export interface XRSettings {
  enabled: boolean;
  clientSideEnableXR?: boolean;
  mode?: 'inline' | 'immersive-vr' | 'immersive-ar';
  // ...
}
```
For the complete and detailed structure of all settings interfaces (`NodeSettings`, `EdgeSettings`, `PhysicsSettings`, `RenderingSettings`, `AnimationSettings`, `LabelSettings`, `BloomSettings`, `HologramSettings`, `CameraSettings`, `WebSocketSettings`, `DebugSettings`, `AuthSettings`, `RAGFlowSettings`, `PerplexitySettings`, `OpenAISettings`, `KokoroSettings`, `WhisperSettings`), please refer directly to [`client/src/features/settings/config/settings.ts`](../../client/src/features/settings/config/settings.ts).

## RAGFlow Specific Types

Types related to interactions with the RAGFlow API.

### `RagflowChatRequestPayload`

Defines the payload for sending a chat request to the RAGFlow service.

```typescript
// From client/src/types/ragflowTypes.ts
// Matches server-side models/ragflow_chat.rs::RagflowChatRequest
export interface RagflowChatRequestPayload {
  question: string;
  sessionId?: string; // Optional: for continuing a conversation
  stream?: boolean;   // Optional: to stream the response
}
```

### `RagflowChatResponse`

Defines the expected structure of a response from the RAGFlow chat service.

```typescript
// From client/src/types/ragflowTypes.ts
// Matches server-side models/ragflow_chat.rs::RagflowChatResponse
export interface RagflowChatResponse {
  answer: string;
  sessionId: string; // ID for the current conversation session
  // ... other potential fields like sources, intermediate steps
}
```

## Other Notable Types

### `BinaryProtocol` related types (`client/src/types/binaryProtocol.ts`)

This file contains types and constants related to the custom binary protocol used for WebSocket communication, especially for efficient transmission of node position and velocity updates. This includes definitions for message headers, data structures like `BinaryNodeData` (shown above), and potentially enums for message types if used in the binary format.

### Feature-Specific Types (`client/src/features/**/types/`)

Many features have their own dedicated `types.ts` or `*.types.ts` files. For example:
-   [`client/src/features/settings/types/settingsTypes.ts`](../../client/src/features/settings/types/settingsTypes.ts): Contains types related to the UI definition and configuration of settings.
-   [`client/src/features/xr/types/xr.ts`](../../client/src/features/xr/types/xr.ts): Defines types specific to WebXR interactions, controller states, and XR session management.
-   [`client/src/features/visualisation/types/visualisationTypes.ts`](../../client/src/features/visualisation/types/visualisationTypes.ts): May contain types for specific visual elements or interaction modes within the 3D scene.

These feature-specific type files help in modularizing the codebase and ensuring type safety within their respective domains.

This document provides a high-level summary. For the most accurate and detailed definitions, always refer to the source TypeScript files linked.