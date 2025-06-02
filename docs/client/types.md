# Client TypeScript Types and Interfaces

This document summarizes key TypeScript types and interfaces used throughout the LogseqXR client application. It focuses on data structures related to graph management, settings, and core application entities.

## Core Graph Data Structures

These types define the fundamental elements of the knowledge graph.

### `Node`

Represents a node in the graph, typically corresponding to a file or a concept.

```typescript
// Primarily defined and used within client/src/features/graph/managers/graphDataManager.ts
export interface Node {
  id: string; // Unique identifier for the node (often numeric, but treated as string)
  label: string; // Display name of the node
  position: { x: number; y: number; z: number }; // Current 3D position
  metadata: Record<string, any>; // Arbitrary metadata (e.g., file type, size, custom tags)
  // Optional properties that might be populated:
  // velocity?: { x: number; y: number; z: number }; // Current velocity (if tracked client-side)
  // mass?: number;
  // fixed?: boolean; // If the node position is fixed
}
```
Note: The `data: BinaryNodeData` field mentioned in previous docs, which mirrored the server-side `BinaryNodeData` (with mass, flags etc.), is not directly part of the primary client-side `Node` interface in `graphDataManager.ts`. The client-side `Node` directly holds `position`. The separate `BinaryNodeData` type below is specifically for WebSocket communication.

```typescript
// From client/src/types/binaryProtocol.ts
// This structure is for WebSocket binary messages.
export interface BinaryNodeData {
  nodeId: number; // Typically the numeric part of the Node's 'id' string
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
}
```

### `Edge`

Represents a link or relationship between two nodes.

```typescript
// Primarily defined and used within client/src/features/graph/managers/graphDataManager.ts
export interface Edge {
  id: string; // Unique identifier for the edge (e.g., "sourceId_targetId")
  source: string; // ID of the source node
  target: string; // ID of the target node
  label?: string; // Optional display label for the edge
  weight?: number; // Strength or importance of the link
  metadata?: Record<string, any>; // Arbitrary metadata for the edge
  // Visual properties might be dynamically applied rather than stored directly on the edge object.
}
```

### `GraphData`

The primary container for all nodes and edges that constitute the graph.

```typescript
// Primarily defined and used within client/src/features/graph/managers/graphDataManager.ts
export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata?: Record<string, any>; // Global metadata about the graph (e.g., graph title, version)
}
```

## Settings Interfaces

These interfaces define the structure of the application settings, managed by `SettingsStore`.

### Main `Settings` Interface

The root interface for all application settings, located in `client/src/features/settings/config/settings.ts`.

```typescript
// The primary Settings interface is defined in:
// client/src/features/settings/config/settings.ts
//
// export interface Settings {
//   visualisation: VisualisationSettings;
//   system: SystemSettings;
//   xr: XRSettings;
//   auth: AuthSettings;
//   ragflow?: RAGFlowSettings;
//   perplexity?: PerplexitySettings;
//   openai?: OpenAISettings;
//   kokoro?: KokoroSettings;
//   // whisper is NOT in the current settings.ts
// }

// Example of a few key sub-categories and fields.
// For the full, accurate structure, ALWAYS refer to settings.ts.
export interface Settings {
  visualisation: {
    nodes: {
      nodeSize: number; // Single number, not a range
      baseColor: string;
      // ... many other node properties
    };
    edges: {
      baseWidth: number;
      color: string;
      // ... many other edge properties
    };
    // ... other categories like labels, physics, rendering, hologram, camera (optional)
  };
  system: {
    websocket: { // ClientWebSocketSettings
      updateRate: number; // Example field
      // ... other websocket settings
    };
    debug: { // DebugSettings
      enabled: boolean;
      // ... other debug settings
    };
    persistSettings: boolean;
    customBackendUrl?: string;
  };
  xr: { // XRSettings
    enabled: boolean;
    clientSideEnableXR?: boolean; // Important client-side toggle
    enableHandTracking: boolean; // Note: not xr.handTracking
    // ... many other XR settings
  };
  auth: { // AuthSettings
    // ... auth related settings
  };
  ragflow?: { /* RAGFlowSettings */ }; // Optional
  perplexity?: { /* PerplexitySettings */ }; // Optional
  openai?: { /* OpenAISettings */ }; // Optional
  kokoro?: { /* KokoroSettings */ }; // Optional
}
```
**Important:** The above is a simplified representation. The definitive source for the `Settings` interface and all its nested types is [`client/src/features/settings/config/settings.ts`](../../client/src/features/settings/config/settings.ts). Please refer to this file for the complete and accurate structure.

## RAGFlow Specific Types

Types related to interactions with the RAGFlow API, defined in [`client/src/types/ragflowTypes.ts`](../../client/src/types/ragflowTypes.ts).

### `RagflowChatRequestPayload`

Defines the payload for sending a chat request to the RAGFlow service.
```typescript
// From client/src/types/ragflowTypes.ts
// This should align with the server's RagflowChatRequest model.
export interface RagflowChatRequestPayload {
  question: string;
  sessionId?: string; // Optional: for continuing a conversation
  stream?: boolean;   // Optional: to stream the response (default: false)
}
```

### `RagflowChatResponsePayload` (or similar name in `ragflowTypes.ts`)

Defines the expected structure of a response from the RAGFlow chat service.
```typescript
// From client/src/types/ragflowTypes.ts
// This should align with the server's RagflowChatResponse model.
export interface RagflowChatResponsePayload { // Name might vary slightly in the file
  answer: string;
  sessionId: string; // ID for the current conversation session
  // Note: The documentation mentioned 'conversation_id' but the plan specifies 'sessionId'.
  // Verify against ragflowTypes.ts and server model src/models/ragflow_chat.rs.
}
```

## Other Notable Types

### `BinaryProtocol` related types ([`client/src/types/binaryProtocol.ts`](../../client/src/types/binaryProtocol.ts))

This file contains types and constants related to the custom binary protocol used for WebSocket communication.
-   `BinaryNodeData`: As defined earlier (nodeId, position, velocity).
-   May include constants for message type identifiers if the binary protocol supports different message types, or for byte offsets/sizes.

### Feature-Specific Types

Many features have their own dedicated `types.ts` or `*.types.ts` files.

-   **Settings UI Types**:
    -   [`client/src/features/settings/types/settingsTypes.ts`](../../client/src/features/settings/types/settingsTypes.ts): This file appears to contain older or more generic UI-related type definitions like `SettingControlProps`, `SettingsSectionProps`.
    -   The primary types driving the current settings UI are `UISettingDefinition` and related types from [`client/src/features/settings/config/settingsUIDefinition.ts`](../../client/src/features/settings/config/settingsUIDefinition.ts). This documentation should clarify that `settingsUIDefinition.ts` is more central for the *current* settings panel structure.

-   **XR Types**:
    -   [`client/src/features/xr/types/xr.ts`](../../client/src/features/xr/types/xr.ts): Defines types specific to WebXR interactions, controller states, hand tracking data, and XR session management.

-   **Visualisation Types**:
    -   [`client/src/features/visualisation/types/visualisationTypes.ts`](../../client/src/features/visualisation/types/visualisationTypes.ts): This file defines a very simple `VisualisationSettings` interface. It's important to note that the comprehensive and detailed visualisation settings are part of the main `Settings` interface in [`client/src/features/settings/config/settings.ts`](../../client/src/features/settings/config/settings.ts) under the `visualisation` key. This `visualisationTypes.ts` might be for a different, more specific concept or could be outdated/less relevant for global settings.

These feature-specific type files help in modularizing the codebase and ensuring type safety within their respective domains.

This document provides a high-level summary. For the most accurate and detailed definitions, always refer to the source TypeScript files linked.