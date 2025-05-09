# State Management

This document details the state management patterns and mechanisms used throughout the client application. The application uses several complementary approaches to state management to handle different types of state.

## State Management Overview

The client application manages several types of state:

1. **Application Settings** - User preferences and application configuration
2. **Graph Data** - Nodes, edges, and metadata for the visualisation
3. **UI State** - Control panel state, selected items, and UI configuration
4. **Rendering State** - Camera position, visibility settings, and rendering options
5. **XR State** - XR session status, controller positions, and interaction state

```mermaid
flowchart TB
    subgraph ApplicationState
        Settings[Settings Store]
        GraphData[Graph Data]
        UIState[UI State]
        RenderState[Rendering State]
        XRState[XR State]
    end
    
    subgraph StateConsumers
        RenderManager
        NodeManager
        EdgeManager
        ControlPanel
        XRManager
    end
    
    Settings --> RenderManager
    Settings --> NodeManager
    Settings --> EdgeManager
    Settings --> ControlPanel
    
    GraphData --> NodeManager
    GraphData --> EdgeManager
    
    UIState --> ControlPanel
    RenderState --> RenderManager
    XRState --> XRManager
```

## Key State Management Components

### Settings Store (`client/state/SettingsStore.ts`)

The Settings Store manages application settings with validation, persistence, and observation.

**Key Features:**
- Schema-based validation of settings
- Persistence to local storage and server
- Observable changes through subscribers
- Default values for all settings

**Implementation Pattern:**
```typescript
class SettingsStore {
  private settings: Settings;
  private observers: Set<Observer>;
  
  updateSetting(path: string, value: any): boolean {
    // Validate against schema
    // Update if valid
    // Notify observers
    // Persist to storage
  }
  
  subscribe(observer: Observer): () => void {
    // Add observer to set
    // Return unsubscribe function
  }
}
```

### Graph Data Manager (`client/state/graphData.ts`)

The Graph Data Manager maintains the state of the graph visualisation data.

**Key Features:**
- Loads and processes graph data from server
- Manages node and edge collections
- Handles real-time position updates via binary protocol
- Provides subscription mechanism for changes

**State Transitions:**
```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Loading: fetchInitialData()
    Loading --> PartiallyLoaded: First page loaded
    PartiallyLoaded --> FullyLoaded: All pages loaded
    FullyLoaded --> LiveUpdates: WebSocket connected
    LiveUpdates --> FullyLoaded: WebSocket disconnected
    LiveUpdates --> LiveUpdates: Position update
    FullyLoaded --> Empty: clear()
    LiveUpdates --> Empty: clear()
```

### Settings Observer (`client/state/SettingsObserver.ts`)

The Settings Observer implements the Observer pattern to propagate settings changes.

**Key Features:**
- Subscribes to Settings Store
- Filters and transforms settings updates
- Distributes settings to appropriate components

**Observer Pattern Implementation:**
```mermaid
sequenceDiagram
    participant SettingsStore
    participant SettingsObserver
    participant Component
    
    Component->>SettingsObserver: subscribe(path, callback)
    SettingsObserver->>SettingsStore: subscribe()
    
    Note over SettingsStore: Setting changes
    
    SettingsStore->>SettingsObserver: notifyUpdate(path, value)
    SettingsObserver->>SettingsObserver: filterRelevantUpdates()
    SettingsObserver->>Component: callback(value)
```

## State Persistence

The application persists state in several ways:

1. **Local Storage** - User preferences and UI state
2. **Server Storage** - User settings synchronized to server
3. **URL Parameters** - Shareable state in URL

### Persistence Flow

```mermaid
flowchart TD
    StateChange[State Change] --> ValidState{Is Valid?}
    ValidState -->|Yes| LocalStorage[Store in Local Storage]
    ValidState -->|Yes| SyncToServer{Sync to Server?}
    ValidState -->|No| LogError[Log Error]
    
    SyncToServer -->|Yes| APICall[POST to API]
    SyncToServer -->|No| Complete[Complete]
    
    APICall --> ServerResponse{Success?}
    ServerResponse -->|Yes| Complete
    ServerResponse -->|No| RetryStrategy[Apply Retry Strategy]
    
    RetryStrategy --> APICall
```

## State Change Propagation

The application uses several mechanisms to propagate state changes:

### Event Emitter (`client/utils/eventEmitter.ts`)

A publish-subscribe system for loose coupling between components.

**Key Features:**
- Named events with typed payloads
- Multiple subscribers per event
- Ability to unsubscribe
- Debugging and tracing capabilities

**Usage Pattern:**
```typescript
// Publisher
eventEmitter.emit('settings:changed', { path: 'visualisation.rendering.quality', value: 'high' });

// Subscriber
const unsubscribe = eventEmitter.on('settings:changed', (data) => {
  // Handle event
});

// Later
unsubscribe();
```

### Direct Subscriptions

Components can subscribe directly to state stores.

**Example:**
```typescript
// Subscribe to graph data changes
const unsubscribe = graphDataManager.subscribe((data) => {
  // Update component based on new data
});

// Subscribe to settings changes
const unsubscribe = settingsStore.subscribe((settings) => {
  // Update component based on new settings
});
```

## Settings Structure

The settings are organized hierarchically by domain:

```typescript
interface Settings {
  visualisation: {
    nodes: {
      quality: 'low' | 'medium' | 'high';
      enableInstancing: boolean;
      enableHologram: boolean;
      enableMetadataShape: boolean;
      sizeRange: [number, number];
      baseColor: string;
      opacity: number;
      // ...
    };
    edges: {
      color: string;
      opacity: number;
      arrowSize: number;
      baseWidth: number;
      enableArrows: boolean;
      widthRange: [number, number];
      quality: 'low' | 'medium' | 'high';
      // ...
    };
    physics: {
      enabled: boolean;
      attractionStrength: number;
      repulsionStrength: number;
      springStrength: number;
      damping: number;
      // ...
    };
    rendering: {
      ambientLightIntensity: number;
      directionalLightIntensity: number;
      environmentIntensity: number;
      backgroundColor: string;
      enableAmbientOcclusion: boolean;
      enableAntialiasing: boolean;
      enableShadows: boolean;
      // ...
    };
    animations: {
      enableNodeAnimations: boolean;
      enableMotionBlur: boolean;
      motionBlurStrength: number;
      // ...
    };
    labels: {
      enableLabels: boolean;
      desktopFontSize: number;
      textColor: string;
      textOutlineColor: string;
      // ...
    };
    bloom: {
      enabled: boolean;
      strength: number;
      radius: number;
      threshold: number;
      // ...
    };
    hologram: {
      ringCount: number;
      sphereSizes: number[];
      ringRotationSpeed: number;
      globalRotationSpeed: number;
      // ...
    };
  };
  system: {
    websocket: {
      reconnectAttempts: number;
      reconnectDelay: number;
      binaryChunkSize: number;
      compressionEnabled: boolean;
      // ...
    };
    debug: {
      enabled: boolean;
      enableDataDebug: boolean;
      enableWebsocketDebug: boolean;
      logBinaryHeaders: boolean;
      // ...
    };
  };
  xr: {
    mode: 'immersive-ar';
    roomScale: number;
    spaceType: 'local-floor';
    quality: 'low' | 'medium' | 'high';
    enableHandTracking: boolean;
    handMeshEnabled: boolean;
    handMeshColor: string;
    handMeshOpacity: number;
    // ...
  };
}
```

### Settings Validation

Settings are validated against schemas that define:
- Type constraints
- Range constraints
- Required properties
- Default values

Invalid settings are rejected with error messages.

## State Immutability

The application uses immutability patterns to prevent unexpected state changes:

1. Creating new objects or arrays when updating state
2. Using spread operators for shallow copies
3. Using deep copy functions for nested structures
4. Using getters without setters for read-only access

**Example of Immutable Update:**
```typescript
// Instead of modifying state directly
this.state.user.name = "New Name"; // BAD

// Create a new state object
this.state = {
  ...this.state,
  user: {
    ...this.state.user,
    name: "New Name"
  }
}; // GOOD
```

## Related Documentation

- [Components](components.md) - Component relationships and dependencies
- [Architecture](architecture.md) - Overall system architecture
- [WebSocket Communication](websocket.md) - Real-time state updates