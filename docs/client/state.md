# State Management

This document details the state management patterns and mechanisms used throughout the client application. The application uses several complementary approaches to state management to handle different types of state.

## State Management Overview

The client application manages several types of state:

1.  **Application Settings** - User preferences and application configuration
2.  **Graph Data** - Nodes, edges, and metadata for the visualisation
3.  **UI State** - Control panel state, selected items, and UI configuration
4.  **Rendering State** - Camera position, visibility settings, and rendering options
5.  **XR State** - XR session status, controller positions, and interaction state

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
        GraphManager[Graph Manager]
        ControlPanel[Control Panel]
        XRComponents[XR Components]
        VisualisationComponents[Visualisation Components]
    end
    
    Settings --> VisualisationComponents
    Settings --> ControlPanel
    Settings --> XRComponents
    
    GraphData --> GraphManager
    GraphData --> VisualisationComponents
    
    UIState --> ControlPanel
    RenderState --> VisualisationComponents
    XRState --> XRComponents
```

## Key State Management Components

### Settings Store (`client/src/store/settingsStore.ts`)

The Settings Store manages application settings with validation, persistence, and observation.

**Key Features:**
- Schema-based validation of settings
- Persistence to local storage and server
- Observable changes through Zustand's subscription mechanism
- Default values for all settings

**Implementation Pattern (Zustand):**
The `settingsStore` is a Zustand store, which provides a simplified API for state management.
```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { Settings } from '../features/settings/config/settings';
import { defaultSettings } from '../features/settings/config/defaultSettings';
import { settingsConfig } from '../features/settings/config/settingsConfig';
import { validateSettings } from '../features/settings/utils/settingsValidation';
import { deepMerge } from '../../utils/deepMerge';
import { logger } from '../../utils/logger';

interface SettingsState {
  settings: Settings;
  setSetting: (path: string, value: any) => void;
  // ... other actions
}

export const useSettingsStore = create<SettingsState>()(
  immer(
    persist(
      (set, get) => ({
        settings: defaultSettings,
        setSetting: (path, value) => {
          set((state) => {
            // Logic to update nested state immutably
            // Validation and persistence handled by middleware or internal logic
          });
        },
        // ... other actions like initialize, reset
      }),
      {
        name: 'logseq-xr-settings',
        getStorage: () => localStorage,
        // ... other persist options
      }
    )
  )
);
```

### Graph Data Manager (`client/src/features/graph/managers/graphDataManager.ts`)

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

### Settings Observer
The `SettingsObserver.ts` file is not present in the current codebase. Instead, the `settingsStore.ts` (which uses Zustand) directly handles subscriptions. Components can subscribe to specific parts of the settings store using Zustand's selectors, eliminating the need for a separate observer pattern implementation.

## State Persistence

The application persists state in several ways:

1.  **Local Storage** - User preferences and UI state (managed by Zustand's `persist` middleware).
2.  **Server Storage** - User settings synchronized to server (for authenticated users).
3.  **URL Parameters** - Shareable state in URL (not extensively used for persistence, more for initial configuration).

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

### Event Emitter
The `client/utils/eventEmitter.ts` file is not present in the current codebase. Communication between components is primarily handled through:
- React's component hierarchy (props and context).
- Zustand store subscriptions.
- Direct function calls.
- WebSocket messages for server-client communication.

### Direct Subscriptions

Components can subscribe directly to state stores.

**Example:**
```typescript
// Subscribe to graph data changes
const unsubscribe = graphDataManager.subscribe((data) => {
  // Update component based on new data
});

// Subscribe to settings changes
const unsubscribe = useSettingsStore.subscribe(
  (settings) => {
    // Update component based on new settings
  },
  (state) => state.settings // Selector to subscribe to the entire settings object
);
```

## Settings Structure

The settings are defined by the `Settings` interface in `client/src/features/settings/config/settings.ts`. The structure of the settings for UI presentation and control is further defined in `client/src/features/settings/config/settingsUIDefinition.ts`. This allows for a flexible and hierarchical organization of settings.

```typescript
// Simplified representation of the Settings interface from client/src/features/settings/config/settings.ts
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
      // ... other node settings
    };
    edges: {
      color: string;
      opacity: number;
      arrowSize: number;
      baseWidth: number;
      enableArrows: boolean;
      widthRange: [number, number];
      quality: 'low' | 'medium' | 'high';
      // ... other edge settings
    };
    physics: {
      enabled: boolean;
      attractionStrength: number;
      repulsionStrength: number;
      springStrength: number;
      damping: number;
      gravityStrength: number;
      centerAttractionStrength: number;
      // ... other physics settings
    };
    rendering: {
      ambientLightIntensity: number;
      directionalLightIntensity: number;
      environmentIntensity: number;
      backgroundColor: string;
      enableAmbientOcclusion: boolean;
      enableAntialiasing: boolean;
      enableShadows: boolean;
      // ... other rendering settings
    };
    animations: {
      enableNodeAnimations: boolean;
      enableMotionBlur: boolean;
      motionBlurStrength: number;
      // ... other animation settings
    };
    labels: {
      enableLabels: boolean;
      desktopFontSize: number;
      textColor: string;
      textOutlineColor: string;
      // ... other label settings
    };
    bloom: {
      enabled: boolean;
      strength: number;
      radius: number;
      threshold: number;
      // ... other bloom settings
    };
    hologram: {
      ringCount: number;
      sphereSizes: number[];
      ringRotationSpeed: number;
      globalRotationSpeed: number;
      // ... other hologram settings
    };
  };
  system: {
    websocket: {
      reconnectAttempts: number;
      reconnectDelay: number;
      binaryChunkSize: number;
      compressionEnabled: boolean;
      // ... other websocket settings
    };
    debug: {
      enabled: boolean;
      enableDataDebug: boolean;
      enableWebsocketDebug: boolean;
      logBinaryHeaders: boolean;
      // ... other debug settings
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
    // ... other XR settings
  };
  ai: {
    enabled: boolean;
    defaultModel: string;
    temperature: number;
    // ... other AI settings
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

The application uses immutability patterns to prevent unexpected state changes, primarily facilitated by the `immer` middleware in Zustand. This ensures that state updates always result in new state objects, simplifying change detection and preventing unintended side effects.

**Example of Immutable Update (handled by Immer in Zustand):**
```typescript
// When using Immer with Zustand, direct modification within the 'set' callback is safe:
set((state) => {
  state.user.name = "New Name"; // This is automatically converted to an immutable update
});

// Without Immer, you would manually create new objects:
set((state) => ({
  ...state,
  user: {
    ...state.user,
    name: "New Name"
  }
}));
```

## Related Documentation

- [Components](components.md) - Component relationships and dependencies
- [Architecture](architecture.md) - Overall system architecture
- [WebSocket Communication](websocket.md) - Real-time state updates