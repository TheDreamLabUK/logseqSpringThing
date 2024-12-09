sequenceDiagram
    participant User
    participant App
    participant WebSocket
    participant Store
    participant Visualization
    participant GPU
    participant ThreeJS
    
    Note over App,ThreeJS: Initialization Phase
    User->>App: Load Application
    App->>Store: Create Pinia Stores
    App->>WebSocket: Initialize Connection
    App->>Visualization: Initialize Scene
    
    Note over WebSocket,Store: ⚠️ Potential Bottleneck #1
    WebSocket->>Store: Request Initial Data
    Store->>WebSocket: Send Binary Position Data
    
    rect rgb(255, 240, 240)
        Note over Visualization,GPU: ⚠️ High CPU Risk Area
        loop Force Simulation
            Visualization->>GPU: Update Positions
            GPU->>Store: Binary Position Update
            Store->>WebSocket: Send Position Data
            Note right of Store: Rate Limited to 60fps
        end
    end
    
    rect rgb(240, 240, 255)
        Note over ThreeJS,User: Render Loop
        loop Animation Frame
            Visualization->>ThreeJS: Update Scene
            ThreeJS->>User: Render Frame
            Note right of ThreeJS: ⚠️ Potential Frame Drop
        end
    end
    
    Note over WebSocket,Store: Data Flow
    WebSocket-->>Store: Binary Position Updates
    Store-->>Visualization: Update Node Positions
    
    rect rgb(255, 240, 240)
        Note over Store,Visualization: ⚠️ Feedback Loop Risk
        loop Position Updates
            Store->>Visualization: Update Positions
            Visualization->>GPU: Process Forces
            GPU->>Store: New Positions
            Note right of GPU: Can cause lockup if<br/>positions invalid
        end
    end
    
    rect rgb(240, 255, 240)
        Note over User,Visualization: User Interaction
        User->>Visualization: Drag Node
        Visualization->>Store: Update Position
        Store->>WebSocket: Send Update
        Note right of Store: ⚠️ Can flood websocket
    end
    
    Note over App,ThreeJS: Critical Points
    Note over WebSocket: 1. WebSocket message flood
    Note over Store: 2. Store update cascade
    Note over GPU: 3. GPU force calculation
    Note over ThreeJS: 4. Three.js render loop
    
    rect rgb(240, 240, 255)
        Note over Store,WebSocket: Safeguards
        Store-->>Store: Rate Limiting
        WebSocket-->>WebSocket: Message Throttling
        GPU-->>GPU: Position Validation
        ThreeJS-->>ThreeJS: Frame Throttling
    end
```

# Client Code Flow and Bottleneck Analysis

## Critical Points

1. **WebSocket Message Flood**
   - Risk: Client sending too many position updates
   - Mitigation: Rate limiting to 60fps
   - Location: websocketService.ts

2. **Store Update Cascade**
   - Risk: Store updates triggering excessive recalculations
   - Mitigation: Batch updates, debouncing
   - Location: visualization store

3. **GPU Force Calculation**
   - Risk: Invalid positions causing infinite force values
   - Mitigation: Position/velocity clamping
   - Location: compute_forces.cu

4. **Three.js Render Loop**
   - Risk: Frame drops from excessive updates
   - Mitigation: Frame throttling
   - Location: useVisualization.ts

## Feedback Loops

1. **Position Update Loop**
   ```
   Store -> Visualization -> GPU -> Store
   ```
   - Risk: Can cause lockup if positions become invalid
   - Mitigation: Position validation and clamping

2. **User Interaction Loop**
   ```
   User -> Visualization -> Store -> WebSocket
   ```
   - Risk: Can flood websocket with updates
   - Mitigation: Rate limiting and update throttling

## Safeguards

1. **Rate Limiting**
   - WebSocket updates limited to 60fps
   - Store updates batched and debounced
   - Frame rendering throttled

2. **Validation**
   - Position values clamped
   - Velocity values clamped
   - Binary message size validated

3. **Error Handling**
   - Error-only logging enforced
   - Invalid updates discarded
   - Connection retry backoff
