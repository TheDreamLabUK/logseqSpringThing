# Debug Enhancement Plan

## Phase 1: Debug System Enhancement

### 1. Debug State Enhancements (client/core/debugState.ts)

Add new debug categories:
```typescript
export interface DebugState {
    enabled: boolean;
    logFullJson: boolean;
    enableDataDebug: boolean;
    enableWebsocketDebug: boolean;
    logBinaryHeaders: boolean;
    // New debug categories
    enablePhysicsDebug: boolean;    // For physics/force calculations
    enableNodeDebug: boolean;       // For node position/velocity tracking
    enableShaderDebug: boolean;     // For shader compilation/linking
    enableMatrixDebug: boolean;     // For matrix transformations
    enablePerformanceDebug: boolean; // For performance monitoring
}
```

### 2. Logger Enhancements (client/core/logger.ts)

Add new logging features:
```typescript
export interface LogMetadata {
    nodeId?: string;
    position?: Vector3;
    velocity?: Vector3;
    component?: string;
    operation?: string;
}

export interface Logger {
    debug: (message: string, metadata?: LogMetadata) => void;
    log: (message: string, metadata?: LogMetadata) => void;
    info: (message: string, metadata?: LogMetadata) => void;
    warn: (message: string, metadata?: LogMetadata) => void;
    error: (message: string, metadata?: LogMetadata) => void;
    physics: (message: string, metadata?: LogMetadata) => void;
    matrix: (message: string, metadata?: LogMetadata) => void;
    performance: (message: string, metadata?: LogMetadata) => void;
}
```

## Phase 2: Implementation Plan

### 1. NodeInstanceManager.ts Enhancements

```typescript
// Debug gates
const shouldLogPhysics = () => debugState.isEnabled() && debugState.getState().enablePhysicsDebug;
const shouldLogMatrix = () => debugState.isEnabled() && debugState.getState().enableMatrixDebug;

class NodeInstanceManager {
    private logger = createLogger('NodeInstanceManager');

    private validateNodeData(nodeId: string, position: Vector3, velocity: Vector3) {
        if (shouldLogPhysics()) {
            this.logger.physics('Validating node data', {
                nodeId,
                position,
                velocity,
                operation: 'validate'
            });
        }

        if (!isFinite(position.x) || !isFinite(position.y) || !isFinite(position.z)) {
            this.logger.error('Invalid position detected', {
                nodeId,
                position,
                operation: 'validate'
            });
            return false;
        }

        if (!isFinite(velocity.x) || !isFinite(velocity.y) || !isFinite(velocity.z)) {
            this.logger.error('Invalid velocity detected', {
                nodeId,
                velocity,
                operation: 'validate'
            });
            return false;
        }

        return true;
    }

    private updateNodePositions() {
        if (shouldLogPhysics()) {
            this.logger.physics('Starting node position updates', {
                component: 'physics',
                operation: 'updatePositions'
            });
        }

        // Log nodes array before processing
        if (shouldLogPhysics()) {
            this.logger.physics('Current nodes state', {
                component: 'physics',
                operation: 'preUpdate',
                nodes: this.nodes.map(node => ({
                    id: node.id,
                    position: node.position,
                    velocity: node.velocity
                }))
            });
        }

        // Existing update logic with validation
        for (const node of this.nodes) {
            if (!this.validateNodeData(node.id, node.position, node.velocity)) {
                // Handle invalid data
                this.handleInvalidNodeData(node);
                continue;
            }

            // Proceed with update
            // ... existing update logic ...
        }
    }

    private handleInvalidNodeData(node: Node) {
        this.logger.error('Handling invalid node data', {
            nodeId: node.id,
            position: node.position,
            velocity: node.velocity,
            operation: 'recovery'
        });

        // Reset to safe values
        node.position.set(0, 0, 0);
        node.velocity.set(0, 0, 0);
    }
}
```

### 2. WebSocket Service Enhancements

```typescript
class WebSocketService {
    private logger = createLogger('WebSocketService');

    private handleBinaryMessage(data: ArrayBuffer) {
        if (debugState.shouldLogBinaryHeaders()) {
            this.logger.debug('Received binary message', {
                size: data.byteLength,
                operation: 'receive'
            });
        }

        try {
            // Decompress and process data
            const nodes = this.processNodeData(data);
            
            if (debugState.isDataDebugEnabled()) {
                this.logger.debug('Processed node data', {
                    nodeCount: nodes.length,
                    operation: 'process'
                });
            }

            // Validate each node
            for (const node of nodes) {
                if (!this.validateNodeData(node)) {
                    this.logger.error('Invalid node data in message', {
                        nodeId: node.id,
                        data: node,
                        operation: 'validate'
                    });
                    continue;
                }
                // Process valid node
            }
        } catch (error) {
            this.logger.error('Binary message processing failed', {
                error,
                operation: 'process'
            });
        }
    }
}
```

### 3. Shader Material Enhancements

```typescript
class HologramShaderMaterial {
    private logger = createLogger('HologramShaderMaterial');

    constructor() {
        if (debugState.getState().enableShaderDebug) {
            this.logger.debug('Initializing shader', {
                component: 'shader',
                operation: 'init'
            });
        }

        // Check WebGL version
        const gl = this.renderer.getContext();
        const isWebGL2 = gl instanceof WebGL2RenderingContext;
        
        if (!isWebGL2 && debugState.getState().enableShaderDebug) {
            this.logger.warn('WebGL 2 not available, using fallback', {
                component: 'shader',
                operation: 'init'
            });
        }

        // Monitor shader compilation
        this.onBeforeCompile = (shader) => {
            if (debugState.getState().enableShaderDebug) {
                this.logger.debug('Compiling shader', {
                    component: 'shader',
                    operation: 'compile',
                    vertexShader: shader.vertexShader,
                    fragmentShader: shader.fragmentShader
                });
            }
        };
    }
}
```

## Phase 3: Performance Monitoring

```typescript
class PerformanceMonitor {
    private logger = createLogger('Performance');
    private metrics: Map<string, number> = new Map();

    public startOperation(name: string) {
        if (debugState.getState().enablePerformanceDebug) {
            this.metrics.set(name, performance.now());
        }
    }

    public endOperation(name: string) {
        if (debugState.getState().enablePerformanceDebug) {
            const startTime = this.metrics.get(name);
            if (startTime) {
                const duration = performance.now() - startTime;
                this.logger.performance(`Operation: ${name}`, {
                    duration,
                    operation: 'measure'
                });
            }
        }
    }
}
```

## Implementation Steps

1. Update debugState.ts with new debug categories
2. Enhance logger.ts with new features
3. Implement NodeInstanceManager changes
4. Update WebSocket service
5. Enhance shader materials
6. Add performance monitoring
7. Update settings UI to expose new debug options
8. Add documentation for new debug features

## Testing Plan

1. Test each debug category independently
2. Verify log output format and content
3. Test performance impact of debug logging
4. Validate error recovery mechanisms
5. Test WebGL version detection and fallbacks
6. Verify performance monitoring accuracy

## Notes

- All debug logging is properly gated behind debug flags
- Performance monitoring is optional and can be enabled/disabled
- Error recovery mechanisms are in place for invalid data
- Shader compilation monitoring helps identify WebGL issues
- Binary message processing has enhanced error handling