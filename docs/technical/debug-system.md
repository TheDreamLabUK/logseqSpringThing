# Debug System Documentation

## Overview

The LogseqXR debug system provides comprehensive debugging capabilities across different aspects of the application. The system is designed to be modular and configurable, allowing developers to enable specific debug categories as needed.

## Debug Categories

### Core Debug Features
- **General Debug** (`enabled`): Master switch for debug functionality
- **Data Debug** (`enableDataDebug`): Debug data processing and transformations
- **WebSocket Debug** (`enableWebsocketDebug`): Debug WebSocket communications
- **Binary Headers** (`logBinaryHeaders`): Log binary message headers
- **Full JSON** (`logFullJson`): Log complete JSON messages

### Enhanced Debug Categories
- **Physics Debug** (`enablePhysicsDebug`): Monitor physics and force calculations
  - Node position updates
  - Velocity calculations
  - Force application
  - Collision detection

- **Node Debug** (`enableNodeDebug`): Track node-related operations
  - Position tracking
  - Velocity monitoring
  - Node validation
  - Instance management

- **Shader Debug** (`enableShaderDebug`): Monitor shader operations
  - Shader compilation
  - WebGL context validation
  - Uniform validation
  - Shader linking
  - WebGL version detection

- **Matrix Debug** (`enableMatrixDebug`): Track matrix transformations
  - Matrix validation
  - Transform operations
  - Position/rotation/scale decomposition

- **Performance Debug** (`enablePerformanceDebug`): Monitor performance metrics
  - Operation timing
  - Average durations
  - Operation counts
  - Performance bottleneck detection

## Usage

### Enabling Debug Features

1. Access the debug settings in the UI control panel under System > Debug
2. Toggle individual debug categories as needed
3. Debug output will appear in the browser console

### Debug Output Format

Debug messages follow a consistent format:
```
[Timestamp] [Context] Message {metadata}
```

Example:
```
14:30:45.123 [NodeInstanceManager] Validating node data {
  "nodeId": "node-1",
  "position": {"x": 0, "y": 0, "z": 0},
  "operation": "validate"
}
```

### Performance Monitoring

The performance monitoring system allows tracking of specific operations:

```typescript
const performanceMonitor = new PerformanceMonitor();

// Start timing an operation
performanceMonitor.startOperation('nodeUpdate');

// ... perform operation ...

// End timing and log results
performanceMonitor.endOperation('nodeUpdate');
```

Output includes:
- Operation duration
- Average duration over time
- Operation count
- Performance trends

## Best Practices

1. **Selective Debugging**: Enable only the debug categories needed for your current task
2. **Performance Impact**: Be aware that enabling debug logging may impact performance
3. **Error Recovery**: Use debug output to track error recovery mechanisms
4. **Validation**: Utilize debug logging to verify data integrity and state transitions

## Implementation Details

The debug system is implemented across several core components:

- `debugState.ts`: Manages debug state and feature flags
- `logger.ts`: Provides logging functionality with metadata support
- `PerformanceMonitor`: Handles performance tracking and metrics
- UI Components: Exposes debug settings in the control panel

Each debug category is properly gated behind feature flags to minimize performance impact when disabled.