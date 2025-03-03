# WebSocket Diagnostics Tools

This directory contains tools for diagnosing WebSocket connection issues in the VisionFlow application.

## Overview

The WebSocket implementation in VisionFlow is critical for real-time graph updates. These diagnostic tools help identify and resolve issues with WebSocket connections, particularly in production environments.

## Available Tools

### 1. Built-in Diagnostics (`diagnostics.ts`)

The `diagnostics.ts` file contains several functions for diagnosing WebSocket issues:

- `runDiagnostics()`: Runs a comprehensive system check, including WebSocket configuration
- `checkWebSocketConfig()`: Specifically checks WebSocket support and configuration
- `diagnoseWebSocketIssues()`: Provides detailed WebSocket diagnostics including URL validation, connection testing, and GraphDataManager integration

Usage:
```typescript
import { diagnoseWebSocketIssues } from './diagnostics';

// Run comprehensive WebSocket diagnostics
diagnoseWebSocketIssues();
```

### 2. WebSocket Diagnostics Tool (`websocket-diagnostics.ts`)

A standalone TypeScript module that provides comprehensive WebSocket diagnostics:

- Connection status monitoring
- Binary protocol validation
- Network latency measurement
- Message size analysis
- Reconnection testing

Usage:
```typescript
import { WebSocketDiagnostics } from './websocket-diagnostics';

// Run the diagnostics
WebSocketDiagnostics.runDiagnostics();

// Configure diagnostics options
WebSocketDiagnostics.CONFIG.testDuration = 60000; // 1 minute test
WebSocketDiagnostics.CONFIG.verbose = true;
```

### 3. Browser Console Tool (`websocket-diagnostics-browser.js`)

A browser-compatible version that can be loaded directly in the browser console:

1. Copy the contents of `websocket-diagnostics-browser.js`
2. Paste into the browser console
3. Run the diagnostics:
   ```javascript
   WebSocketDiagnostics.runDiagnostics();
   ```

## Binary Protocol

The WebSocket binary protocol uses a specific format based on Three.js Vector3:

- Node data: 26 bytes per node
  - Node ID (uint16): 2 bytes
  - Position (3 x float32): 12 bytes (THREE.Vector3)
  - Velocity (3 x float32): 12 bytes (THREE.Vector3)

The client code now uses THREE.Vector3 objects throughout for consistency and performance. The protocol is a simple "firehose" of node updates without additional header metadata.

The diagnostics tools validate that binary messages conform to this format.

## Common Issues and Solutions

### Connection Issues

- **Mixed Content Blocking**: When using HTTPS, ensure WebSocket connections use WSS
- **CORS Issues**: Check server CORS configuration if connecting from different origins
- **Proxy/Firewall Blocking**: Some networks block WebSocket connections

### Binary Protocol Issues

- **Message Size Mismatch**: Ensure client and server agree on the binary format (26 bytes per node)
- **Compression Issues**: Large messages are compressed with zlib (pako.js); ensure decompression works correctly
- **Vector3 Integrity**: Verify Vector3 objects contain valid, non-NaN values for reliable rendering
- **Edge Updates**: Edge positions are derived from node positions and not directly included in the binary protocol

### Performance Issues

- **High Latency**: Check network conditions and server load
- **Message Frequency**: Adjust update frequency if too many messages are causing performance issues

## Debugging in Production

For production debugging:

1. Open the browser console
2. Load the browser diagnostics tool:
   ```javascript
   fetch('/client/websocket-diagnostics-browser.js')
     .then(response => response.text())
     .then(code => eval(code))
     .then(() => console.log('Diagnostics tool loaded'));
   ```
3. Run the diagnostics:
   ```javascript
   WebSocketDiagnostics.runDiagnostics();
   ```
4. Check the console for detailed logs and recommendations

## Troubleshooting Steps

1. Verify API connectivity with `WebSocketDiagnostics.testApiConnectivity()`
2. Run full diagnostics with `WebSocketDiagnostics.runDiagnostics()`
3. Review the diagnostics report for specific issues
4. Verify Vector3 data integrity with `WebSocketDiagnostics.validateVectorData()`
5. Check edge rendering efficiency with the Edge Manager diagnostics
6. Apply recommended fixes based on the diagnostics results

## Contributing

When modifying the WebSocket implementation:

1. Update the diagnostics tools to match any protocol changes
2. Maintain the use of THREE.Vector3 throughout the codebase
3. Test with both the TypeScript and browser versions
4. Document any changes to the binary protocol format
5. Ensure edge handling is properly tested since it depends on node position updates