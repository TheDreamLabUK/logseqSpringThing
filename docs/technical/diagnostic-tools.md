# Diagnostic Tools Documentation

This document provides a comprehensive guide to all the diagnostic tools available in the codebase for troubleshooting, debugging, and system verification.

## Overview

The codebase includes several diagnostic tools to help identify and fix issues across different system components. These tools can be run from the browser console, CLI, or integrated into testing environments.

## Available Diagnostic Tools

### 1. Node ID Binding Diagnostics

**Location**: `client/diagnostics/nodeIdBindingDiagnostics.ts`

**Purpose**: Diagnoses issues with node ID binding between binary WebSocket protocol and metadata visualization.

**Usage**:
```javascript
// Run in browser console
NodeBindingDiagnostics.runDiagnostics();

// Enable real-time monitoring
NodeBindingDiagnostics.enableMonitoring();

// Get recommendations for fixing issues
NodeBindingDiagnostics.getRecommendations();
```

**Common Scenarios**:
- All nodes showing the same metadata (e.g., "1000B" file size)
- Metadata not updating with node position changes
- Missing or incorrect metadata fields

### 2. WebSocket Diagnostics

**Location**: 
- `client/websocket-diagnostics.ts` (TypeScript module)
- `client/websocket-diagnostics-browser.js` (Browser-compatible version)
- `client/websocket-test.ts` (Simple connection testing)

**Purpose**: Diagnose WebSocket connection issues, binary protocol problems, and network performance.

**Usage**:
```javascript
// In browser console
WebSocketDiagnostics.runDiagnostics();

// Test specific components
WebSocketDiagnostics.testBinaryProtocol();
WebSocketDiagnostics.checkLatency();
WebSocketDiagnostics.validateNodeData();

// Simple connection test
testWebSocket.runTests();
```

**Common Scenarios**:
- Connection failures or frequent disconnects
- Binary protocol errors or data corruption
- High latency or performance issues
- Reconnection problems

### 3. WebGL Diagnostics

**Location**: `client/webgl-diagnostics.js`

**Purpose**: Diagnose WebGL rendering issues, compatibility problems, and performance bottlenecks.

**Usage**:
```javascript
// In browser console
WebGLDiagnostics.runDiagnostics();

// Test specific WebGL features
WebGLDiagnostics.checkExtensionSupport();
WebGLDiagnostics.testShaderCompilation();
WebGLDiagnostics.checkContextLimits();
```

**Common Scenarios**:
- Shader compilation errors
- WebGL context loss
- Rendering artifacts or missing elements
- Performance issues with complex scenes

### 4. System Diagnostics

**Location**: `client/diagnostics/systemDiagnostics.ts`

**Purpose**: Comprehensive system checks for environment, capabilities, and configuration.

**Usage**:
```javascript
// Import in code
import { runSystemDiagnostics } from './diagnostics/systemDiagnostics';

// Run all diagnostics
runSystemDiagnostics();

// Run specific checks
checkPlatformCapabilities();
checkWebGLSupport();
checkNetworkConnectivity();
```

**Common Scenarios**:
- Startup issues or initialization failures
- Platform compatibility problems
- Feature detection and capability reporting
- Environment validation

### 5. General Diagnostics

**Location**: `client/diagnostics.ts`

**Purpose**: Entry point for all diagnostic functions and user-facing diagnostic tools.

**Usage**:
```javascript
// Import in code
import { runDiagnostics } from './diagnostics';

// Run diagnostic suite
runDiagnostics();

// For browser console
window.VisionFlowDiagnostics.run();
```

**Common Scenarios**:
- General troubleshooting and health checks
- Automated testing and CI/CD validation
- User-facing diagnostic reporting

## Using Diagnostic Tools in Production

### Browser Console Method

For production environments, you can load and run diagnostics directly in the browser console:

1. Open browser dev tools (F12 or Ctrl+Shift+I)
2. Copy and paste the following code:

```javascript
// Load WebSocket diagnostics
fetch('/client/websocket-diagnostics-browser.js')
  .then(response => response.text())
  .then(code => eval(code))
  .then(() => console.log('WebSocket diagnostics loaded'));

// Load WebGL diagnostics
fetch('/client/webgl-diagnostics.js')
  .then(response => response.text())
  .then(code => eval(code))
  .then(() => console.log('WebGL diagnostics loaded'));

// Then run diagnostics
WebSocketDiagnostics.runDiagnostics();
WebGLDiagnostics.runDiagnostics();
```

### Alternative Method

If the fetch method doesn't work, you can copy the diagnostic tool code from your development environment and paste it directly into the console.

## Integrated Debugging

The diagnostic tools integrate with the application's debug state system. To enable specific debug categories:

```javascript
// Check current debug state
console.log('Node debugging:', debugState.isNodeDebugEnabled());
console.log('WebSocket debugging:', debugState.isWebsocketDebugEnabled());
console.log('Data debugging:', debugState.isDataDebugEnabled());
```

## Diagnostic Output Formats

Diagnostic tools produce output in several formats:

1. **Console Logs**: Detailed information sent to browser console
2. **JSON Reports**: Structured data for programmatic analysis
3. **Visual Indicators**: On-screen elements showing diagnostic status
4. **Performance Metrics**: Timing and resource utilization data

## Common Issues and Troubleshooting

### WebSocket Connectivity

If you encounter WebSocket connection issues:

1. Run `WebSocketDiagnostics.runDiagnostics()`
2. Check network tab for failed WebSocket connections
3. Verify API endpoints in network settings
4. Check for CORS or mixed content issues

### Rendering Problems

For visual or rendering issues:

1. Run `WebGLDiagnostics.runDiagnostics()`
2. Check for WebGL context support
3. Verify shader compilation success
4. Look for errors in materials or geometries

### Node Binding Issues

For issues with node data or metadata:

1. Run `NodeBindingDiagnostics.runDiagnostics()`
2. Check node ID consistency across components
3. Verify metadata is correctly associated with nodes
4. Look for type conversion issues between numeric and string IDs

### System Performance

For performance issues:

1. Run system diagnostics
2. Check FPS counter and resource utilization
3. Look for rendering bottlenecks
4. Verify data processing efficiency

## Extending the Diagnostic Framework

### Adding New Diagnostic Tools

To add a new diagnostic tool:

1. Create a new file in `client/diagnostics/` following the pattern of existing tools
2. Implement the singleton pattern for consistency
3. Expose run methods and test-specific functions
4. Add browser console integration
5. Update this documentation

### Diagnostic Tool API Conventions

All diagnostic tools should follow these conventions:

1. Implement `runDiagnostics()` as the main entry point
2. Return diagnostic results in a consistent format
3. Provide method-specific tests for targeted debugging
4. Include clear documentation and usage examples
5. Add console-friendly output with appropriate log levels

## Development Best Practices

When working with diagnostic tools:

1. Run relevant diagnostics early when investigating issues
2. Include diagnostic output in bug reports
3. Write tests that verify diagnostic accuracy
4. Keep diagnostic tools updated as the codebase evolves
5. Use diagnostics proactively during development to catch issues early

## Conclusion

These diagnostic tools provide a powerful framework for identifying and resolving issues throughout the application. By understanding and effectively using these tools, developers can quickly troubleshoot problems and ensure application stability and performance.