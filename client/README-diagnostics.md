# VisionFlow Diagnostics Tools

This directory contains diagnostic tools for troubleshooting issues with the VisionFlow application, particularly focusing on WebSocket connections and WebGL rendering.

## Available Diagnostic Tools

### 1. WebSocket Diagnostics

#### Browser Console Tool (`websocket-diagnostics-browser.js`)

A standalone browser-compatible tool that can be loaded directly in the browser console:

1. Open the browser console (F12 or Ctrl+Shift+I)
2. Copy the contents of `websocket-diagnostics-browser.js`
3. Paste into the browser console
4. The diagnostics will run automatically, or you can run them manually:
   ```javascript
   WebSocketDiagnostics.runDiagnostics();
   ```

This tool provides detailed information about:
- WebSocket connection status
- Binary protocol validation
- Network latency measurements
- Message size analysis
- Reconnection testing

#### WebSocket Test Script (`websocket-test.ts`)

A simpler test script focused specifically on WebSocket connection testing:

```javascript
// In the browser console
testWebSocket.runTests();
```

### 2. WebGL Diagnostics (`webgl-diagnostics.js`)

A browser-compatible tool for diagnosing WebGL rendering issues:

1. Open the browser console (F12 or Ctrl+Shift+I)
2. Copy the contents of `webgl-diagnostics.js`
3. Paste into the browser console
4. The diagnostics will run automatically, or you can run them manually:
   ```javascript
   WebGLDiagnostics.runDiagnostics();
   ```

This tool provides detailed information about:
- WebGL version and capabilities
- Extension support
- Shader compilation testing
- WebGL context limits
- Context loss recovery support

## Common Issues and Solutions

### WebSocket Issues

1. **Connection Failures**
   - Check if the WebSocket URL is correct (wss:// for HTTPS sites)
   - Verify network connectivity to the API endpoint
   - Check for firewall or proxy blocking WebSocket connections

2. **Binary Protocol Errors**
   - Ensure client and server agree on the binary format (28 bytes per node)
   - Check for compression/decompression issues with large messages

3. **Reconnection Problems**
   - Verify the server is properly handling reconnection attempts
   - Check for network stability issues

### WebGL Issues

1. **Shader Compilation Errors**
   - Replace custom shaders with built-in Three.js materials
   - Ensure shaders are compatible with the browser's WebGL version
   - Simplify shader code to avoid complex operations

2. **WebGL Context Loss**
   - Reduce the number of WebGL contexts (canvases/renderers)
   - Dispose unused materials, textures, and geometries
   - Use shared materials and geometries where possible
   - Reduce texture sizes and complexity

3. **Performance Issues**
   - Reduce the number of objects in the scene
   - Optimize shader complexity
   - Use level-of-detail techniques for complex scenes

## Using Built-in Three.js Materials

To avoid custom shader issues, consider using built-in Three.js materials:

```javascript
// Instead of custom shader materials:
const material = new THREE.MeshPhongMaterial({
  color: 0x00ff00,
  emissive: 0x003300,
  specular: 0x00ff00,
  shininess: 30,
  transparent: true,
  opacity: 0.8
});

// For edges, use LineBasicMaterial:
const edgeMaterial = new THREE.LineBasicMaterial({
  color: 0x0000ff,
  linewidth: 1,
  transparent: true,
  opacity: 0.7
});
```

## Production Debugging

For debugging in production:

1. Open the browser console on the production site
2. Load the diagnostic tools:
   ```javascript
   // For WebSocket diagnostics
   fetch('/client/websocket-diagnostics-browser.js')
     .then(response => response.text())
     .then(code => eval(code));

   // For WebGL diagnostics
   fetch('/client/webgl-diagnostics.js')
     .then(response => response.text())
     .then(code => eval(code));
   ```

3. If the files aren't directly accessible, copy and paste the tool code from your development environment

## Reporting Issues

When reporting issues, please include:
1. The complete diagnostic output from both tools
2. Browser and OS information
3. Steps to reproduce the issue
4. Any error messages from the console 