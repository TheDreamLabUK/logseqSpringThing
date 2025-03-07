# Debugging Guide

This guide provides tools and techniques for debugging LogseqXR during development.

## WebSocket Diagnostics

WebSocket connections are critical for real-time graph updates. These diagnostic tools help identify and resolve connection issues.

### Built-in Diagnostics

LogseqXR includes built-in diagnostic functions for WebSocket troubleshooting:

```typescript
import { diagnoseWebSocketIssues } from './diagnostics';

// Run comprehensive WebSocket diagnostics
diagnoseWebSocketIssues();
```

The diagnostics will check:
- WebSocket URL validation
- Connection status
- Binary protocol validation
- Message processing
- GraphDataManager integration

### Common WebSocket Issues

1. **Connection Failures**
   - Check that the server is running on the expected port (default: 4000)
   - Verify network connectivity between client and server
   - Check for firewall or proxy restrictions

2. **Binary Data Issues**
   - Ensure data format matches the expected binary protocol
   - Check for compression/decompression errors
   - Verify that node IDs are correctly parsed

3. **Performance Problems**
   - Monitor message size and frequency
   - Check for excessive updates that might overwhelm the client
   - Verify that position deadbands are properly configured

### WebSocket Monitoring

You can monitor WebSocket traffic using browser developer tools:
1. Open DevTools (F12 or Ctrl+Shift+I)
2. Go to the Network tab
3. Filter by "WS" to show WebSocket connections
4. Select the connection to view messages

## Rendering Issues

### Three.js Debugging

For Three.js rendering issues:

1. **Enable Stats Panel**
   ```javascript
   import Stats from 'three/examples/jsm/libs/stats.module';
   const stats = new Stats();
   document.body.appendChild(stats.dom);
   ```

2. **Scene Inspector**
   ```javascript
   import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
   const gui = new GUI();
   gui.add(mesh.position, 'x', -10, 10);
   ```

3. **Common Rendering Issues**
   - Check camera position and frustum
   - Verify that lights are properly configured
   - Ensure materials and textures are loaded correctly

## Performance Optimization

### Client-Side Performance

1. **Identify Bottlenecks**
   - Use Chrome DevTools Performance tab to record and analyze performance
   - Look for long tasks in the Main thread
   - Check for excessive garbage collection

2. **Rendering Optimization**
   - Reduce draw calls by combining geometries
   - Use instanced meshes for similar objects
   - Implement level-of-detail (LOD) for complex scenes

3. **Memory Management**
   - Dispose of unused Three.js objects (geometries, materials, textures)
   - Use object pooling for frequently created/destroyed objects
   - Monitor memory usage with Chrome DevTools Memory tab

### Server-Side Performance

1. **Profiling Rust Code**
   ```bash
   RUSTFLAGS='-C debuginfo=2' cargo build --release
   perf record -g ./target/release/logseq-xr
   perf report
   ```

2. **WebSocket Optimization**
   - Adjust compression settings for optimal balance
   - Configure position and velocity deadbands appropriately
   - Tune update frequency based on available bandwidth

## Logging

### Client-Side Logging

LogseqXR uses a custom logger with different verbosity levels:

```typescript
import { logger } from './utils/logger';

logger.debug('Detailed information');
logger.info('General information');
logger.warn('Warning message');
logger.error('Error message');
```

Enable verbose logging in the browser console:
```javascript
localStorage.setItem('debug', 'true');
```

### Server-Side Logging

Control Rust logging level through the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug cargo run
```

Available log levels:
- `error`: Only errors
- `warn`: Warnings and errors
- `info`: General information (default)
- `debug`: Detailed debugging information
- `trace`: Very verbose tracing information

## Common Issues and Solutions

### Node Position Issues

If nodes are not positioned correctly:
1. Check the binary protocol implementation
2. Verify that position data is correctly encoded/decoded
3. Ensure the client is correctly applying position updates

### WebSocket Connection Problems

If WebSocket connections fail:
1. Verify the server is running and accessible
2. Check for network issues or firewall restrictions
3. Ensure the WebSocket URL is correctly constructed

### Performance Degradation

If performance degrades over time:
1. Check for memory leaks using browser memory profiling
2. Monitor CPU usage for excessive calculations
3. Verify that unused Three.js objects are properly disposed
4. Check for excessive WebSocket message traffic
