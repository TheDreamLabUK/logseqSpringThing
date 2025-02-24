import { createLogger, createDataMetadata } from './core/logger';
import { WebSocketService } from './websocket/websocketService';
import { graphDataManager } from './state/graphData';
import { platformManager } from './platform/platformManager';
import { HologramShaderMaterial } from './rendering/materials/HologramShaderMaterial';
import { EdgeShaderMaterial } from './rendering/materials/EdgeShaderMaterial';
import { defaultSettings } from './state/defaultSettings';
import { buildWsUrl } from './core/api';

const logger = createLogger('Diagnostics');

export function runDiagnostics() {
  logger.info('Running system diagnostics...');
  
  // Check WebGL support
  checkWebGLSupport();
  
  // Check WebSocket configuration
  checkWebSocketConfig();
  
  // Check shader compatibility
  checkShaderCompatibility();
  
  // Check platform capabilities
  checkPlatformCapabilities();
}

function checkWebGLSupport() {
  logger.info('Checking WebGL support...');
  
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  const gl1 = canvas.getContext('webgl');
  
  if (!gl && !gl1) {
    logger.error('WebGL not supported at all');
    return;
  }
  
  if (gl) {
    logger.info('WebGL2 is supported');
    // Check for specific extensions needed by shaders
    const extensions = gl.getSupportedExtensions();
    if (extensions) {
      logger.info('Supported WebGL2 extensions:', createDataMetadata({ extensions }));
    }
  } else {
    logger.warn('WebGL2 not supported, falling back to WebGL1');
    // This could be a problem for shaders using #version 300 es
  }
}

function checkWebSocketConfig() {
  logger.info('Checking WebSocket configuration...');
  
  // Check if WebSocket is supported
  if (!('WebSocket' in window)) {
    logger.error('WebSocket not supported in this browser');
    return;
  }
  
  // Check if WebSocketService is properly initialized
  const wsService = WebSocketService.getInstance();
  
  // Get the WebSocket URL that would be used
  const wsUrl = buildWsUrl();
  logger.info('WebSocket URL:', createDataMetadata({ url: wsUrl }));
  
  // Check connection state
  const connectionState = wsService.getConnectionStatus();
  logger.info('WebSocketService status:', createDataMetadata({ 
    state: connectionState,
    isInitialized: wsService !== null
  }));
  
  // Test WebSocket connectivity
  try {
    // Create a test WebSocket to check if the endpoint is reachable
    const testWs = new WebSocket(wsUrl);
    testWs.onopen = () => {
      logger.info('Test WebSocket connection successful');
      testWs.close();
    };
    testWs.onerror = (error) => {
      logger.error('Test WebSocket connection failed:', createDataMetadata({ error }));
    };
    
    // Set a timeout to close the test connection if it doesn't connect
    setTimeout(() => {
      if (testWs.readyState !== WebSocket.OPEN) {
        logger.warn('Test WebSocket connection timed out');
        testWs.close();
      }
    }, 5000);
  } catch (error) {
    logger.error('Failed to create test WebSocket:', createDataMetadata({ error }));
  }
  
  // Check if GraphDataManager has WebSocketService configured
  const gdm = graphDataManager;
  
  // Try to set the WebSocket service
  try {
    // Create a temporary WebSocket adapter to test connection
    const testWsAdapter = {
      send: (_data: ArrayBuffer) => {
        logger.info('Test WebSocket send called');
      }
    };
    
    gdm.setWebSocketService(testWsAdapter);
    logger.info('Successfully configured WebSocketService in GraphDataManager');
  } catch (error) {
    logger.error('Failed to configure WebSocketService:', createDataMetadata({ error }));
  }
}

function checkShaderCompatibility() {
  logger.info('Checking shader compatibility...');
  
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  
  if (!gl) {
    logger.error('Cannot check shader compatibility - WebGL not available');
    return;
  }
  
  // Check if we're using WebGL2 (required for #version 300 es)
  const isWebGL2 = gl instanceof WebGL2RenderingContext;
  logger.info(`Using WebGL${isWebGL2 ? '2' : '1'}`);
  
  if (!isWebGL2) {
    logger.warn('WebGL2 not available - shaders using #version 300 es will fail');
    logger.info('Recommendation: Update shader code to be compatible with WebGL1');
  }
  
  // Try to create shader materials to check for compilation errors
  try {
    // Test HologramShaderMaterial creation without assigning to unused variable
    if (new HologramShaderMaterial(defaultSettings)) {
      logger.info('HologramShaderMaterial created successfully');
    }
  } catch (error) {
    logger.error('Failed to create HologramShaderMaterial:', createDataMetadata({ error }));
  }
  
  try {
    // Test EdgeShaderMaterial creation without assigning to unused variable
    if (new EdgeShaderMaterial(defaultSettings)) {
      logger.info('EdgeShaderMaterial created successfully');
    }
  } catch (error) {
    logger.error('Failed to create EdgeShaderMaterial:', createDataMetadata({ error }));
  }
}

function checkPlatformCapabilities() {
  logger.info('Checking platform capabilities...');
  
  const capabilities = platformManager.getCapabilities();
  logger.info('Platform capabilities:', createDataMetadata({ capabilities }));
  
  const platform = platformManager.getPlatform();
  logger.info('Detected platform:', createDataMetadata({ platform }));
  
  if (platformManager.isXRSupported()) {
    logger.info('XR is supported on this platform');
  } else {
    logger.warn('XR is not supported on this platform');
  }
}

// Export a function to fix common issues
export function applyFixes() {
  logger.info('Applying fixes for common issues...');
  
  // Fix 1: Configure WebSocket service
  const wsService = WebSocketService.getInstance();
  // Need to adapt the WebSocketService to match the expected interface
  const wsAdapter = {
    send: (data: ArrayBuffer) => {
      wsService.sendMessage({ type: 'binaryData', data });
    }
  };
  
  graphDataManager.setWebSocketService(wsAdapter);
  logger.info('WebSocket service configured for GraphDataManager');
  
  // Fix 2: Check if we need to modify shader version
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  
  if (!gl) {
    logger.warn('WebGL2 not available - shaders need to be modified');
    logger.info('Please update shader code in EdgeShaderMaterial.ts and HologramShaderMaterial.ts');
    logger.info('Change "#version 300 es" to be compatible with WebGL1');
  }
}

// Add a function to verify shader materials are properly configured
export function verifyShaderMaterials() {
  logger.info('Verifying shader materials configuration...');
  
  // Check WebGL version
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  const isWebGL2 = !!gl;
  
  logger.info(`WebGL2 support: ${isWebGL2 ? 'Yes' : 'No'}`);
  
  // Create test materials
  try {
    // Test material creation without assigning to unused variables
    if (new HologramShaderMaterial(defaultSettings) && 
        new EdgeShaderMaterial(defaultSettings)) {
      logger.info('Shader materials created successfully');
    }
    
    // Check if the renderer is properly set for both materials
    if (isWebGL2) {
      logger.info('Using WebGL2 for shader materials');
    } else {
      logger.warn('Using WebGL1 for shader materials - some advanced effects may be limited');
    }
    
    logger.info('Shader materials verification complete');
  } catch (error) {
    logger.error('Failed to verify shader materials:', createDataMetadata({ error }));
  }
}

// Add a comprehensive WebSocket diagnostic function
export function diagnoseWebSocketIssues() {
  logger.info('Running comprehensive WebSocket diagnostics...');
  
  // 1. Check WebSocket URL construction
  const wsUrl = buildWsUrl();
  logger.info('WebSocket URL:', createDataMetadata({ url: wsUrl }));
  
  // Parse the URL to check components
  try {
    const parsedUrl = new URL(wsUrl);
    logger.info('WebSocket URL components:', createDataMetadata({
      protocol: parsedUrl.protocol,
      host: parsedUrl.host,
      hostname: parsedUrl.hostname,
      port: parsedUrl.port,
      pathname: parsedUrl.pathname,
      search: parsedUrl.search
    }));
    
    // Check if using secure WebSocket
    if (parsedUrl.protocol !== 'wss:' && window.location.protocol === 'https:') {
      logger.warn('Using insecure WebSocket (ws://) with HTTPS site - browsers may block this');
    }
  } catch (error) {
    logger.error('Failed to parse WebSocket URL:', createDataMetadata({ error }));
  }
  
  // 2. Check WebSocketService state
  const wsService = WebSocketService.getInstance();
  const connectionState = wsService.getConnectionStatus();
  
  // Log detailed WebSocketService information
  logger.info('WebSocketService details:', createDataMetadata({ 
    state: connectionState,
    isInitialized: wsService !== null,
    reconnectAttempts: wsService['reconnectAttempts'] || 'unknown',
    maxReconnectAttempts: wsService['_maxReconnectAttempts'] || 'unknown'
  }));
  
  // 3. Test network connectivity
  try {
    // Try to fetch a small resource to check general network connectivity
    fetch('/api/user-settings', { method: 'HEAD' })
      .then(response => {
        logger.info('Network connectivity test successful:', createDataMetadata({ 
          status: response.status,
          ok: response.ok
        }));
      })
      .catch(error => {
        logger.error('Network connectivity test failed:', createDataMetadata({ error }));
      });
  } catch (error) {
    logger.error('Failed to initiate network test:', createDataMetadata({ error }));
  }
  
  // 4. Test WebSocket endpoint
  try {
    logger.info('Testing WebSocket endpoint...');
    const testWs = new WebSocket(wsUrl);
    
    testWs.onopen = () => {
      logger.info('WebSocket connection successful');
      // Send a ping message to test bidirectional communication
      testWs.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      
      // Close after 3 seconds to allow for response
      setTimeout(() => testWs.close(), 3000);
    };
    
    testWs.onmessage = (event) => {
      logger.info('Received WebSocket message:', createDataMetadata({ 
        type: typeof event.data,
        data: typeof event.data === 'string' ? event.data : 'binary data',
        size: typeof event.data === 'string' ? event.data.length : 
              (event.data instanceof ArrayBuffer ? event.data.byteLength : 'unknown')
      }));
    };
    
    testWs.onerror = (error) => {
      logger.error('WebSocket connection error:', createDataMetadata({ error }));
    };
    
    testWs.onclose = (event) => {
      logger.info('WebSocket connection closed:', createDataMetadata({ 
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      }));
    };
    
    // Set a timeout to close the test connection if it doesn't connect
    setTimeout(() => {
      if (testWs.readyState !== WebSocket.OPEN) {
        logger.warn('WebSocket connection timed out');
        testWs.close();
      }
    }, 5000);
  } catch (error) {
    logger.error('Failed to create test WebSocket:', createDataMetadata({ error }));
  }
  
  // 5. Check GraphDataManager configuration
  try {
    const gdm = graphDataManager;
    // Create a test message to see if it's properly configured
    // Using a comment instead of creating an unused variable
    // A typical node update is 28 bytes per node
    const testAdapter = {
      send: (data: ArrayBuffer) => {
        logger.info('GraphDataManager WebSocket send test:', createDataMetadata({ 
          byteLength: data.byteLength
        }));
        return true;
      }
    };
    
    gdm.setWebSocketService(testAdapter);
    logger.info('GraphDataManager WebSocket configuration test successful');
  } catch (error) {
    logger.error('GraphDataManager WebSocket configuration test failed:', createDataMetadata({ error }));
  }
  
  // 6. Check if WebSocketService can be used directly
  try {
    // Check if the connection state is CONNECTED
    if (connectionState === 'connected') {
      logger.info('WebSocketService is currently connected');
    } else {
      logger.warn(`WebSocketService is not connected (state: ${connectionState})`);
      
      // Try to connect if not already connecting or reconnecting
      if (connectionState === 'disconnected') {
        logger.info('Attempting to connect WebSocketService...');
        wsService.connect().catch(error => {
          logger.error('Failed to connect WebSocketService:', createDataMetadata({ error }));
        });
      }
    }
  } catch (error) {
    logger.error('Error checking WebSocketService connection:', createDataMetadata({ error }));
  }
  
  logger.info('WebSocket diagnostics complete');
} 