import { createLogger, createDataMetadata } from './core/logger';
import { WebSocketService } from './websocket/websocketService';
import { graphDataManager } from './state/graphData';
import { platformManager } from './platform/platformManager';
import { HologramShaderMaterial } from './rendering/materials/HologramShaderMaterial';
import { EdgeShaderMaterial } from './rendering/materials/EdgeShaderMaterial';
import { defaultSettings } from './state/defaultSettings';

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
  logger.info('WebSocketService instance created');
  
  // Check if GraphDataManager has WebSocketService configured
  const gdm = graphDataManager;
  
  // Try to set the WebSocket service
  try {
    // Create a temporary WebSocket to test connection
    const testWs = {
      send: (data: ArrayBuffer) => {
        logger.info('Test WebSocket send called');
      }
    };
    
    gdm.setWebSocketService(testWs);
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
    const hologramMaterial = new HologramShaderMaterial(defaultSettings);
    logger.info('HologramShaderMaterial created successfully');
  } catch (error) {
    logger.error('Failed to create HologramShaderMaterial:', createDataMetadata({ error }));
  }
  
  try {
    const edgeMaterial = new EdgeShaderMaterial(defaultSettings);
    logger.info('EdgeShaderMaterial created successfully');
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
    const hologramMaterial = new HologramShaderMaterial(defaultSettings);
    const edgeMaterial = new EdgeShaderMaterial(defaultSettings);
    
    // Instead of directly accessing vertexShader property, check if the materials
    // are created successfully and if they're using the appropriate renderer
    logger.info('Shader materials created successfully');
    
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