/**
 * WebGL Diagnostics Tool
 * 
 * This script can be run in the browser console to diagnose WebGL issues.
 * It provides detailed information about WebGL capabilities and limitations.
 * 
 * Usage:
 * 1. Open the browser console (F12 or Ctrl+Shift+I)
 * 2. Copy and paste this entire script into the console
 * 3. Press Enter to run the diagnostics
 * 4. Check the console for detailed logs about WebGL support
 */

(function() {
  // Utility functions
  const utils = {
    log(message, data) {
      const timestamp = new Date().toISOString();
      if (data) {
        console.log(`[${timestamp}] ${message}`, data);
      } else {
        console.log(`[${timestamp}] ${message}`);
      }
    },
    
    error(message, err) {
      const timestamp = new Date().toISOString();
      if (err) {
        console.error(`[${timestamp}] ERROR: ${message}`, err);
      } else {
        console.error(`[${timestamp}] ERROR: ${message}`);
      }
    },
    
    warn(message, data) {
      const timestamp = new Date().toISOString();
      if (data) {
        console.warn(`[${timestamp}] WARNING: ${message}`, data);
      } else {
        console.warn(`[${timestamp}] WARNING: ${message}`);
      }
    },
    
    success(message, data) {
      const timestamp = new Date().toISOString();
      const style = 'color: green; font-weight: bold';
      if (data) {
        console.log(`[${timestamp}] %c${message}`, style, data);
      } else {
        console.log(`[${timestamp}] %c${message}`, style);
      }
    }
  };

  // WebGL diagnostics
  const webglDiagnostics = {
    // Check if WebGL is supported
    checkWebGLSupport() {
      utils.log('Checking WebGL support...');
      
      const canvas = document.createElement('canvas');
      const gl2 = canvas.getContext('webgl2');
      const gl1 = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      
      if (!gl1 && !gl2) {
        utils.error('WebGL not supported in this browser');
        return null;
      }
      
      if (gl2) {
        utils.success('WebGL2 is supported');
        return { version: 2, context: gl2 };
      } else {
        utils.warn('WebGL2 not supported, falling back to WebGL1');
        return { version: 1, context: gl1 };
      }
    },
    
    // Get detailed information about WebGL capabilities
    getWebGLInfo(gl) {
      if (!gl) return null;
      
      const isWebGL2 = gl instanceof WebGL2RenderingContext;
      const version = isWebGL2 ? 2 : 1;
      
      // Get basic info
      const info = {
        version,
        vendor: gl.getParameter(gl.VENDOR),
        renderer: gl.getParameter(gl.RENDERER),
        glVersion: gl.getParameter(gl.VERSION),
        shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxCubeMapTextureSize: gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE),
        maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS),
        maxTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
        maxVertexTextureImageUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
        maxCombinedTextureImageUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
        maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
        maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
        maxVertexUniformVectors: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
        maxFragmentUniformVectors: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
        extensions: gl.getSupportedExtensions()
      };
      
      // Get WebGL2-specific info if available
      if (isWebGL2) {
        info.maxVertexUniformBlocks = gl.getParameter(gl.MAX_VERTEX_UNIFORM_BLOCKS);
        info.maxFragmentUniformBlocks = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_BLOCKS);
        info.maxUniformBufferBindings = gl.getParameter(gl.MAX_UNIFORM_BUFFER_BINDINGS);
        info.maxUniformBlockSize = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE);
        info.maxCombinedUniformBlocks = gl.getParameter(gl.MAX_COMBINED_UNIFORM_BLOCKS);
        info.maxCombinedVertexUniformComponents = gl.getParameter(gl.MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS);
        info.maxCombinedFragmentUniformComponents = gl.getParameter(gl.MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS);
        info.maxTransformFeedbackSeparateComponents = gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS);
        info.maxTransformFeedbackInterleavedComponents = gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
      }
      
      return info;
    },
    
    // Check if specific extensions are supported
    checkExtensions(gl) {
      if (!gl) return null;
      
      const extensions = gl.getSupportedExtensions();
      const criticalExtensions = [
        'ANGLE_instanced_arrays',
        'OES_texture_float',
        'OES_texture_half_float',
        'WEBGL_depth_texture',
        'OES_element_index_uint',
        'OES_standard_derivatives',
        'EXT_shader_texture_lod',
        'WEBGL_compressed_texture_s3tc',
        'WEBGL_debug_renderer_info'
      ];
      
      const extensionStatus = {};
      criticalExtensions.forEach(ext => {
        extensionStatus[ext] = extensions.includes(ext);
      });
      
      return extensionStatus;
    },
    
    // Test shader compilation
    testShaderCompilation(gl) {
      if (!gl) return null;
      
      const isWebGL2 = gl instanceof WebGL2RenderingContext;
      
      // Simple vertex shader
      const vertexShaderSource = isWebGL2 ?
        `#version 300 es
        in vec4 position;
        void main() {
          gl_Position = position;
        }` :
        `attribute vec4 position;
        void main() {
          gl_Position = position;
        }`;
      
      // Simple fragment shader
      const fragmentShaderSource = isWebGL2 ?
        `#version 300 es
        precision highp float;
        out vec4 fragColor;
        void main() {
          fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }` :
        `precision highp float;
        void main() {
          gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }`;
      
      try {
        // Create vertex shader
        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexShaderSource);
        gl.compileShader(vertexShader);
        
        if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
          const info = gl.getShaderInfoLog(vertexShader);
          throw new Error('Vertex shader compilation failed: ' + info);
        }
        
        // Create fragment shader
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentShaderSource);
        gl.compileShader(fragmentShader);
        
        if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
          const info = gl.getShaderInfoLog(fragmentShader);
          throw new Error('Fragment shader compilation failed: ' + info);
        }
        
        // Create program
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
          const info = gl.getProgramInfoLog(program);
          throw new Error('Program linking failed: ' + info);
        }
        
        // Clean up
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        gl.deleteProgram(program);
        
        return { success: true };
      } catch (error) {
        return { 
          success: false, 
          error: error.message 
        };
      }
    },
    
    // Count active WebGL contexts
    countActiveContexts() {
      // Create multiple canvases and contexts to see if we hit limits
      const contexts = [];
      const maxAttempts = 32; // Try to create up to 32 contexts
      
      utils.log(`Attempting to create ${maxAttempts} WebGL contexts to test limits...`);
      
      for (let i = 0; i < maxAttempts; i++) {
        const canvas = document.createElement('canvas');
        canvas.width = 16;
        canvas.height = 16;
        
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (gl) {
          contexts.push(gl);
        } else {
          break;
        }
      }
      
      utils.log(`Successfully created ${contexts.length} WebGL contexts`);
      
      // Clean up
      contexts.forEach((gl, i) => {
        const loseContext = gl.getExtension('WEBGL_lose_context');
        if (loseContext) {
          loseContext.loseContext();
        }
      });
      
      return contexts.length;
    },
    
    // Check for WebGL context loss issues
    checkContextLoss() {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (!gl) {
        utils.error('Could not create WebGL context for context loss test');
        return { supported: false };
      }
      
      const loseContext = gl.getExtension('WEBGL_lose_context');
      if (!loseContext) {
        utils.warn('WEBGL_lose_context extension not supported, cannot test context loss recovery');
        return { supported: false };
      }
      
      return { 
        supported: true,
        extension: 'WEBGL_lose_context'
      };
    },
    
    // Run all diagnostics
    runDiagnostics() {
      utils.log('Starting WebGL diagnostics...');
      
      // Check WebGL support
      const webglSupport = this.checkWebGLSupport();
      if (!webglSupport) {
        utils.error('WebGL diagnostics failed: WebGL not supported');
        return;
      }
      
      const { version, context: gl } = webglSupport;
      
      // Get detailed WebGL info
      const webglInfo = this.getWebGLInfo(gl);
      utils.log('WebGL capabilities:', webglInfo);
      
      // Check extensions
      const extensionStatus = this.checkExtensions(gl);
      utils.log('Critical extensions status:', extensionStatus);
      
      // Test shader compilation
      const shaderTest = this.testShaderCompilation(gl);
      if (shaderTest.success) {
        utils.success('Shader compilation test passed');
      } else {
        utils.error('Shader compilation test failed:', shaderTest.error);
      }
      
      // Count active contexts
      const contextCount = this.countActiveContexts();
      utils.log(`Maximum WebGL contexts: ${contextCount}`);
      
      // Check context loss
      const contextLossCheck = this.checkContextLoss();
      if (contextLossCheck.supported) {
        utils.log('Context loss recovery is supported');
      } else {
        utils.warn('Context loss recovery may not be supported');
      }
      
      // Generate recommendations
      this.generateRecommendations(webglInfo, extensionStatus, shaderTest, contextCount);
    },
    
    // Generate recommendations based on diagnostics
    generateRecommendations(webglInfo, extensionStatus, shaderTest, contextCount) {
      utils.log('--- WebGL Recommendations ---');
      
      // WebGL version recommendations
      if (webglInfo.version === 1) {
        utils.warn('Using WebGL1 - Consider using simpler shaders without #version 300 es');
        utils.warn('Replace custom shaders with built-in Three.js materials where possible');
      }
      
      // Extension recommendations
      const missingExtensions = Object.entries(extensionStatus)
        .filter(([ext, supported]) => !supported)
        .map(([ext]) => ext);
      
      if (missingExtensions.length > 0) {
        utils.warn(`Missing critical extensions: ${missingExtensions.join(', ')}`);
        utils.warn('Some advanced rendering features may not work correctly');
      }
      
      // Shader compilation recommendations
      if (!shaderTest.success) {
        utils.error('Shader compilation failed - Use built-in Three.js materials instead of custom shaders');
      }
      
      // Context count recommendations
      if (contextCount < 16) {
        utils.warn(`Limited WebGL contexts available (${contextCount}) - Reduce the number of canvases/renderers`);
      }
      
      // Memory recommendations
      utils.log('To avoid WebGL context loss:');
      utils.log('1. Dispose unused materials, textures, and geometries');
      utils.log('2. Use shared materials and geometries where possible');
      utils.log('3. Reduce texture sizes and complexity');
      utils.log('4. Consider using a single renderer for multiple scenes');
      
      utils.success('WebGL diagnostics complete');
    }
  };

  // Run diagnostics
  webglDiagnostics.runDiagnostics();
  
  // Export to global scope
  window.WebGLDiagnostics = webglDiagnostics;
  
  utils.log('WebGL diagnostics tool loaded. You can run diagnostics again with WebGLDiagnostics.runDiagnostics()');
})(); 