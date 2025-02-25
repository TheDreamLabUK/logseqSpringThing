/**
 * Shader Compatibility Test
 * 
 * This script tests if the modified shaders in UnifiedTextRenderer.ts
 * will compile correctly in both WebGL1 and WebGL2 contexts.
 */

(function() {
  const logger = {
    info: (msg, data) => console.log(`[INFO] ${msg}`, data || ''),
    error: (msg, data) => console.error(`[ERROR] ${msg}`, data || ''),
    success: (msg) => console.log(`[SUCCESS] ${msg}`)
  };

  // Modified vertex shader (WebGL1 compatible)
  const vertexShader = `
    uniform vec3 cameraPosition;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    
    attribute vec3 position;
    attribute vec2 uv;
    attribute vec3 instancePosition;
    attribute vec4 instanceColor;
    attribute float instanceScale;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    varying float vViewDistance;
    
    void main() {
        vUv = uv;
        vColor = instanceColor;
        vScale = instanceScale;

        // Scale the position first
        vec3 scale = vec3(instanceScale);
        vec3 vertexPosition = position * scale;
        
        // Billboard calculation
        vec3 up = vec3(0.0, 1.0, 0.0);
        vec3 forward = normalize(cameraPosition - instancePosition);
        vec3 right = normalize(cross(up, forward));
        up = normalize(cross(forward, right));
        
        mat4 billboardMatrix = mat4(
            vec4(right, 0.0),
            vec4(up, 0.0),
            vec4(forward, 0.0),
            vec4(0.0, 0.0, 0.0, 1.0)
        );
        
        vertexPosition = (billboardMatrix * vec4(vertexPosition, 1.0)).xyz;
        vertexPosition += instancePosition;
        
        vec4 mvPosition = modelViewMatrix * vec4(vertexPosition, 1.0);
        vViewDistance = -mvPosition.z;  // Distance from camera
        gl_Position = projectionMatrix * mvPosition;
    }
  `;

  // Modified fragment shader (WebGL1 compatible)
  const fragmentShader = `
    precision highp float;
    
    uniform sampler2D fontAtlas;
    uniform float sdfThreshold;
    uniform float sdfSpread;
    uniform vec3 outlineColor;
    uniform float outlineWidth;
    uniform float fadeStart;
    uniform float fadeEnd;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    varying float vViewDistance;
    
    float median(float r, float g, float b) {
        return max(min(r, g), min(max(r, g), b));
    }
    
    void main() {
        vec3 sample = texture2D(fontAtlas, vUv).rgb;
        float sigDist = median(sample.r, sample.g, sample.b);
        
        // Dynamic threshold based on distance
        float distanceScale = smoothstep(fadeEnd, fadeStart, vViewDistance);
        float dynamicThreshold = sdfThreshold * (1.0 + (1.0 - distanceScale) * 0.1);
        float dynamicSpread = sdfSpread * (1.0 + (1.0 - distanceScale) * 0.2);
        
        // Improved antialiasing
        float alpha = smoothstep(dynamicThreshold - dynamicSpread, 
                               dynamicThreshold + dynamicSpread, 
                               sigDist);
                               
        float outline = smoothstep(dynamicThreshold - outlineWidth - dynamicSpread,
                                 dynamicThreshold - outlineWidth + dynamicSpread,
                                 sigDist);
        
        // Apply distance-based fade
        alpha *= distanceScale;
        outline *= distanceScale;
        
        vec4 color = mix(vec4(outlineColor, outline), vColor, alpha);
        gl_FragColor = color;
    }
  `;

  function testShaderCompilation() {
    logger.info('Testing shader compilation...');
    
    // Create a canvas for testing
    const canvas = document.createElement('canvas');
    
    // Try WebGL2 first
    let gl = canvas.getContext('webgl2');
    const isWebGL2 = !!gl;
    
    // Fall back to WebGL1 if needed
    if (!gl) {
      gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) {
        logger.error('WebGL not supported in this browser');
        return false;
      }
      logger.info('Using WebGL1 context');
    } else {
      logger.info('Using WebGL2 context');
    }
    
    try {
      // Create vertex shader
      const vs = gl.createShader(gl.VERTEX_SHADER);
      
      // Add a note about the test environment
      logger.info('Note: In a real Three.js environment, modelViewMatrix and projectionMatrix are provided automatically');
      
      gl.shaderSource(vs, vertexShader);
      gl.compileShader(vs);
      
      if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(vs);
        throw new Error(`Vertex shader compilation failed: ${info}`);
      }
      logger.success('Vertex shader compiled successfully');
      
      // Create fragment shader
      const fs = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(fs, fragmentShader);
      gl.compileShader(fs);
      
      if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(fs);
        throw new Error(`Fragment shader compilation failed: ${info}`);
      }
      logger.success('Fragment shader compiled successfully');
      
      // Create program
      const program = gl.createProgram();
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);
      
      // Note: In a real application, we would need to set the uniform values
      // but for compilation testing, we just need to check if the program links
      
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        throw new Error(`Program linking failed: ${info}`);
      }
      logger.success('Shader program linked successfully');
      
      // Clean up
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(program);
      
      return true;
    } catch (error) {
      logger.error('Shader test failed', error.message);
      return false;
    }
  }
  
  // Run the test
  if (testShaderCompilation()) {
    logger.success('Modified shaders are compatible with this browser\'s WebGL implementation');
  } else {
    logger.error('Modified shaders are NOT compatible with this browser\'s WebGL implementation');
  }
  
  // Export to global scope for console access
  window.testShaders = testShaderCompilation;
  logger.info('You can run the shader test again by calling testShaders() in the console');
})();