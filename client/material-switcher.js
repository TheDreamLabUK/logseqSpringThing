/**
 * Material Switcher
 * 
 * This script can be run in the browser console to switch from custom shader materials
 * to simplified materials at runtime, helping to resolve WebGL context issues.
 * 
 * Usage:
 * 1. Open the browser console (F12 or Ctrl+Shift+I)
 * 2. Copy and paste this entire script into the console
 * 3. Press Enter to run the script
 * 4. The script will automatically replace problematic materials
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

  // Material switcher
  const materialSwitcher = {
    // Replace custom shader materials with simplified materials
    replaceCustomShaderMaterials() {
      utils.log('Searching for custom shader materials to replace...');
      
      // Get the visualization instance
      const viz = window.visualization;
      if (!viz) {
        utils.error('Visualization not found. Make sure the application is initialized.');
        return;
      }
      
      // Get the scene manager
      const sceneManager = window.SceneManager?.getInstance();
      if (!sceneManager) {
        utils.error('SceneManager not found. Make sure the application is initialized.');
        return;
      }
      
      // Get the scene
      const scene = sceneManager.getScene();
      if (!scene) {
        utils.error('Scene not found. Make sure the application is initialized.');
        return;
      }
      
      // Count of replaced materials
      let replacedCount = 0;
      
      // Traverse the scene and replace materials
      scene.traverse(object => {
        if (!object.material) return;
        
        // Handle arrays of materials
        if (Array.isArray(object.material)) {
          for (let i = 0; i < object.material.length; i++) {
            const material = object.material[i];
            if (this.isCustomShaderMaterial(material)) {
              object.material[i] = this.createSimplifiedMaterial(material);
              replacedCount++;
            }
          }
        } 
        // Handle single material
        else if (this.isCustomShaderMaterial(object.material)) {
          object.material = this.createSimplifiedMaterial(object.material);
          replacedCount++;
        }
      });
      
      utils.success(`Replaced ${replacedCount} custom shader materials with simplified materials`);
      
      // Force a render update
      sceneManager.render();
    },
    
    // Check if a material is a custom shader material
    isCustomShaderMaterial(material) {
      // Check for our custom shader materials
      if (!material) return false;
      
      // Check for ShaderMaterial or custom material types
      return (
        material.type === 'ShaderMaterial' || 
        material.constructor.name === 'HologramShaderMaterial' ||
        material.constructor.name === 'EdgeShaderMaterial' ||
        (material.uniforms && material.vertexShader && material.fragmentShader)
      );
    },
    
    // Create a simplified material based on the original material
    createSimplifiedMaterial(material) {
      // Default values
      const color = material.color || material.uniforms?.color?.value || new THREE.Color(0x00ff00);
      const opacity = material.opacity || material.uniforms?.opacity?.value || 0.7;
      const wireframe = material.wireframe || material.uniforms?.isEdgeOnly?.value || false;
      
      // Create a basic material
      const simplifiedMaterial = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
        wireframe: wireframe,
        side: THREE.DoubleSide
      });
      
      utils.log('Created simplified material', {
        originalType: material.type || material.constructor.name,
        color: color,
        opacity: opacity,
        wireframe: wireframe
      });
      
      return simplifiedMaterial;
    },
    
    // Replace all materials in the scene
    replaceAllMaterials() {
      utils.log('Replacing all materials with simplified versions...');
      
      // Get the scene manager
      const sceneManager = window.SceneManager?.getInstance();
      if (!sceneManager) {
        utils.error('SceneManager not found. Make sure the application is initialized.');
        return;
      }
      
      // Get the scene
      const scene = sceneManager.getScene();
      if (!scene) {
        utils.error('Scene not found. Make sure the application is initialized.');
        return;
      }
      
      // Count of replaced materials
      let replacedCount = 0;
      
      // Traverse the scene and replace all materials
      scene.traverse(object => {
        if (!object.material) return;
        
        // Handle arrays of materials
        if (Array.isArray(object.material)) {
          for (let i = 0; i < object.material.length; i++) {
            const material = object.material[i];
            object.material[i] = this.createSimplifiedMaterial(material);
            replacedCount++;
          }
        } 
        // Handle single material
        else {
          object.material = this.createSimplifiedMaterial(object.material);
          replacedCount++;
        }
      });
      
      utils.success(`Replaced ${replacedCount} materials with simplified materials`);
      
      // Force a render update
      sceneManager.render();
    },
    
    // Reduce WebGL context usage
    optimizeWebGLContextUsage() {
      utils.log('Optimizing WebGL context usage...');
      
      // Get the scene manager
      const sceneManager = window.SceneManager?.getInstance();
      if (!sceneManager) {
        utils.error('SceneManager not found. Make sure the application is initialized.');
        return;
      }
      
      // Get the renderer
      const renderer = sceneManager.getRenderer();
      if (!renderer) {
        utils.error('Renderer not found. Make sure the application is initialized.');
        return;
      }
      
      // Optimize renderer settings
      renderer.shadowMap.enabled = false;
      renderer.shadowMap.autoUpdate = false;
      renderer.shadowMap.needsUpdate = false;
      
      // Disable automatic clearing
      renderer.autoClear = false;
      renderer.autoClearColor = false;
      renderer.autoClearDepth = false;
      renderer.autoClearStencil = false;
      
      // Reduce precision if possible
      try {
        renderer.getContext().getShaderPrecisionFormat(
          renderer.getContext().FRAGMENT_SHADER,
          renderer.getContext().HIGH_FLOAT
        );
        utils.log('Using HIGH_FLOAT precision');
      } catch (e) {
        utils.warn('Failed to set shader precision', e);
      }
      
      utils.success('WebGL context usage optimized');
    }
  };

  // Run the material switcher
  try {
    materialSwitcher.replaceCustomShaderMaterials();
    materialSwitcher.optimizeWebGLContextUsage();
    utils.success('Material switcher completed successfully');
  } catch (error) {
    utils.error('Error running material switcher', error);
  }
  
  // Export to global scope
  window.MaterialSwitcher = materialSwitcher;
  
  utils.log('Material switcher loaded. You can run it again with MaterialSwitcher.replaceCustomShaderMaterials()');
  utils.log('To replace all materials, run MaterialSwitcher.replaceAllMaterials()');
})(); 