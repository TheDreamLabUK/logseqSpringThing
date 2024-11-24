// public/js/gpuUtils.js

/**
 * GPU/WebGL utilities for rendering
 * Note: Force-directed calculations are now handled server-side
 */

/**
 * Check if GPU/WebGL is available for rendering
 * @returns {boolean} True if GPU rendering is available
 */
export function isGPUAvailable() {
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        const isAvailable = !!gl;
        
        if (!gl) {
            console.warn('WebGL not available, rendering may be limited');
            return false;
        }

        // Check if it's WebGL 2
        if (gl instanceof WebGL2RenderingContext) {
            console.log('WebGL 2 available');
        } else {
            console.log('WebGL 1 available');
        }

        return isAvailable;
    } catch (error) {
        console.error('Error checking GPU availability:', error);
        return false;
    }
}

/**
 * Initialize GPU/WebGL context
 * @returns {object|null} GPU context if available
 */
export function initGPU() {
    if (isGPUAvailable()) {
        return {
            initialized: true,
            webgl2: typeof WebGL2RenderingContext !== 'undefined' && 
                   document.createElement('canvas').getContext('webgl2') instanceof WebGL2RenderingContext
        };
    }
    return null;
}

/**
 * Apply position updates received from server
 * @param {Float32Array} buffer - Binary position data from server
 * @returns {object} Processed position data
 */
export function processPositionUpdate(buffer) {
    try {
        const dataView = new Float32Array(buffer);
        const isInitialLayout = dataView[0] === 1.0;
        const positions = [];
        
        // Skip header (first float32)
        for (let i = 1; i < dataView.length; i += 6) {
            if (i + 5 < dataView.length) {
                positions.push({
                    x: dataView[i],
                    y: dataView[i + 1],
                    z: dataView[i + 2],
                    vx: dataView[i + 3],
                    vy: dataView[i + 4],
                    vz: dataView[i + 5]
                });
            }
        }

        return {
            isInitialLayout,
            positions
        };
    } catch (error) {
        console.error('Error processing position update:', error);
        return null;
    }
}
