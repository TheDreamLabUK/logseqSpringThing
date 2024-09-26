// public/js/gpuUtils.js

/**
 * Checks if the GPU is available for acceleration.
 * @returns {boolean} True if GPU is available, false otherwise.
 */
export function isGPUAvailable() {
  try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return !!gl;
  } catch (e) {
      console.error('GPU availability check failed:', e);
      return false;
  }
}

/**
* Initializes GPU computation utilities.
* @returns {object} An object containing GPU-related methods.
*/
export function initGPU() {
  // Placeholder for GPU computation initialization
  console.log('Initializing GPU utilities.');

  // Implement GPU-related initializations here, such as setting up compute shaders
  // For this example, we'll return an empty object
  return {
      compute: (data) => {
          // Implement GPU computation logic here
          console.log('Performing GPU computation with data:', data);
      }
  };
}

/**
* Performs computations on the GPU.
* @param {object} gpu - The GPU utilities object.
* @param {object} data - The data to compute.
* @returns {boolean} True if computation was successful, false otherwise.
*/
export function computeOnGPU(gpu, data) {
  if (gpu && typeof gpu.compute === 'function') {
      gpu.compute(data);
      return true;
  } else {
      console.warn('GPU compute function is not available.');
      return false;
  }
}
