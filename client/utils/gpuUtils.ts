import type { Vector3 } from 'three';

export interface GPUContext {
  initialized: boolean;
  webgl2: boolean;
}

export interface PositionUpdate {
  isInitialLayout: boolean;
  positions: NodePosition[];
}

export interface NodePosition {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

/**
 * Check if GPU/WebGL is available for rendering
 * @returns Promise that resolves to true if GPU rendering is available
 */
export async function isGPUAvailable(): Promise<boolean> {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || 
               canvas.getContext('webgl') || 
               canvas.getContext('experimental-webgl');
    
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

    return true;
  } catch (error) {
    console.error('Error checking GPU availability:', error);
    return false;
  }
}

/**
 * Initialize GPU/WebGL context
 * @returns Promise that resolves to GPU context if available
 */
export async function initGPU(): Promise<GPUContext | null> {
  const available = await isGPUAvailable();
  if (available) {
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
 * @param buffer - Binary position data from server
 * @returns Processed position data
 */
export function processPositionUpdate(buffer: ArrayBuffer): PositionUpdate | null {
  try {
    const dataView = new Float32Array(buffer);
    const isInitialLayout = dataView[0] === 1.0;
    const positions: NodePosition[] = [];
    
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

/**
 * Convert NodePosition to Vector3
 * @param position - Node position data
 * @returns THREE.Vector3 position
 */
export function positionToVector3(position: NodePosition): Vector3 {
  return {
    x: position.x,
    y: position.y,
    z: position.z
  } as Vector3;
}

/**
 * Convert Vector3 to NodePosition
 * @param vector - THREE.Vector3 position
 * @returns Node position data
 */
export function vector3ToPosition(vector: Vector3): NodePosition {
  return {
    x: vector.x,
    y: vector.y,
    z: vector.z,
    vx: 0,
    vy: 0,
    vz: 0
  };
}
