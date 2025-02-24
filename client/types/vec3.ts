import { Vector3 } from 'three';

/**
 * Vec3 interface that mirrors the Rust Vec3Data struct
 * Used for consistent vector representation across the stack
 */
export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

export const Vec3 = {
    /**
     * Create a new Vec3
     */
    new: (x: number, y: number, z: number): Vec3 => ({
        x, y, z
    }),

    /**
     * Create a Vec3 with all components set to zero
     */
    zero: (): Vec3 => ({
        x: 0, y: 0, z: 0
    }),

    /**
     * Convert from Three.js Vector3
     */
    fromVector3: (v: Vector3): Vec3 => ({
        x: v.x,
        y: v.y,
        z: v.z
    }),

    /**
     * Convert to Three.js Vector3
     */
    toVector3: (v: Vec3): Vector3 => new Vector3(v.x, v.y, v.z)
};