// Interface for UI control settings
export interface SettingControl {
    label: string;
    type: 'slider' | 'toggle' | 'color' | 'select' | 'number' | 'text';
    options?: string[];
    min?: number;
    max?: number;
    step?: number;
    tooltip?: string;
}

export const settingsMap: Record<string, Record<string, SettingControl | Record<string, SettingControl>>> = {
    visualization: {
        physics: {
            enabled: { 
                label: 'Enable Physics', 
                type: 'toggle', 
                tooltip: 'Enable/disable the physics simulation.' 
            },
            iterations: { 
                label: 'Iterations', 
                type: 'number', 
                min: 1, 
                max: 500, 
                tooltip: 'Number of physics iterations per frame. Higher values increase accuracy but impact performance.' 
            },
            attractionStrength: { 
                label: 'Attraction', 
                type: 'slider', 
                min: 0.1, 
                max: 10, 
                step: 0.1, 
                tooltip: 'Strength of attraction between connected nodes. Higher values pull connected nodes closer together.' 
            },
            repulsionStrength: { 
                label: 'Repulsion', 
                type: 'slider', 
                min: 1, 
                max: 2000, 
                step: 1, 
                tooltip: 'Strength of repulsion between all nodes. Higher values push nodes further apart.' 
            },
            repulsionDistance: { 
                label: 'Repulsion Range', 
                type: 'slider', 
                min: 100, 
                max: 2000, 
                step: 100, 
                tooltip: 'Maximum distance for repulsion forces. Nodes beyond this range do not repel each other.' 
            },
            springStrength: { 
                label: 'Spring Strength', 
                type: 'slider', 
                min: 0.1, 
                max: 10, 
                step: 0.1, 
                tooltip: 'Strength of spring forces between connected nodes. Affects the equilibrium distance.' 
            },
            damping: { 
                label: 'Damping', 
                type: 'slider', 
                min: 0, 
                max: 1, 
                step: 0.01, 
                tooltip: 'Velocity damping factor. Higher values make the simulation more stable but less dynamic.' 
            },
            boundaryDamping: { 
                label: 'Boundary Damping', 
                type: 'slider', 
                min: 0.5, 
                max: 1, 
                step: 0.05, 
                tooltip: 'Additional damping applied near boundaries. Higher values prevent boundary oscillation.' 
            },
            massScale: { 
                label: 'Mass Scale', 
                type: 'slider', 
                min: 0.1, 
                max: 5, 
                step: 0.1, 
                tooltip: 'Scaling factor for node masses. Affects force calculations and node movement speed.' 
            },
            maxVelocity: { 
                label: 'Max Velocity', 
                type: 'slider', 
                min: 0.1, 
                max: 10, 
                step: 0.1, 
                tooltip: 'Maximum velocity limit for nodes. Higher values allow faster movement but may cause instability.' 
            },
            collisionRadius: { 
                label: 'Collision Radius', 
                type: 'slider', 
                min: 0.1, 
                max: 100, 
                step: 0.1, 
                tooltip: 'Radius for node collision detection. Affects minimum distance between nodes.' 
            },
            enableBounds: { 
                label: 'Enable Bounds', 
                type: 'toggle', 
                tooltip: 'Enable bounding box to contain nodes within a specific volume.' 
            },
            boundsSize: { 
                label: 'Bounds Size', 
                type: 'slider', 
                min: 100, 
                max: 5000, 
                step: 100, 
                tooltip: 'Size of the bounding box that contains the nodes. Larger values allow more spread.' 
            }
        }
    }
};