// Physics parameter ranges and defaults based on GPU-accelerated implementations
export const PHYSICS_CONSTANTS = {
    // Attraction force (cohesion between nodes)
    ATTRACTION: {
        MIN: 0.001,
        MAX: 0.2,
        DEFAULT: 0.1,  // Balanced for CUDA implementation
        RECOMMENDED_RANGE: {
            MIN: 0.05,
            MAX: 0.15
        }
    },

    // Repulsion force (separation between nodes)
    REPULSION: {
        MIN: 0.1,
        MAX: 2.0,
        DEFAULT: 0.3,  // Balanced for CUDA implementation
        RECOMMENDED_RANGE: {
            MIN: 0.2,
            MAX: 0.5
        }
    },

    // Spring force (edge elasticity)
    SPRING: {
        MIN: 0.001,
        MAX: 0.2,
        DEFAULT: 0.15,  // Balanced for CUDA implementation
        RECOMMENDED_RANGE: {
            MIN: 0.1,
            MAX: 0.2
        }
    },

    // Damping (velocity decay)
    DAMPING: {
        MIN: 0.5,
        MAX: 0.95,
        DEFAULT: 0.85,  // Balanced for CUDA implementation
        RECOMMENDED_RANGE: {
            MIN: 0.85,
            MAX: 0.92
        }
    },

    // Simulation iterations
    ITERATIONS: {
        MIN: 1,
        MAX: 1000,
        DEFAULT: 100,  // Balanced for performance
        RECOMMENDED_RANGE: {
            MIN: 50,
            MAX: 200
        }
    },

    // Maximum velocity
    MAX_VELOCITY: {
        MIN: 0.1,
        MAX: 5.0,
        DEFAULT: 0.4,  // In meters per second
        RECOMMENDED_RANGE: {
            MIN: 0.2,
            MAX: 0.6
        }
    },

    // Additional physics parameters
    COLLISION_RADIUS: {
        MIN: 0.1,
        MAX: 1.0,
        DEFAULT: 0.1,  // 10cm collision radius
        RECOMMENDED_RANGE: {
            MIN: 0.05,
            MAX: 0.15
        }
    },

    BOUNDS_SIZE: {
        MIN: 0.1,
        MAX: 2.0,
        DEFAULT: 0.5,  // 50cm bounds
        RECOMMENDED_RANGE: {
            MIN: 0.3,
            MAX: 0.7
        }
    }
};

// Helper types for physics parameters
export type PhysicsParameter = keyof typeof PHYSICS_CONSTANTS;
export type PhysicsRange = {
    MIN: number;
    MAX: number;
    DEFAULT: number;
    RECOMMENDED_RANGE: {
        MIN: number;
        MAX: number;
    };
};

// Helper functions for physics parameters
export const isWithinPhysicsRange = (param: PhysicsParameter, value: number): boolean => {
    const range = PHYSICS_CONSTANTS[param];
    return value >= range.MIN && value <= range.MAX;
};

export const isWithinRecommendedRange = (param: PhysicsParameter, value: number): boolean => {
    const range = PHYSICS_CONSTANTS[param].RECOMMENDED_RANGE;
    return value >= range.MIN && value <= range.MAX;
};

export const getPhysicsRange = (param: PhysicsParameter): PhysicsRange => {
    return PHYSICS_CONSTANTS[param];
};

export const getDefaultPhysicsValue = (param: PhysicsParameter): number => {
    return PHYSICS_CONSTANTS[param].DEFAULT;
};