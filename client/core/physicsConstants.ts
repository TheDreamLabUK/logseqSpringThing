// Physics parameter ranges and defaults based on GPU-accelerated implementations
export const PHYSICS_CONSTANTS = {
    // Attraction force (cohesion between nodes)
    ATTRACTION: {
        MIN: 0.001,
        MAX: 1.0,
        DEFAULT: 0.015,  // Proven stable in Rust backend
        RECOMMENDED_RANGE: {
            MIN: 0.01,
            MAX: 0.05
        }
    },

    // Repulsion force (separation between nodes)
    REPULSION: {
        MIN: 1.0,
        MAX: 10000.0,
        DEFAULT: 1500.0,  // Proven stable in Rust backend
        RECOMMENDED_RANGE: {
            MIN: 1000.0,
            MAX: 2000.0
        }
    },

    // Spring force (edge elasticity)
    SPRING: {
        MIN: 0.001,
        MAX: 1.0,
        DEFAULT: 0.018,  // Proven stable in Rust backend
        RECOMMENDED_RANGE: {
            MIN: 0.01,
            MAX: 0.05
        }
    },

    // Damping (velocity decay)
    DAMPING: {
        MIN: 0.5,
        MAX: 0.95,
        DEFAULT: 0.88,  // Proven stable in Rust backend
        RECOMMENDED_RANGE: {
            MIN: 0.85,
            MAX: 0.92
        }
    },

    // Simulation iterations
    ITERATIONS: {
        MIN: 1,
        MAX: 1000,
        DEFAULT: 500,  // Proven stable in Rust backend
        RECOMMENDED_RANGE: {
            MIN: 200,
            MAX: 600
        }
    },

    // Maximum velocity
    MAX_VELOCITY: {
        MIN: 0.1,
        MAX: 5.0,
        DEFAULT: 2.5,  // Proven stable in Rust backend
        RECOMMENDED_RANGE: {
            MIN: 1.0,
            MAX: 3.0
        }
    },

    // Additional physics parameters
    COLLISION_RADIUS: {
        MIN: 0.1,
        MAX: 1.0,
        DEFAULT: 0.25,
        RECOMMENDED_RANGE: {
            MIN: 0.2,
            MAX: 0.4
        }
    },

    BOUNDS_SIZE: {
        MIN: 5.0,
        MAX: 50.0,
        DEFAULT: 12.0,
        RECOMMENDED_RANGE: {
            MIN: 10.0,
            MAX: 15.0
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