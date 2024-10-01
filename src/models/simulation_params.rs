// models/simulation_params.rs

use bytemuck::{Pod, Zeroable};

/// Parameters for the force-directed graph simulation.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SimulationParams {
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub damping: f32,
    pub delta_time: f32,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            repulsion_strength: 30.0,  // Increase this to push nodes further apart
            attraction_strength: 0.005, // Decrease this to reduce the attraction force
            damping: 0.95,              // Keep this unchanged to prevent excessive oscillation
            delta_time: 0.016,          // Standard frame rate delta time, can be kept the same
        }
    }
}
