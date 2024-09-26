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
            repulsion_strength: 100.0,
            attraction_strength: 0.01,
            damping: 0.95,
            delta_time: 0.016,
        }
    }
}