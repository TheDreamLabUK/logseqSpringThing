use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationMode {
    Remote,  // GPU-accelerated remote computation
    GPU,     // Local GPU computation
    Local,   // CPU-based computation
}

impl Default for SimulationMode {
    fn default() -> Self {
        SimulationMode::Remote
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationPhase {
    Initial,    // Heavy computation for initial layout
    Dynamic,    // Lighter computation for dynamic updates
    Finalize,   // Final positioning and cleanup
}

impl Default for SimulationPhase {
    fn default() -> Self {
        SimulationPhase::Initial
    }
}

// GPU-compatible simulation parameters
#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable, Debug)]
pub struct GPUSimulationParams {
    pub iterations: u32,
    pub spring_length: f32,
    pub spring_strength: f32,
    pub repulsion: f32,
    pub attraction: f32,
    pub damping: f32,
    pub time_step: f32,
    pub padding: u32,  // For alignment
}

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SimulationParams {
    pub iterations: u32,           // Range: 1-500, Default: varies by phase
    pub spring_length: f32,        // Range: 10-500, Default: 100
    pub spring_strength: f32,      // Range: 0.1-10, Default: 0.5
    pub repulsion: f32,           // Range: 1-1000, Default: 100
    pub attraction: f32,          // Range: 0.1-10, Default: 1.0
    pub damping: f32,             // Range: 0-1, Default: 0.5
    pub time_step: f32,           // Range: 0.01-1, Default: 0.016 (60fps)
    pub phase: SimulationPhase,   // Current simulation phase
    pub mode: SimulationMode,     // Computation mode
}

impl SimulationParams {
    pub fn new() -> Self {
        Self {
            iterations: 100,
            spring_length: 100.0,
            spring_strength: 0.5,
            repulsion: 100.0,
            attraction: 1.0,
            damping: 0.5,
            time_step: 0.016,
            phase: SimulationPhase::Initial,
            mode: SimulationMode::Remote,
        }
    }

    pub fn with_phase(phase: SimulationPhase) -> Self {
        match phase {
            SimulationPhase::Initial => Self {
                iterations: 500,
                spring_length: 100.0,
                spring_strength: 1.0,
                repulsion: 200.0,
                attraction: 2.0,
                damping: 0.9,
                time_step: 0.016,
                phase,
                mode: SimulationMode::Remote,
            },
            SimulationPhase::Dynamic => Self {
                iterations: 50,
                spring_length: 100.0,
                spring_strength: 0.5,
                repulsion: 100.0,
                attraction: 1.0,
                damping: 0.5,
                time_step: 0.016,
                phase,
                mode: SimulationMode::Remote,
            },
            SimulationPhase::Finalize => Self {
                iterations: 200,
                spring_length: 100.0,
                spring_strength: 0.1,
                repulsion: 50.0,
                attraction: 0.5,
                damping: 0.95,
                time_step: 0.016,
                phase,
                mode: SimulationMode::Remote,
            },
        }
    }

    // Convert to GPU-compatible parameters
    pub fn to_gpu_params(&self) -> GPUSimulationParams {
        GPUSimulationParams {
            iterations: self.iterations,
            spring_length: self.spring_length,
            spring_strength: self.spring_strength,
            repulsion: self.repulsion,
            attraction: self.attraction,
            damping: self.damping,
            time_step: self.time_step,
            padding: 0,
        }
    }
}
