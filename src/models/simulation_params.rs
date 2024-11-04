use serde::{Deserialize, Serialize};

/// Enum defining different simulation computation modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SimulationMode {
    Remote,  // GPU-accelerated remote computation
    GPU,     // Local GPU computation
    Local,   // CPU-based local computation
}

/// Parameters controlling the force-directed graph layout simulation
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct SimulationParams {
    pub iterations: u32,
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub damping: f32,
    pub padding1: u32,        // First padding to ensure 16-byte alignment
    pub padding2: u32,        // Second padding
    pub padding3: u32,        // Third padding
    pub padding4: u32,        // Fourth padding to complete 32-byte alignment
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            iterations: 100,
            repulsion_strength: 1000.0,
            attraction_strength: 0.01,
            damping: 0.9,
            padding1: 0,
            padding2: 0,
            padding3: 0,
            padding4: 0,
        }
    }
}

impl SimulationParams {
    pub fn new(iterations: u32, repulsion_strength: f32, attraction_strength: f32, damping: f32) -> Self {
        Self {
            iterations,
            repulsion_strength,
            attraction_strength,
            damping,
            padding1: 0,
            padding2: 0,
            padding3: 0,
            padding4: 0,
        }
    }

    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_repulsion(mut self, repulsion: f32) -> Self {
        self.repulsion_strength = repulsion;
        self
    }

    pub fn with_attraction(mut self, attraction: f32) -> Self {
        self.attraction_strength = attraction;
        self
    }

    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }
}

// Manual implementations for required GPU traits
impl Clone for SimulationParams {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for SimulationParams {}

impl std::fmt::Debug for SimulationParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulationParams")
            .field("iterations", &self.iterations)
            .field("repulsion_strength", &self.repulsion_strength)
            .field("attraction_strength", &self.attraction_strength)
            .field("damping", &self.damping)
            .finish()
    }
}

// SAFETY: This type is #[repr(C)], contains only Pod types (u32 and f32),
// and has explicit padding for proper alignment. All bit patterns are valid.
unsafe impl bytemuck::Pod for SimulationParams {}

// SAFETY: All fields are primitives that can be safely zeroed
unsafe impl bytemuck::Zeroable for SimulationParams {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_params_default() {
        let params = SimulationParams::default();
        assert_eq!(params.iterations, 100);
        assert_eq!(params.repulsion_strength, 1000.0);
        assert_eq!(params.attraction_strength, 0.01);
        assert_eq!(params.damping, 0.9);
    }

    #[test]
    fn test_simulation_params_builder() {
        let params = SimulationParams::default()
            .with_iterations(200)
            .with_repulsion(2000.0)
            .with_attraction(0.02)
            .with_damping(0.8);

        assert_eq!(params.iterations, 200);
        assert_eq!(params.repulsion_strength, 2000.0);
        assert_eq!(params.attraction_strength, 0.02);
        assert_eq!(params.damping, 0.8);
    }

    #[test]
    fn test_simulation_params_new() {
        let params = SimulationParams::new(150, 1500.0, 0.015, 0.85);
        assert_eq!(params.iterations, 150);
        assert_eq!(params.repulsion_strength, 1500.0);
        assert_eq!(params.attraction_strength, 0.015);
        assert_eq!(params.damping, 0.85);
    }

    #[test]
    fn test_simulation_params_clone() {
        let params = SimulationParams::default();
        let cloned = params.clone();
        assert_eq!(params.iterations, cloned.iterations);
        assert_eq!(params.repulsion_strength, cloned.repulsion_strength);
        assert_eq!(params.attraction_strength, cloned.attraction_strength);
        assert_eq!(params.damping, cloned.damping);
    }
}
