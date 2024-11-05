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
    pub iterations: u32,              // Range: 10-500, Default: 100
    pub repulsion_strength: f32,      // Range: 100-5000, Default: 1000.0
    pub attraction_strength: f32,     // Range: 0.01-1.0, Default: 0.01
    pub damping: f32,                 // Range: 0.1-0.9, Default: 0.8
    pub padding1: u32,                // First padding to ensure 16-byte alignment
    pub padding2: u32,                // Second padding
    pub padding3: u32,                // Third padding
    pub padding4: u32,                // Fourth padding to complete 32-byte alignment
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            iterations: 100,           // Matches settings.toml force_directed_iterations
            repulsion_strength: 1000.0,// Matches settings.toml force_directed_repulsion
            attraction_strength: 0.01, // Matches settings.toml force_directed_attraction
            damping: 0.8,             // Matches settings.toml force_directed_damping
            padding1: 0,
            padding2: 0,
            padding3: 0,
            padding4: 0,
        }
    }
}

impl SimulationParams {
    /// Creates new simulation parameters with validation
    pub fn new(iterations: u32, repulsion_strength: f32, attraction_strength: f32, damping: f32) -> Self {
        Self {
            iterations: iterations.clamp(10, 500),
            repulsion_strength: repulsion_strength.clamp(100.0, 5000.0),
            attraction_strength: attraction_strength.clamp(0.01, 1.0),
            damping: damping.clamp(0.1, 0.9),
            padding1: 0,
            padding2: 0,
            padding3: 0,
            padding4: 0,
        }
    }

    /// Updates iterations with validation
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations.clamp(10, 500);
        self
    }

    /// Updates repulsion strength with validation
    pub fn with_repulsion(mut self, repulsion: f32) -> Self {
        self.repulsion_strength = repulsion.clamp(100.0, 5000.0);
        self
    }

    /// Updates attraction strength with validation
    pub fn with_attraction(mut self, attraction: f32) -> Self {
        self.attraction_strength = attraction.clamp(0.01, 1.0);
        self
    }

    /// Updates damping with validation
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.1, 0.9);
        self
    }

    /// Creates simulation parameters from configuration settings
    pub fn from_config(config: &crate::config::VisualizationSettings) -> Self {
        Self::new(
            config.force_directed_iterations,
            config.force_directed_repulsion,
            config.force_directed_attraction,
            config.force_directed_damping,
        )
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
        assert_eq!(params.damping, 0.8);
    }

    #[test]
    fn test_simulation_params_validation() {
        let params = SimulationParams::new(5, 50.0, 0.001, 0.05);
        assert_eq!(params.iterations, 10); // Clamped to min
        assert_eq!(params.repulsion_strength, 100.0); // Clamped to min
        assert_eq!(params.attraction_strength, 0.01); // Clamped to min
        assert_eq!(params.damping, 0.1); // Clamped to min

        let params = SimulationParams::new(600, 6000.0, 2.0, 1.0);
        assert_eq!(params.iterations, 500); // Clamped to max
        assert_eq!(params.repulsion_strength, 5000.0); // Clamped to max
        assert_eq!(params.attraction_strength, 1.0); // Clamped to max
        assert_eq!(params.damping, 0.9); // Clamped to max
    }

    #[test]
    fn test_simulation_params_builder() {
        let params = SimulationParams::default()
            .with_iterations(200)
            .with_repulsion(2000.0)
            .with_attraction(0.05)
            .with_damping(0.7);

        assert_eq!(params.iterations, 200);
        assert_eq!(params.repulsion_strength, 2000.0);
        assert_eq!(params.attraction_strength, 0.05);
        assert_eq!(params.damping, 0.7);
    }
}
