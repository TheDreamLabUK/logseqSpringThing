use serde::{Deserialize, Serialize};

/// Enum defining different simulation computation modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SimulationMode {
    Remote,  // GPU-accelerated remote computation
    GPU,     // Local GPU computation
    Local,   // CPU-based local computation
}

/// Enum defining the simulation phase
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SimulationPhase {
    Initial,    // Heavy computation for initial layout
    Interactive // Light computation for real-time updates
}

/// Parameters controlling the force-directed graph layout simulation
#[derive(Serialize, Deserialize, Clone, Debug)]
#[repr(C)]
pub struct SimulationParams {
    pub iterations: u32,           // Range: 1-500, Default: varies by phase
    pub spring_strength: f32,      // Range: 0.001-1.0, Default: 0.01
    pub repulsion_strength: f32,   // Range: 1.0-10000.0, Default: 1000.0
    pub attraction_strength: f32,  // Range: 0.001-1.0, Default: 0.01
    pub damping: f32,             // Range: 0.5-0.95, Default: 0.8
    pub is_initial_layout: bool,   // true for initial layout, false for interactive
    pub time_step: f32,           // Animation time step (0.1-1.0)
    pub padding: u32,             // Complete 32-byte alignment
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            iterations: 5,
            spring_strength: 0.01,
            repulsion_strength: 1000.0,
            attraction_strength: 0.01,
            damping: 0.8,
            is_initial_layout: false,
            time_step: 0.5,
            padding: 0,
        }
    }
}

impl SimulationParams {
    /// Creates new simulation parameters with validation
    pub fn new(
        iterations: u32,
        spring_strength: f32,
        repulsion_strength: f32,
        attraction_strength: f32,
        damping: f32,
        is_initial: bool
    ) -> Self {
        Self {
            iterations: if is_initial { 
                iterations.clamp(200, 500) // More iterations for initial layout
            } else {
                iterations.clamp(1, 10)    // Fewer iterations for interactive updates
            },
            spring_strength: spring_strength.clamp(0.001, 1.0),
            repulsion_strength: repulsion_strength.clamp(1.0, 10000.0),
            attraction_strength: attraction_strength.clamp(0.001, 1.0),
            damping: damping.clamp(0.5, 0.95),
            is_initial_layout: is_initial,
            time_step: 0.5,
            padding: 0,
        }
    }

    /// Creates simulation parameters from configuration settings
    pub fn from_config(config: &crate::config::VisualizationSettings, phase: SimulationPhase) -> Self {
        let is_initial = matches!(phase, SimulationPhase::Initial);
        Self::new(
            if is_initial { config.force_directed_iterations } else { 5 },
            config.force_directed_spring,
            config.force_directed_repulsion,
            config.force_directed_attraction,
            config.force_directed_damping,
            is_initial
        )
    }

    /// Updates iterations with phase-appropriate validation
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = if self.is_initial_layout {
            iterations.clamp(200, 500)
        } else {
            iterations.clamp(1, 10)
        };
        self
    }

    /// Updates spring strength with validation
    pub fn with_spring_strength(mut self, strength: f32) -> Self {
        self.spring_strength = strength.clamp(0.001, 1.0);
        self
    }

    /// Updates repulsion strength with validation
    pub fn with_repulsion_strength(mut self, strength: f32) -> Self {
        self.repulsion_strength = strength.clamp(1.0, 10000.0);
        self
    }

    /// Updates attraction strength with validation
    pub fn with_attraction_strength(mut self, strength: f32) -> Self {
        self.attraction_strength = strength.clamp(0.001, 1.0);
        self
    }

    /// Updates damping with validation
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.5, 0.95);
        self
    }

    /// Updates time step with validation
    pub fn with_time_step(mut self, time_step: f32) -> Self {
        self.time_step = time_step.clamp(0.1, 1.0);
        self
    }
}

// Manual implementations for required GPU traits
impl Copy for SimulationParams {}

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
        assert_eq!(params.iterations, 5);
        assert_eq!(params.spring_strength, 0.01);
        assert_eq!(params.repulsion_strength, 1000.0);
        assert_eq!(params.attraction_strength, 0.01);
        assert_eq!(params.damping, 0.8);
        assert!(!params.is_initial_layout);
    }

    #[test]
    fn test_simulation_params_validation() {
        let params = SimulationParams::new(20, 0.0001, 0.5, 0.0001, 0.3, false);
        assert_eq!(params.iterations, 10); // Clamped to interactive max
        assert_eq!(params.spring_strength, 0.001); // Clamped to min
        assert_eq!(params.repulsion_strength, 1.0); // Clamped to min
        assert_eq!(params.attraction_strength, 0.001); // Clamped to min
        assert_eq!(params.damping, 0.5); // Clamped to min
        assert!(!params.is_initial_layout);

        let params = SimulationParams::new(600, 2.0, 20000.0, 2.0, 1.0, true);
        assert_eq!(params.iterations, 500); // Clamped to max
        assert_eq!(params.spring_strength, 1.0); // Clamped to max
        assert_eq!(params.repulsion_strength, 10000.0); // Clamped to max
        assert_eq!(params.attraction_strength, 1.0); // Clamped to max
        assert_eq!(params.damping, 0.95); // Clamped to max
        assert!(params.is_initial_layout);
    }

    #[test]
    fn test_simulation_params_builder() {
        let params = SimulationParams::default()
            .with_iterations(200)
            .with_spring_strength(0.5)
            .with_repulsion_strength(5000.0)
            .with_attraction_strength(0.05)
            .with_damping(0.7)
            .with_time_step(0.8);

        assert_eq!(params.iterations, 10); // Clamped to interactive max
        assert_eq!(params.spring_strength, 0.5);
        assert_eq!(params.repulsion_strength, 5000.0);
        assert_eq!(params.attraction_strength, 0.05);
        assert_eq!(params.damping, 0.7);
        assert_eq!(params.time_step, 0.8);
    }
}
