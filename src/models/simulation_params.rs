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
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct SimulationParams {
    pub iterations: u32,           // Range: 1-500, Default: varies by phase
    pub spring_strength: f32,      // Range: 0.001-1.0, Default: 0.01 (reduced for stability)
    pub damping: f32,             // Range: 0.5-0.95, Default: 0.8
    pub is_initial_layout: u32,   // 1 for initial layout, 0 for interactive
    pub time_step: f32,           // Animation time step (0.1-1.0)
    pub padding2: u32,            // Padding for alignment
    pub padding3: u32,            // Padding for alignment
    pub padding4: u32,            // Complete 32-byte alignment
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            iterations: 5,          // Default to interactive mode iterations
            spring_strength: 0.01,  // Reduced default strength for better stability
            damping: 0.8,          // Balanced damping for smooth movement
            is_initial_layout: 0,   // Default to interactive mode
            time_step: 0.5,        // Reduced time step for smoother animation
            padding2: 0,
            padding3: 0,
            padding4: 0,
        }
    }
}

impl SimulationParams {
    /// Creates new simulation parameters with validation
    pub fn new(iterations: u32, spring_strength: f32, damping: f32, is_initial: bool) -> Self {
        Self {
            iterations: if is_initial { 
                iterations.clamp(200, 500) // More iterations for initial layout
            } else {
                iterations.clamp(1, 10)    // Fewer iterations for interactive updates
            },
            spring_strength: spring_strength.clamp(0.001, 1.0), // Reduced range for better control
            damping: damping.clamp(0.5, 0.95), // Increased minimum damping for stability
            is_initial_layout: if is_initial { 1 } else { 0 },
            time_step: 0.5, // Default to moderate time step
            padding2: 0,
            padding3: 0,
            padding4: 0,
        }
    }

    /// Creates simulation parameters from configuration settings
    pub fn from_config(config: &crate::config::VisualizationSettings, phase: SimulationPhase) -> Self {
        let is_initial = matches!(phase, SimulationPhase::Initial);
        Self::new(
            if is_initial { config.force_directed_iterations } else { 5 },
            config.force_directed_spring,
            config.force_directed_damping,
            is_initial
        )
    }

    /// Updates iterations with phase-appropriate validation
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = if self.is_initial_layout == 1 {
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
            .field("spring_strength", &self.spring_strength)
            .field("damping", &self.damping)
            .field("is_initial_layout", &self.is_initial_layout)
            .field("time_step", &self.time_step)
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
        assert_eq!(params.iterations, 5);
        assert_eq!(params.spring_strength, 0.01);
        assert_eq!(params.damping, 0.8);
        assert_eq!(params.is_initial_layout, 0);
        assert_eq!(params.time_step, 0.5);
    }

    #[test]
    fn test_simulation_params_validation() {
        // Test interactive mode limits
        let params = SimulationParams::new(20, 0.0001, 0.3, false);
        assert_eq!(params.iterations, 10); // Clamped to interactive max
        assert_eq!(params.spring_strength, 0.001); // Clamped to min
        assert_eq!(params.damping, 0.5); // Clamped to min
        assert_eq!(params.is_initial_layout, 0);

        // Test initial layout mode limits
        let params = SimulationParams::new(600, 2.0, 1.0, true);
        assert_eq!(params.iterations, 500); // Clamped to max
        assert_eq!(params.spring_strength, 1.0); // Clamped to max
        assert_eq!(params.damping, 0.95); // Clamped to max
        assert_eq!(params.is_initial_layout, 1);
    }

    #[test]
    fn test_simulation_params_builder() {
        let params = SimulationParams::default()
            .with_iterations(200)
            .with_spring_strength(0.5)
            .with_damping(0.7)
            .with_time_step(0.8);

        assert_eq!(params.iterations, 10); // Clamped to interactive max
        assert_eq!(params.spring_strength, 0.5);
        assert_eq!(params.damping, 0.7);
        assert_eq!(params.time_step, 0.8);
    }
}
