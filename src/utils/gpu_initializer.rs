use std::sync::Arc;
use tokio::sync::RwLock;
use log::{error, info, debug};
use std::io::{Error, ErrorKind};

use crate::{
    models::graph::GraphData,
    models::simulation_params::SimulationParams,
    utils::gpu_compute::GPUCompute,
    config::Settings,
};

/// Initialize GPU compute with optimized settings for NVIDIA GPUs
pub async fn initialize_gpu(
    settings: Arc<RwLock<Settings>>,
    graph_data: &GraphData,
) -> Result<Option<Arc<RwLock<GPUCompute>>>, Error> {
    info!("Initializing GPU compute system...");

    if !cfg!(feature = "gpu") {
        info!("GPU feature disabled, using CPU computations");
        return Ok(None);
    }

    // Read GPU settings from settings
    let settings = settings.read().await;
    let gpu_enabled = settings.system.debug.enabled;
    if !gpu_enabled {
        info!("GPU disabled in settings, using CPU computations");
        return Ok(None);
    }

    // Initialize with optimal parameters for NVIDIA GPUs
    let params = SimulationParams {
        iterations: 1,               // One iteration per frame for real-time updates
        spring_strength: 5.0,        // Strong spring force for tight clustering
        repulsion: 0.05,            // Minimal repulsion to prevent node overlap
        damping: 0.98,              // High damping for stability
        max_repulsion_distance: 0.1, // Small repulsion range for local interactions
        viewport_bounds: 1.0,        // Normalized bounds
        mass_scale: 1.0,            // Default mass scaling
        boundary_damping: 0.95,      // Strong boundary damping
        enable_bounds: true,         // Enable bounds for contained layout
        time_step: 0.01,            // Small timestep for numerical stability
        phase: crate::models::simulation_params::SimulationPhase::Dynamic,
        mode: crate::models::simulation_params::SimulationMode::Remote,
    };

    match GPUCompute::new(graph_data).await {
        Ok(gpu) => {
            // Unwrap the GPU compute instance from the Arc<RwLock<>>
            let mut gpu_instance = match Arc::try_unwrap(gpu) {
                Ok(lock) => lock.into_inner(),
                Err(_) => return Err(Error::new(ErrorKind::Other, "Failed to get exclusive access to GPU compute")),
            };
            
            // Update simulation parameters
            if let Err(e) = gpu_instance.update_simulation_params(&params) {
                error!("Failed to set simulation parameters: {}", e);
                return Err(Error::new(ErrorKind::Other, e.to_string()));
            }

            // Verify GPU memory allocation
            if let Err(e) = gpu_instance.update_graph_data(graph_data) {
                error!("Failed to update graph data: {}", e);
                return Err(Error::new(ErrorKind::Other, e.to_string()));
            }

            info!("GPU initialization successful - Ready for computation");
            // Create new Arc<RwLock<>> with the configured GPU instance
            Ok(Some(Arc::new(RwLock::new(gpu_instance))))
        }
        
        Err(e) => {
            error!("Failed to initialize GPU: {}. Falling back to CPU computations.", e);
            debug!("GPU initialization error details: {:?}", e);
            Ok(None)
        }
    }
}

