use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use cudarc::driver::sys::CUdevice_attribute_enum;

use std::io::{Error, ErrorKind};
use std::sync::Arc;
use log::{error, warn, info, debug, trace};
use crate::models::graph::GraphData;
use std::collections::HashMap;
use crate::models::simulation_params::SimulationParams;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use std::path::Path;
use std::env;
use tokio::sync::RwLock;
use std::time::Duration;
use tokio::time::sleep;

// Constants for GPU computation
const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const NODE_SIZE: u32 = std::mem::size_of::<BinaryNodeData>() as u32;
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * NODE_SIZE;

// Constants for retry mechanism
const MAX_GPU_INIT_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 500; // 500ms delay between retries

// Throttle debug output every 60 iterations (or adjust as needed)
const DEBUG_THROTTLE: u32 = 60;

// Note: CPU fallback code has been removed as we're always using GPU now

#[derive(Debug)]
pub struct GPUCompute {
    pub device: Arc<CudaDevice>,
    pub force_kernel: CudaFunction,
    pub node_data: CudaSlice<BinaryNodeData>,
    pub num_nodes: u32,
    pub node_indices: HashMap<String, usize>,
    pub simulation_params: SimulationParams,
    pub iteration_count: u32,
}

impl GPUCompute {
    /// Runs a basic GPU test.
    pub async fn test_gpu() -> Result<(), Error> {
        info!("Running GPU test");
        sleep(Duration::from_millis(500)).await;
        trace!("About to create CUDA device for testing");
        let device = Self::create_cuda_device().await?;
        trace!("Device created successfully, performing memory test");
        sleep(Duration::from_millis(500)).await;
        let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gpu_data = device.alloc_zeros::<f32>(5)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        sleep(Duration::from_millis(500)).await;
        device.dtoh_sync_copy_into(&gpu_data, &mut test_data.clone())
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        info!("GPU test successful");
        Ok(())
    }
    
    async fn create_cuda_device() -> Result<Arc<CudaDevice>, Error> {
        trace!("Starting CUDA device initialization sequence");
        if let Ok(uuid) = env::var("NVIDIA_GPU_UUID") {
            trace!("Found NVIDIA_GPU_UUID: {}", uuid);
            info!("Attempting to create CUDA device with UUID: {}", uuid);
            info!("Using GPU UUID {} via environment variables", uuid);
            if let Ok(devices) = env::var("CUDA_VISIBLE_DEVICES") {
                trace!("Found CUDA_VISIBLE_DEVICES: {}", devices);
                info!("CUDA_VISIBLE_DEVICES is set to: {}", devices);
            }
        }
        trace!("Preparing to create CUDA device with index 0");
        sleep(Duration::from_millis(500)).await;
        trace!("Checking CUDA device availability");
        sleep(Duration::from_millis(500)).await;
        trace!("Attempting CUDA device creation");
        sleep(Duration::from_millis(1000)).await;
        info!("Creating CUDA device with index 0");
        match CudaDevice::new(0) {
            Ok(device) => {
                trace!("CUDA device creation successful");
                info!("Successfully created CUDA device with index 0 (for GPU UUID: {})",
                    env::var("NVIDIA_GPU_UUID").unwrap_or_else(|_| "unknown".to_string()));
                Ok(device.into())
            },
            Err(e) => {
                trace!("CUDA device creation failed with error: {}", e);
                error!("Failed to create CUDA device with index 0: {}", e);
                Err(Error::new(ErrorKind::Other,
                    format!("Failed to create CUDA device: {}. Ensure CUDA drivers are installed and GPU is detected.", e)))
            }
        }
    }

    /// Initializes the GPUCompute instance with retry logic.
    pub async fn new(graph: &GraphData) -> Result<Arc<RwLock<Self>>, Error> {
        let num_nodes = graph.nodes.len() as u32;
        info!("Initializing GPU compute with {} nodes (with retry mechanism)", num_nodes);

        if num_nodes > MAX_NODES {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Node count exceeds limit: {}", MAX_NODES),
            ));
        }
        Self::with_retry(MAX_GPU_INIT_RETRIES, RETRY_DELAY_MS, |attempt| async move {
            Self::initialize_gpu(graph, num_nodes, attempt).await
        }).await
    }
    
    async fn test_gpu_capabilities() -> Result<(), Error> {
        trace!("Starting GPU capabilities test");
        info!("Testing CUDA capabilities");
        sleep(Duration::from_millis(300)).await;
        trace!("Checking environment variables");
        match env::var("NVIDIA_GPU_UUID") {
            Ok(uuid) => {
                trace!("Found NVIDIA_GPU_UUID");
                info!("NVIDIA_GPU_UUID is set to: {}", uuid)
            },
            Err(_) => {
                trace!("NVIDIA_GPU_UUID not found");
                warn!("NVIDIA_GPU_UUID environment variable is not set")
            }
        }
        sleep(Duration::from_millis(500)).await;
        trace!("Querying CUDA device count");
        match CudaDevice::count() {
            Ok(count) => {
                trace!("CUDA device count: {}", count);
                info!("Found {} CUDA device(s)", count);
                if count == 0 {
                    trace!("No CUDA devices found, returning error");
                    return Err(Error::new(ErrorKind::NotFound,
                        "No CUDA devices found. Ensure NVIDIA drivers are installed and working."));
                }
                trace!("GPU capabilities test completed successfully");
                Ok(())
            },
            Err(e) => {
                trace!("Failed to get CUDA device count: {}", e);
                error!("Failed to get CUDA device count: {}", e);
                Err(Error::new(ErrorKind::Other,
                    format!("Failed to get CUDA device count: {}. Check NVIDIA drivers.", e)))
            }
        }
    }
    
    fn diagnostic_cuda_info() -> Result<(), Error> {
        trace!("Starting CUDA diagnostic info collection");
        info!("Running CUDA diagnostic checks");
        trace!("Checking CUDA-related environment variables");
        info!("Checking CUDA-related environment variables:");
        for var in &["NVIDIA_GPU_UUID", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"] {
            trace!("Checking variable: {}", var);
            match env::var(var) {
                Ok(val) => {
                    trace!("Found {}: {}", var, val);
                    info!("  {}={}", var, val)
                },
                Err(_) => {
                    trace!("{} not set", var);
                    info!("  {} is not set", var)
                }
            }
        }
        trace!("Attempting to get CUDA device count");
        match CudaDevice::count() {
            Ok(count) => {
                trace!("Retrieved CUDA device count: {}", count);
                info!("CUDA device count: {}", count)
            },
            Err(e) => {
                trace!("Failed to get device count: {}", e);
                error!("Failed to get CUDA device count: {}", e)
            }
        }
        trace!("CUDA diagnostic info collection completed");
        Ok(())
    }
    
    async fn initialize_gpu(graph: &GraphData, num_nodes: u32, attempt: u32) -> Result<Arc<RwLock<Self>>, Error> {
        info!("GPU initialization attempt {}/{}", attempt + 1, MAX_GPU_INIT_RETRIES);
        match Self::test_gpu_capabilities().await {
            Ok(_) => info!("GPU capabilities check passed"),
            Err(e) => {
                warn!("GPU capabilities check failed on attempt {}: {}", attempt + 1, e);
                return Err(e);
            }
        }
        info!("Attempting to create CUDA device (attempt {}/{})", attempt + 1, MAX_GPU_INIT_RETRIES);
        let device = match Self::create_cuda_device().await {
            Ok(dev) => {
                info!("CUDA device created successfully");
                let max_threads = dev.as_ref().attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let compute_mode = dev.as_ref().attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let multiprocessor_count = dev.as_ref().attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                
                info!("GPU Device detected:");
                info!("  Max threads per MP: {}", max_threads);
                info!("  Multiprocessor count: {}", multiprocessor_count);
                info!("  Compute mode: {}", compute_mode);
                
                if max_threads < 256 {
                    let err = Error::new(ErrorKind::Other,
                        format!("GPU capability too low: {} threads per multiprocessor; minimum required is 256", max_threads));
                    warn!("GPU capability check failed: {}", err);
                    return Err(err);
                }
                dev
            },
            Err(e) => {
                error!("Failed to create CUDA device (attempt {}/{}): {}", attempt + 1, MAX_GPU_INIT_RETRIES, e);
                Self::diagnostic_cuda_info()?;
                return Err(Error::new(ErrorKind::Other,
                   format!("Failed to create CUDA device: {}. See logs for diagnostics.", e)));
            }
        };

        info!("Proceeding to load compute kernel (attempt {}/{})", attempt + 1, MAX_GPU_INIT_RETRIES);
        Self::load_compute_kernel(device, num_nodes, graph).await
    }
    
    /// Generic asynchronous retry mechanism with exponential backoff.
    async fn with_retry<F, Fut, T>(max_attempts: u32, base_delay_ms: u64, operation: F) -> Result<T, Error>
    where
        F: Fn(u32) -> Fut,
        Fut: std::future::Future<Output = Result<T, Error>>,
    {
        let mut last_error: Option<Error> = None;
        for attempt in 0..max_attempts {
            match operation(attempt).await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    let delay = base_delay_ms * (1 << attempt);
                    warn!("Operation failed (attempt {}/{}): {}. Retrying in {}ms...", 
                          attempt + 1, max_attempts, e, delay);
                    last_error = Some(e);
                    if attempt + 1 < max_attempts {
                        sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        error!("Operation failed after {} attempts", max_attempts);
        Err(last_error.unwrap_or_else(|| Error::new(ErrorKind::Other, 
            format!("All {} retry attempts failed", max_attempts))))
    }
    
    async fn load_compute_kernel(
        device: Arc<CudaDevice>, 
        num_nodes: u32, 
        graph: &GraphData
    ) -> Result<Arc<RwLock<Self>>, Error> {
        let ptx_path = "/app/src/utils/compute_forces.ptx";
        let ptx_path_obj = Path::new(ptx_path);
        if !ptx_path_obj.exists() {
            error!("PTX file does not exist at {} - required for GPU physics", ptx_path);
            return Err(Error::new(ErrorKind::NotFound, format!("PTX file not found at {}", ptx_path)));
        }
        let ptx = Ptx::from_file(ptx_path);
        info!("Successfully loaded PTX file");
        
        device.load_ptx(ptx, "compute_forces_kernel", &["compute_forces_kernel"])
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let force_kernel = device.get_func("compute_forces_kernel", "compute_forces_kernel")
            .ok_or_else(|| Error::new(ErrorKind::Other, "Function compute_forces_kernel not found"))?;
        
        info!("Allocating device memory for {} nodes", num_nodes);
        let node_data = device.alloc_zeros::<BinaryNodeData>(num_nodes as usize)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        info!("Creating GPU compute instance");
        let mut node_indices = HashMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            node_indices.insert(node.id.clone(), idx);
        }

        let mut instance = Self {
            device: Arc::clone(&device),
            force_kernel,
            node_data,
            num_nodes,
            node_indices,
            simulation_params: SimulationParams::default(),
            iteration_count: 0,
        };

        info!("Copying initial graph data to device memory");
        instance.update_graph_data(graph)?;
        info!("GPU compute initialization complete");
        Ok(Arc::new(RwLock::new(instance)))
    }

    pub fn update_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        trace!("Updating graph data for {} nodes", graph.nodes.len());
        self.node_indices.clear();
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_indices.insert(node.id.clone(), idx);
        }
        if graph.nodes.len() as u32 != self.num_nodes {
            info!("Reallocating GPU buffer for {} nodes", graph.nodes.len());
            self.node_data = self.device.alloc_zeros::<BinaryNodeData>(graph.nodes.len())
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            self.num_nodes = graph.nodes.len() as u32;
            self.iteration_count = 0;
        }
        let mut node_data = Vec::with_capacity(graph.nodes.len());
        if !graph.nodes.is_empty() {
            let sample_size = std::cmp::min(3, graph.nodes.len());
            trace!("Sample of first {} nodes before GPU transfer:", sample_size);
            for i in 0..sample_size {
                let node = &graph.nodes[i];
                trace!(
                    "Node[{}] id={}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                    i, node.id,
                    node.data.position.x, node.data.position.y, node.data.position.z,
                    node.data.velocity.x, node.data.velocity.y, node.data.velocity.z
                );
            }
        }
        for node in &graph.nodes {
            node_data.push(BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: node.data.padding,
            });
            if node.id == "0" || node.id == "1" {
                trace!(
                    "Node {} data prepared for GPU: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                    node.id,
                    node.data.position.x, node.data.position.y, node.data.position.z,
                    node.data.velocity.x, node.data.velocity.y, node.data.velocity.z
                );
            }
        }
        trace!("Copying {} nodes to GPU", graph.nodes.len());
        self.device.htod_sync_copy_into(&node_data, &mut self.node_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy node data to GPU: {}", e)))?;
        Ok(())
    }

    pub fn update_simulation_params(&mut self, params: &SimulationParams) -> Result<(), Error> {
        trace!("Updating simulation parameters: {:?}", params);
        self.simulation_params = params.clone();
        Ok(())
    }

    /// Computes forces on the GPU. To reduce log clutter from repeated messages, some logging is gated.
    pub fn compute_forces(&mut self) -> Result<(), Error> {
        // Only log detailed GPU computation info every DEBUG_THROTTLE iterations.
        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting force computation on GPU");
        }
        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };
        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Launch config: blocks={}, threads={}, shared_mem={}", blocks, BLOCK_SIZE, SHARED_MEM_SIZE);
        }
        unsafe {
            self.force_kernel.clone().launch(cfg, (
                &self.node_data,
                self.num_nodes as i32,
                self.simulation_params.spring_strength,
                self.simulation_params.damping,
                self.simulation_params.repulsion,
                self.simulation_params.time_step,
                self.simulation_params.max_repulsion_distance,
                if self.simulation_params.enable_bounds {
                    self.simulation_params.viewport_bounds
                } else {
                    f32::MAX // disable bounds
                },
                self.iteration_count as i32,
            )).map_err(|e| {
                error!("Kernel launch failed: {}", e);
                Error::new(ErrorKind::Other, e.to_string())
            })?;
        }
        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Force computation completed");
        }
        self.iteration_count += 1;
        Ok(())
    }

    pub fn get_node_data(&self) -> Result<Vec<BinaryNodeData>, Error> {
        let mut gpu_raw_data = vec![BinaryNodeData {
            position: Vec3Data::zero(),
            velocity: Vec3Data::zero(),
            mass: 0,
            flags: 0,
            padding: [0, 0],
        }; self.num_nodes as usize];
        self.device.dtoh_sync_copy_into(&self.node_data, &mut gpu_raw_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy data from GPU: {}", e)))?;
        if !gpu_raw_data.is_empty() {
            let sample_size = std::cmp::min(5, gpu_raw_data.len());
            trace!("Sample of first {} nodes after GPU calculation:", sample_size);
            for i in 0..sample_size {
                let node = &gpu_raw_data[i];
                let force_mag = (node.velocity.x * node.velocity.x +
                                 node.velocity.y * node.velocity.y +
                                 node.velocity.z * node.velocity.z).sqrt();
                trace!(
                    "Node[{}]: force_mag={:.6}, pos=[{:.3},{:.3},{:.3}], vel=[{:.6},{:.6},{:.6}]",
                    i, force_mag,
                    node.position.x, node.position.y, node.position.z,
                    node.velocity.x, node.velocity.y, node.velocity.z
                );
            }
        }
        Ok(gpu_raw_data)
    }

    /// Advances one simulation step.
    pub fn step(&mut self) -> Result<(), Error> {
        trace!("Executing physics step (iteration {})", self.iteration_count);
        self.compute_forces()?;
        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Detailed simulation status:");
            trace!("  - Iteration: {}", self.iteration_count);
            trace!("  - Node count: {}", self.num_nodes);
            trace!("  - Spring strength: {}", self.simulation_params.spring_strength);
            trace!("  - Repulsion: {}", self.simulation_params.repulsion);
            trace!("  - Damping: {}", self.simulation_params.damping);
        } else {
            trace!("Physics step complete, iteration count: {}", self.iteration_count);
        }
        Ok(())
    }
    
    /// Runs a minimal test computation on the GPU.
    pub fn test_compute(&self) -> Result<(), Error> {
        info!("Running test computation on GPU instance");
        match self.device.synchronize() {
            Ok(_) => { info!("GPU device access test passed"); },
            Err(e) => {
                error!("GPU device access test failed: {}", e);
                return Err(Error::new(ErrorKind::Other, format!("GPU device access test failed: {}", e)));
            }
        }
        info!("GPU test computation successful");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_compute_initialization() {
        info!("Running GPU compute initialization test");
        let graph = GraphData::default();
        let gpu_compute = GPUCompute::new(&graph).await;
        assert!(gpu_compute.is_ok());
    }

    #[tokio::test]
    async fn test_node_data_transfer() {
        info!("Running node data transfer test");
        let mut graph = GraphData::default();
        let gpu_compute = GPUCompute::new(&graph).await.unwrap();
        let gpu_compute = Arc::try_unwrap(gpu_compute).unwrap().into_inner();
        let node_data = gpu_compute.get_node_data().unwrap();
        assert_eq!(node_data.len(), graph.nodes.len());
    }

    #[test]
    fn test_node_data_memory_layout() {
        info!("Checking BinaryNodeData memory layout");
        use std::mem::size_of;
        assert_eq!(size_of::<BinaryNodeData>(), 28);
    }
}
