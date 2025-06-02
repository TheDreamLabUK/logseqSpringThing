use actix::prelude::*;
use log::{error, warn, info, trace};
use std::io::{Error, ErrorKind};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use cudarc::driver::sys::CUdevice_attribute_enum;

use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use crate::actors::messages::*;
use std::path::Path;
use std::env;
use std::sync::Arc;

// Constants for GPU computation
const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const NODE_SIZE: u32 = std::mem::size_of::<BinaryNodeData>() as u32;
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * NODE_SIZE;
const DEBUG_THROTTLE: u32 = 60;

// Constants for retry mechanism
const MAX_GPU_INIT_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 500;

// Constants for error watchdog (Insight 1.9)
const MAX_GPU_FAILURES: u32 = 5;
const FAILURE_RESET_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug)]
pub struct GPUComputeActor {
    device: Option<Arc<CudaDevice>>,
    force_kernel: Option<CudaFunction>,
    node_data: Option<CudaSlice<BinaryNodeData>>,
    num_nodes: u32,
    node_indices: HashMap<u32, usize>, // Changed from String to u32 for node IDs
    simulation_params: SimulationParams,
    iteration_count: u32,
    
    // Error watchdog state (Insight 1.9)
    gpu_failure_count: u32,
    last_failure_reset: Instant,
    cpu_fallback_active: bool,
}

impl GPUComputeActor {
    pub fn new() -> Self {
        Self {
            device: None,
            force_kernel: None,
            node_data: None,
            num_nodes: 0,
            node_indices: HashMap::new(),
            simulation_params: SimulationParams::default(),
            iteration_count: 0,
            gpu_failure_count: 0,
            last_failure_reset: Instant::now(),
            cpu_fallback_active: false,
        }
    }

    async fn initialize_gpu(&mut self, graph: &GraphData) -> Result<(), Error> {
        let num_nodes = graph.nodes.len() as u32;
        info!("Initializing GPU compute actor with {} nodes", num_nodes);

        if num_nodes > MAX_NODES {
            return Err(Error::new(
                ErrorKind::Other,
                format!("Node count exceeds limit: {}", MAX_NODES),
            ));
        }

        // Test GPU capabilities
        match self.test_gpu_capabilities().await {
            Ok(_) => info!("GPU capabilities check passed"),
            Err(e) => {
                warn!("GPU capabilities check failed: {}", e);
                return Err(e);
            }
        }

        // Create CUDA device
        let device = match self.create_cuda_device().await {
            Ok(dev) => {
                info!("CUDA device created successfully");
                dev
            },
            Err(e) => {
                error!("Failed to create CUDA device: {}", e);
                return Err(e);
            }
        };

        // Load compute kernel
        match self.load_compute_kernel(&device, num_nodes, graph).await {
            Ok(_) => {
                info!("GPU compute actor initialization complete");
                Ok(())
            },
            Err(e) => {
                error!("Failed to load compute kernel: {}", e);
                Err(e)
            }
        }
    }

    async fn create_cuda_device(&self) -> Result<Arc<CudaDevice>, Error> {
        trace!("Starting CUDA device initialization sequence");
        
        if let Ok(uuid) = env::var("NVIDIA_GPU_UUID") {
            info!("Using GPU UUID {} via environment variables", uuid);
        }

        info!("Creating CUDA device with index 0");
        match CudaDevice::new(0) {
            Ok(device) => {
                let max_threads = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let compute_mode = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let multiprocessor_count = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                
                info!("GPU Device detected:");
                info!("  Max threads per MP: {}", max_threads);
                info!("  Multiprocessor count: {}", multiprocessor_count);
                info!("  Compute mode: {}", compute_mode);
                
                if max_threads < 256 {
                    return Err(Error::new(ErrorKind::Other,
                        format!("GPU capability too low: {} threads per multiprocessor; minimum required is 256", max_threads)));
                }
                
                Ok(device.into())
            },
            Err(e) => {
                error!("Failed to create CUDA device: {}", e);
                Err(Error::new(ErrorKind::Other, format!("Failed to create CUDA device: {}", e)))
            }
        }
    }

    async fn test_gpu_capabilities(&self) -> Result<(), Error> {
        info!("Testing CUDA capabilities");
        
        match CudaDevice::count() {
            Ok(count) => {
                info!("Found {} CUDA device(s)", count);
                if count == 0 {
                    return Err(Error::new(ErrorKind::NotFound,
                        "No CUDA devices found. Ensure NVIDIA drivers are installed and working."));
                }
                Ok(())
            },
            Err(e) => {
                error!("Failed to get CUDA device count: {}", e);
                Err(Error::new(ErrorKind::Other,
                    format!("Failed to get CUDA device count: {}. Check NVIDIA drivers.", e)))
            }
        }
    }

    async fn load_compute_kernel(&mut self, device: &Arc<CudaDevice>, num_nodes: u32, graph: &GraphData) -> Result<(), Error> {
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
        
        // Build node indices map
        let mut node_indices = HashMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            node_indices.insert(node.id, idx); // node.id is now u32
        }

        self.device = Some(device.clone());
        self.force_kernel = Some(force_kernel);
        self.node_data = Some(node_data);
        self.num_nodes = num_nodes;
        self.node_indices = node_indices;

        // Copy initial graph data to GPU
        self.update_graph_data_internal(graph)?;
        
        Ok(())
    }

    fn update_graph_data_internal(&mut self, graph: &GraphData) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let node_data_slice = self.node_data.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;
 
        trace!("Updating graph data for {} nodes", graph.nodes.len());
        
        // Update node indices
        self.node_indices.clear();
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_indices.insert(node.id, idx); // node.id is now u32
        }

        // Reallocate if needed
        if graph.nodes.len() as u32 != self.num_nodes {
            info!("Reallocating GPU buffer for {} nodes", graph.nodes.len());
            *node_data_slice = device.alloc_zeros::<BinaryNodeData>(graph.nodes.len())
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            self.num_nodes = graph.nodes.len() as u32;
            self.iteration_count = 0;
        }

        // Prepare node data for GPU
        let mut node_data = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            node_data.push(BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: node.data.padding,
            });
        }

        // Copy to GPU
        device.htod_sync_copy_into(&node_data, node_data_slice)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy node data to GPU: {}", e)))?;
        
        Ok(())
    }

    fn compute_forces_internal(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let force_kernel = self.force_kernel.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Kernel not initialized"))?;
        let node_data = self.node_data.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;

        if self.cpu_fallback_active {
            warn!("GPU compute in CPU fallback mode, skipping GPU kernel");
            return Ok(());
        }

        // Check if we should reset failure count
        if self.last_failure_reset.elapsed() > FAILURE_RESET_INTERVAL {
            if self.gpu_failure_count > 0 {
                info!("Resetting GPU failure count after {} seconds", FAILURE_RESET_INTERVAL.as_secs());
                self.gpu_failure_count = 0;
                self.cpu_fallback_active = false;
            }
            self.last_failure_reset = Instant::now();
        }

        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting force computation on GPU");
        }

        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        // Launch kernel with error handling (Insight 1.9)
        let launch_result = unsafe {
            force_kernel.clone().launch(cfg, (
                node_data,
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
            ))
        };

        match launch_result {
            Ok(_) => {
                // Synchronize and check for errors
                match device.synchronize() {
                    Ok(_) => {
                        if self.iteration_count % DEBUG_THROTTLE == 0 {
                            trace!("Force computation completed successfully");
                        }
                        self.iteration_count += 1;
                        Ok(())
                    },
                    Err(e) => {
                        self.handle_gpu_error(format!("GPU synchronization failed: {}", e))
                    }
                }
            },
            Err(e) => {
                self.handle_gpu_error(format!("Kernel launch failed: {}", e))
            }
        }
    }

    fn handle_gpu_error(&mut self, error_msg: String) -> Result<(), Error> {
        self.gpu_failure_count += 1;
        error!("GPU error (failure {}/{}): {}", self.gpu_failure_count, MAX_GPU_FAILURES, error_msg);

        if self.gpu_failure_count >= MAX_GPU_FAILURES {
            warn!("GPU failure count exceeded limit, activating CPU fallback mode");
            self.cpu_fallback_active = true;
            // Reset failure count to allow retry later
            self.gpu_failure_count = 0;
            self.last_failure_reset = Instant::now();
        }

        Err(Error::new(ErrorKind::Other, error_msg))
    }

    fn get_node_data_internal(&self) -> Result<Vec<BinaryNodeData>, Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let node_data = self.node_data.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;

        let mut gpu_raw_data = vec![BinaryNodeData {
            position: Vec3Data::zero(),
            velocity: Vec3Data::zero(),
            mass: 0,
            flags: 0,
            padding: [0, 0],
        }; self.num_nodes as usize];

        device.dtoh_sync_copy_into(node_data, &mut gpu_raw_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy data from GPU: {}", e)))?;

        Ok(gpu_raw_data)
    }
}

impl Actor for GPUComputeActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("GPUComputeActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GPUComputeActor stopped");
    }
}

impl Handler<InitializeGPU> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: InitializeGPU, _ctx: &mut Self::Context) -> Self::Result {
        use actix::fut::{wrap_future, ActorFutureExt};
        
        let graph_data = msg.graph; // Capture graph data from the message

        // Directly wrap the future returned by `self.initialize_gpu`
        // `self.initialize_gpu` is an async method taking `&mut self` and `&GraphData`.
        // It returns `impl Future<Output = Result<(), std::io::Error>>`.
        // `wrap_future` makes this future `Send` by running it within the actor's context.
        Box::pin(
            wrap_future(self.initialize_gpu(&graph_data))
                .map(|result, _actor, _ctx| { // `result` is `Result<(), std::io::Error>`
                    match result {
                        Ok(_) => {
                            info!("GPU initialization successful via wrap_future().map()");
                            Ok(())
                        }
                        Err(e) => {
                            error!("GPU initialization failed via wrap_future().map(): {}", e);
                            Err(e.to_string()) // Convert std::io::Error to String
                        }
                    }
                })
        )
    }
}

impl Handler<UpdateGPUGraphData> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGPUGraphData, _ctx: &mut Self::Context) -> Self::Result {
        match self.update_graph_data_internal(&msg.graph) {
            Ok(_) => {
                trace!("Graph data updated successfully");
                Ok(())
            },
            Err(e) => {
                error!("Failed to update graph data: {}", e);
                Err(e.to_string())
            }
        }
    }
}

impl Handler<UpdateSimulationParams> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateSimulationParams, _ctx: &mut Self::Context) -> Self::Result {
        trace!("Updating simulation parameters: {:?}", msg.params);
        self.simulation_params = msg.params;
        Ok(())
    }
}

impl Handler<ComputeForces> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: ComputeForces, _ctx: &mut Self::Context) -> Self::Result {
        match self.compute_forces_internal() {
            Ok(_) => Ok(()),
            Err(e) => {
                if self.cpu_fallback_active {
                    // Return success to indicate CPU fallback should be used
                    warn!("GPU compute failed, CPU fallback active: {}", e);
                    Ok(())
                } else {
                    error!("GPU compute failed: {}", e);
                    Err(e.to_string())
                }
            }
        }
    }
}

impl Handler<GetNodeData> for GPUComputeActor {
    type Result = Result<Vec<BinaryNodeData>, String>;

    fn handle(&mut self, _msg: GetNodeData, _ctx: &mut Self::Context) -> Self::Result {
        match self.get_node_data_internal() {
            Ok(data) => {
                trace!("Retrieved {} node data items from GPU", data.len());
                Ok(data)
            },
            Err(e) => {
                error!("Failed to get node data from GPU: {}", e);
                Err(e.to_string())
            }
        }
    }
}

impl Handler<GetGPUStatus> for GPUComputeActor {
    type Result = MessageResult<GetGPUStatus>;

    fn handle(&mut self, _msg: GetGPUStatus, _ctx: &mut Self::Context) -> Self::Result {
        MessageResult(GPUStatus {
            is_initialized: self.device.is_some(),
            cpu_fallback_active: self.cpu_fallback_active,
            failure_count: self.gpu_failure_count,
            iteration_count: self.iteration_count,
            num_nodes: self.num_nodes,
        })
    }
}