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
use actix::fut::{ActorFutureExt}; // For .map() on ActorFuture
// use futures_util::future::FutureExt as _; // For .into_actor() - note the `as _` to avoid name collision if FutureExt is also in scope from elsewhere

// Constants for GPU computation
const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const NODE_SIZE: u32 = std::mem::size_of::<BinaryNodeData>() as u32;
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * NODE_SIZE;
const DEBUG_THROTTLE: u32 = 60;

// Constants for retry mechanism
// const MAX_GPU_INIT_RETRIES: u32 = 3; // Unused
// const RETRY_DELAY_MS: u64 = 500; // Unused

// Constants for error watchdog (Insight 1.9)
const MAX_GPU_FAILURES: u32 = 5;
const FAILURE_RESET_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug)]
pub struct GPUComputeActor {
    device: Option<Arc<CudaDevice>>,
    force_kernel: Option<CudaFunction>,
    node_data: Option<CudaSlice<BinaryNodeData>>,
    num_nodes: u32,
    node_indices: HashMap<u32, usize>,
    simulation_params: SimulationParams,
    iteration_count: u32,
    gpu_failure_count: u32,
    last_failure_reset: Instant,
    cpu_fallback_active: bool,
}

// Struct to hold the results of GPU initialization
struct GpuInitializationResult {
    device: Arc<CudaDevice>,
    force_kernel: CudaFunction,
    node_data: CudaSlice<BinaryNodeData>,
    num_nodes: u32,
    node_indices: HashMap<u32, usize>,
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

    // --- Static GPU Initialization Logic ---

    async fn static_test_gpu_capabilities() -> Result<(), Error> {
        info!("(Static) Testing CUDA capabilities");
        match CudaDevice::count() {
            Ok(count) => {
                info!("Found {} CUDA device(s)", count);
                if count == 0 {
                    Err(Error::new(ErrorKind::NotFound, "No CUDA devices found. Ensure NVIDIA drivers are installed and working."))
                } else {
                    Ok(())
                }
            }
            Err(e) => Err(Error::new(ErrorKind::Other, format!("Failed to get CUDA device count: {}. Check NVIDIA drivers.", e))),
        }
    }

    async fn static_create_cuda_device() -> Result<Arc<CudaDevice>, Error> {
        trace!("(Static) Starting CUDA device initialization sequence");
        if let Ok(uuid) = env::var("NVIDIA_GPU_UUID") {
            info!("(Static) Using GPU UUID {} via environment variables", uuid);
        }
        info!("(Static) Creating CUDA device with index 0");
        match CudaDevice::new(0) {
            Ok(device) => {
                let max_threads = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let compute_mode = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let multiprocessor_count = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                
                info!("(Static) GPU Device detected:");
                info!("  Max threads per MP: {}", max_threads);
                info!("  Multiprocessor count: {}", multiprocessor_count);
                info!("  Compute mode: {}", compute_mode);

                if max_threads < 256 {
                    Err(Error::new(ErrorKind::Other, format!("GPU capability too low: {} threads per multiprocessor; minimum required is 256", max_threads)))
                } else {
                    Ok(device.into())
                }
            }
            Err(e) => Err(Error::new(ErrorKind::Other, format!("Failed to create CUDA device: {}", e))),
        }
    }

    async fn static_load_compute_kernel(
        device: Arc<CudaDevice>, 
        num_nodes: u32,
        graph_nodes: &[crate::models::node::Node], // Pass slice of nodes
    ) -> Result<(CudaFunction, CudaSlice<BinaryNodeData>, HashMap<u32, usize>), Error> {
        let ptx_path = "/app/src/utils/compute_forces.ptx";
        if !Path::new(ptx_path).exists() {
            return Err(Error::new(ErrorKind::NotFound, format!("PTX file not found at {}", ptx_path)));
        }
        
        let ptx = Ptx::from_file(ptx_path);
        info!("(Static) Successfully loaded PTX file");
        
        device.load_ptx(ptx, "compute_forces_kernel", &["compute_forces_kernel"]).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let force_kernel = device.get_func("compute_forces_kernel", "compute_forces_kernel").ok_or_else(|| Error::new(ErrorKind::Other, "Function compute_forces_kernel not found"))?;
        
        info!("(Static) Allocating device memory for {} nodes", num_nodes);
        let mut node_data_gpu = device.alloc_zeros::<BinaryNodeData>(num_nodes as usize).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        let mut node_indices = HashMap::new();
        let mut node_data_host = Vec::with_capacity(graph_nodes.len());

        for (idx, node) in graph_nodes.iter().enumerate() {
            node_indices.insert(node.id, idx);
            node_data_host.push(BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: node.data.padding,
            });
        }
        
        device.htod_sync_copy_into(&node_data_host, &mut node_data_gpu).map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy node data to GPU: {}", e)))?;
        
        Ok((force_kernel, node_data_gpu, node_indices))
    }

    async fn perform_gpu_initialization(graph: GraphData) -> Result<GpuInitializationResult, Error> {
        let num_nodes = graph.nodes.len() as u32;
        info!("(Static Logic) Initializing GPU for {} nodes", num_nodes);

        if num_nodes > MAX_NODES {
            return Err(Error::new(ErrorKind::Other, format!("Node count {} exceeds limit {}", num_nodes, MAX_NODES)));
        }

        Self::static_test_gpu_capabilities().await?;
        info!("(Static Logic) GPU capabilities check passed");

        let device = Self::static_create_cuda_device().await?;
        info!("(Static Logic) CUDA device created successfully");
        
        // Pass graph.nodes which is Vec<Node>
        let (force_kernel, node_data, node_indices) = Self::static_load_compute_kernel(device.clone(), num_nodes, &graph.nodes).await?;
        info!("(Static Logic) Compute kernel loaded and data copied");
        
        Ok(GpuInitializationResult {
            device, // No Some() needed, it's Arc<CudaDevice>
            force_kernel, // No Some()
            node_data,    // No Some()
            num_nodes,
            node_indices,
        })
    }

    // --- Instance Methods ---

    fn update_graph_data_internal(&mut self, graph: &GraphData) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let node_data_slice = self.node_data.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;
 
        trace!("Updating graph data for {} nodes", graph.nodes.len());
        
        self.node_indices.clear();
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_indices.insert(node.id, idx);
        }

        if graph.nodes.len() as u32 != self.num_nodes {
            info!("Reallocating GPU buffer for {} nodes", graph.nodes.len());
            *node_data_slice = device.alloc_zeros::<BinaryNodeData>(graph.nodes.len())
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            self.num_nodes = graph.nodes.len() as u32;
            self.iteration_count = 0; // Reset iteration count on realloc
        }

        let mut host_node_data = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            host_node_data.push(BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: node.data.padding,
            });
        }

        device.htod_sync_copy_into(&host_node_data, node_data_slice)
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

        if self.last_failure_reset.elapsed() > FAILURE_RESET_INTERVAL {
            if self.gpu_failure_count > 0 {
                info!("Resetting GPU failure count after {} seconds", FAILURE_RESET_INTERVAL.as_secs());
                self.gpu_failure_count = 0;
                self.cpu_fallback_active = false; // Attempt to re-enable GPU
            }
            self.last_failure_reset = Instant::now();
        }

        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting force computation on GPU (iteration {})", self.iteration_count);
        }

        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

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
                    f32::MAX
                },
                self.iteration_count as i32,
            ))
        };

        match launch_result {
            Ok(_) => {
                match device.synchronize() {
                    Ok(_) => {
                        if self.iteration_count % DEBUG_THROTTLE == 0 {
                            trace!("Force computation completed successfully");
                        }
                        self.iteration_count += 1;
                        Ok(())
                    },
                    Err(e) => self.handle_gpu_error(format!("GPU synchronization failed: {}", e)),
                }
            },
            Err(e) => self.handle_gpu_error(format!("Kernel launch failed: {}", e)),
        }
    }

    fn handle_gpu_error(&mut self, error_msg: String) -> Result<(), Error> {
        self.gpu_failure_count += 1;
        error!("GPU error (failure {}/{}): {}", self.gpu_failure_count, MAX_GPU_FAILURES, error_msg);

        if self.gpu_failure_count >= MAX_GPU_FAILURES {
            warn!("GPU failure count exceeded limit, activating CPU fallback mode");
            self.cpu_fallback_active = true;
            // Reset failure count to allow retry later, but keep fallback active until reset interval
            // self.gpu_failure_count = 0; // Don't reset immediately, let the interval handle it
            // self.last_failure_reset = Instant::now();
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
        let graph_data_owned = msg.graph;

        let fut = GPUComputeActor::perform_gpu_initialization(graph_data_owned);
        
        // Use FutureActorExt trait's into_actor method
        let actor_fut = fut.into_actor(self);

        Box::pin(
            actor_fut.map(|result_of_logic, actor, _ctx_map| {
                match result_of_logic {
                    Ok(init_result) => {
                        actor.device = Some(init_result.device);
                        actor.force_kernel = Some(init_result.force_kernel);
                        actor.node_data = Some(init_result.node_data);
                        actor.num_nodes = init_result.num_nodes;
                        actor.node_indices = init_result.node_indices;
                        
                        // Reset other relevant state
                        actor.iteration_count = 0;
                        actor.gpu_failure_count = 0;
                        actor.last_failure_reset = Instant::now();
                        actor.cpu_fallback_active = false;

                        info!("GPU initialization successful (applied static logic result)");
                        Ok(())
                    }
                    Err(e) => {
                        error!("GPU initialization failed (static logic): {}", e);
                        actor.device = None;
                        actor.force_kernel = None;
                        actor.node_data = None;
                        actor.num_nodes = 0;
                        actor.node_indices.clear();
                        actor.cpu_fallback_active = true; // Fallback on init failure
                        Err(e.to_string())
                    }
                }
            })
        )
    }
}

impl Handler<UpdateGPUGraphData> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGPUGraphData, _ctx: &mut Self::Context) -> Self::Result {
        if self.device.is_none() {
            warn!("Attempted to update GPU graph data, but GPU is not initialized. CPU fallback may be active.");
            // Depending on desired behavior, could return Ok(()) or an error.
            // For now, let it proceed to update_graph_data_internal which will fail if device is None.
        }
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
        if self.device.is_none() {
            warn!("Attempted to compute forces, but GPU is not initialized. CPU fallback may be active.");
            return Ok(()); // Or Err, if strict GPU mode is required
        }
        match self.compute_forces_internal() {
            Ok(_) => Ok(()),
            Err(e) => {
                if self.cpu_fallback_active {
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
         if self.device.is_none() {
            warn!("Attempted to get node data, but GPU is not initialized.");
            return Err("GPU not initialized".to_string());
        }
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
