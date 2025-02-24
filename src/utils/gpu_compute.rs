#[cfg(feature = "gpu")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;
use cudarc::driver::sys::CUdevice_attribute_enum;

use std::io::{Error, ErrorKind};
use std::sync::Arc;
use log::{debug, error, warn};
use crate::models::graph::GraphData;
use std::collections::HashMap;
use crate::models::simulation_params::SimulationParams;
use crate::utils::socket_flow_messages::BinaryNodeData;
use tokio::sync::RwLock;

// Constants for GPU computation
#[cfg(feature = "gpu")]
const BLOCK_SIZE: u32 = 256;
#[cfg(feature = "gpu")]
const MAX_NODES: u32 = 1_000_000;
#[cfg(feature = "gpu")]
const NODE_SIZE: u32 = std::mem::size_of::<BinaryNodeData>() as u32;
#[cfg(feature = "gpu")]
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * NODE_SIZE;

// CPU-only version
#[cfg(not(feature = "gpu"))]
#[derive(Debug)]
pub struct GPUCompute;

#[cfg(not(feature = "gpu"))]
impl GPUCompute {
    pub async fn new(_graph: &GraphData) -> Result<Arc<RwLock<Self>>, Error> {
        Err(Error::new(std::io::ErrorKind::Unsupported, "GPU support is not enabled"))
    }
}

// GPU-enabled version
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GPUCompute {
    pub device: Arc<CudaDevice>,
    pub force_kernel: CudaFunction,
    pub node_data: CudaSlice<BinaryNodeData>,
    pub num_nodes: u32,
    pub node_indices: HashMap<String, usize>,
    pub simulation_params: SimulationParams,
}

#[cfg(feature = "gpu")]
impl GPUCompute {
    pub fn test_gpu() -> Result<(), Error> {
        debug!("Running GPU test");
        
        // Create a simple test device
        let device = CudaDevice::new(0)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        // Try to allocate and manipulate some memory
        let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gpu_data = device.alloc_zeros::<f32>(5)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        device.dtoh_sync_copy_into(&gpu_data, &mut test_data.clone())
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        debug!("GPU test successful");
        Ok(())
    }

    pub async fn new(graph: &GraphData) -> Result<Arc<RwLock<Self>>, Error> {
        let num_nodes = graph.nodes.len() as u32;
        debug!("Initializing GPU compute with {} nodes", num_nodes);
        
        if num_nodes > MAX_NODES {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                format!("Node count exceeds limit: {}", MAX_NODES),
            ));
        }

        debug!("Attempting to create CUDA device");
        let device = match CudaDevice::new(0) {
            Ok(dev) => {
                debug!("CUDA device created successfully");
                let max_threads = dev.as_ref().attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let compute_mode = dev.as_ref().attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let multiprocessor_count = dev.as_ref().attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _)
                    .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                debug!("GPU Device detected:");
                debug!("  Max threads per MP: {}", max_threads);
                debug!("  Multiprocessor count: {}", multiprocessor_count);
                debug!("  Compute mode: {}", compute_mode);
                
                if max_threads < 256 {
                    return Err(Error::new(ErrorKind::Other, 
                        format!("GPU capability too low. Device supports only {} threads per multiprocessor, minimum required is 256", 
                            max_threads)));
                }
                Arc::new(dev)
            },
            Err(e) => {
                error!("Failed to create CUDA device: {}", e);
                return Err(Error::new(ErrorKind::Other, e.to_string()));
            }
        };

        debug!("Loading force computation kernel");
        let ptx_path = "/app/src/utils/compute_forces.ptx";

        if !std::path::Path::new(ptx_path).exists() {
            warn!("PTX file does not exist at {}", ptx_path);
            return Err(Error::new(ErrorKind::NotFound, 
                format!("PTX file not found at {}", ptx_path)));
        }

        let ptx = Ptx::from_file(ptx_path);

        debug!("Successfully loaded PTX file");
            
        device.load_ptx(ptx, "compute_forces_kernel", &["compute_forces_kernel"])
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            
        let force_kernel = device.get_func("compute_forces_kernel", "compute_forces_kernel")
            .ok_or_else(|| Error::new(std::io::ErrorKind::Other, "Function compute_forces_kernel not found"))?;

        debug!("Allocating device memory for {} nodes", num_nodes);
        let node_data = device.alloc_zeros::<BinaryNodeData>(num_nodes as usize)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        debug!("Creating GPU compute instance");
        // Create node ID to index mapping
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
        };

        debug!("Copying initial graph data to device memory");
        instance.update_graph_data(graph)?;

        Ok(Arc::new(RwLock::new(instance)))
    }

    pub fn update_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        debug!("Updating graph data for {} nodes", graph.nodes.len());

        // Update node index mapping
        self.node_indices.clear();
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_indices.insert(node.id.clone(), idx);
        }

        // Reallocate buffer if node count changed
        if graph.nodes.len() as u32 != self.num_nodes {
            debug!("Reallocating GPU buffer for {} nodes", graph.nodes.len());
            self.node_data = self.device.alloc_zeros::<BinaryNodeData>(graph.nodes.len())
                .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            self.num_nodes = graph.nodes.len() as u32;
        }

        // Prepare node data
        let mut node_data = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            node_data.push(BinaryNodeData {
                position: node.data.position,
                velocity: node.data.velocity,
                mass: node.data.mass,
                flags: 1, // Active by default
                padding: [0, 0],
            });
        }

        debug!("Copying {} nodes to GPU", graph.nodes.len());

        // Copy data to GPU
        self.device.htod_sync_copy_into(&node_data, &mut self.node_data)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    pub fn update_simulation_params(&mut self, params: &SimulationParams) -> Result<(), Error> {
        debug!("Updating simulation parameters: {:?}", params);
        self.simulation_params = params.clone();
        Ok(())
    }

    pub fn compute_forces(&mut self) -> Result<(), Error> {
        debug!("Starting force computation on GPU");
        
        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        debug!("Launch config: blocks={}, threads={}, shared_mem={}",
            blocks, BLOCK_SIZE, SHARED_MEM_SIZE);

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
                    f32::MAX // Effectively disable bounds
                }
            )).map_err(|e| {
                error!("Kernel launch failed: {}", e);
                Error::new(ErrorKind::Other, e.to_string())
            })?;
        }

        debug!("Force computation completed");
        Ok(())
    }

    pub fn get_node_data(&self) -> Result<Vec<BinaryNodeData>, Error> {
        let mut gpu_nodes = vec![BinaryNodeData {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            mass: 0,
            flags: 0,
            padding: [0, 0],
        }; self.num_nodes as usize];

        self.device.dtoh_sync_copy_into(&self.node_data, &mut gpu_nodes)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(gpu_nodes)
    }

    pub fn step(&mut self) -> Result<(), Error> {
        self.compute_forces()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_compute_initialization() {
        let graph = GraphData::default();
        let gpu_compute = GPUCompute::new(&graph).await;
        #[cfg(feature = "gpu")]
        assert!(gpu_compute.is_ok());
        #[cfg(not(feature = "gpu"))]
        assert!(gpu_compute.is_err());
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_node_data_transfer() {
        let mut graph = GraphData::default();
        let gpu_compute = GPUCompute::new(&graph).await.unwrap();
        let gpu_compute = Arc::try_unwrap(gpu_compute).unwrap().into_inner();
        let node_data = gpu_compute.get_node_data().unwrap();
        assert_eq!(node_data.len(), graph.nodes.len());
    }

    #[test]
    fn test_node_data_memory_layout() {
        use std::mem::size_of;
        assert_eq!(size_of::<BinaryNodeData>(), 28); // 24 bytes for position/velocity + 4 bytes for mass/flags/padding
    }
}
