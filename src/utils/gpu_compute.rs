#[cfg(feature = "gpu")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

use std::io::Error;
use std::sync::Arc;
use log::{debug, warn};
use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::utils::socket_flow_messages::NodeData;
use tokio::sync::RwLock;

// Constants for GPU computation
#[cfg(feature = "gpu")]
const BLOCK_SIZE: u32 = 256;
#[cfg(feature = "gpu")]
const MAX_NODES: u32 = 1_000_000;
#[cfg(feature = "gpu")]
const NODE_SIZE: u32 = std::mem::size_of::<NodeData>() as u32;
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
    pub node_data: CudaSlice<NodeData>,
    pub num_nodes: u32,
    pub simulation_params: SimulationParams,
}

#[cfg(feature = "gpu")]
impl GPUCompute {
    pub async fn new(graph: &GraphData) -> Result<Arc<RwLock<Self>>, Error> {
        let num_nodes = graph.nodes.len() as u32;
        if num_nodes > MAX_NODES {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                format!("Node count exceeds limit: {}", MAX_NODES),
            ));
        }

        debug!("Initializing CUDA device");
        let device = Arc::new(CudaDevice::new(0)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?);

        debug!("Loading force computation kernel");
        let ptx = Ptx::from_file("/app/compute_forces.ptx");
            
        device.load_ptx(ptx, "compute_forces", &["compute_forces"])
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            
        let force_kernel = device.get_func("compute_forces", "compute_forces")
            .ok_or_else(|| Error::new(std::io::ErrorKind::Other, "Function compute_forces not found"))?;

        debug!("Allocating device memory for {} nodes", num_nodes);
        let node_data = device.alloc_zeros::<NodeData>(num_nodes as usize)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        debug!("Creating GPU compute instance");
        let mut instance = Self {
            device: Arc::clone(&device),
            force_kernel,
            node_data,
            num_nodes,
            simulation_params: SimulationParams::default(),
        };

        debug!("Copying initial graph data to device memory");
        instance.update_graph_data(graph)?;

        Ok(Arc::new(RwLock::new(instance)))
    }

    pub fn update_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        debug!("Updating graph data for {} nodes", graph.nodes.len());

        // Get node data directly
        let node_data: Vec<NodeData> = graph.nodes.iter()
            .map(|node| node.data.clone())
            .collect();

        // Copy data to GPU
        self.device.htod_sync_copy_into(&node_data, &mut self.node_data)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        self.num_nodes = graph.nodes.len() as u32;
        Ok(())
    }

    pub fn update_simulation_params(&mut self, params: &SimulationParams) -> Result<(), Error> {
        debug!("Updating simulation parameters: {:?}", params);
        self.simulation_params = params.clone();
        Ok(())
    }

    pub fn step(&mut self) -> Result<(), Error> {
        // Debug initial node data
        let initial_nodes = self.get_node_data()?;
        debug!("Initial node data before GPU step:");
        for (i, node) in initial_nodes.iter().take(5).enumerate() {
            debug!("Node {}: pos={:?}, vel={:?}", i, node.position, node.velocity);
        }

        let blocks = (self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        // Adjust simulation parameters for stability
        let mut params = self.simulation_params.clone();
        params.spring_strength *= 0.1; // Reduce force strength
        params.repulsion *= 0.1;      // Reduce repulsion force
        params.damping = params.damping.max(0.95); // Increase damping for stability

        unsafe {
            self.force_kernel.clone().launch(cfg, (
                &mut self.node_data,
                self.num_nodes as i32,
                params.spring_strength,
                params.repulsion,
                params.damping,
            )).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        
        // Sanity-check: retrieve updated node data and correct any NaN values
        let mut updated_nodes = self.get_node_data()?;
        let mut needs_fix = false;
        for node in &mut updated_nodes {
            for v in &mut node.position {
                if v.is_nan() {
                    *v = 0.0;
                    needs_fix = true;
                }
            }
            for v in &mut node.velocity {
                if v.is_nan() {
                    *v = 0.0;
                    needs_fix = true;
                }
            }
        }
        if needs_fix {
            // Log the correction
            warn!("GPUCompute: Detected NaN in node data. Correcting values to 0.");
            self.device.htod_sync_copy_into(&updated_nodes, &mut self.node_data)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        
        // Debug: log a sample of node positions after update
        debug!("Sample node positions after update:");
        for (i, node) in updated_nodes.iter().take(5).enumerate() {
            debug!("Node {}: pos={:?}, vel={:?}", i, node.position, node.velocity);
        }
        
        Ok(())
    }

    pub fn get_node_data(&self) -> Result<Vec<NodeData>, Error> {
        let mut gpu_nodes = vec![NodeData {
            position: [0.0; 3],
            velocity: [0.0; 3],
            mass: 0,
            flags: 0,
            padding: [0; 2],
        }; self.num_nodes as usize];

        self.device.dtoh_sync_copy_into(&self.node_data, &mut gpu_nodes)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(gpu_nodes)
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
        assert_eq!(size_of::<NodeData>(), 28); // 24 bytes for position/velocity + 4 bytes for mass/flags/padding
    }
}
