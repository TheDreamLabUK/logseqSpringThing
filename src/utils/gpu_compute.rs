use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use std::io::Error;
use std::sync::Arc;
use log::debug;
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::NodeData;
use crate::models::simulation_params::SimulationParams;
use tokio::sync::RwLock;

const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const NODE_SIZE: u32 = 28; // 24 bytes for position/velocity + 4 bytes for mass/flags/padding
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * NODE_SIZE;

pub struct GPUCompute {
    device: Arc<CudaDevice>,
    force_kernel: CudaFunction,
    node_data: CudaSlice<NodeData>,
    num_nodes: u32,
    simulation_params: SimulationParams,
}

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

        // Extract NodeData from graph nodes
        let node_data: Vec<NodeData> = graph.nodes.iter()
            .map(|node| node.data)
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
        let blocks = (self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        let params = &self.simulation_params;
        unsafe {
            self.force_kernel.clone().launch(cfg, (
                &mut self.node_data,
                self.num_nodes as i32,
                params.spring_strength,
                params.repulsion,
                params.damping,
            )).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        Ok(())
    }

    pub fn get_node_data(&self) -> Result<Vec<NodeData>, Error> {
        let mut node_data = vec![NodeData {
            position: [0.0; 3],
            velocity: [0.0; 3],
            mass: 0,
            flags: 0,
            padding: [0; 2],
        }; self.num_nodes as usize];

        self.device.dtoh_sync_copy_into(&self.node_data, &mut node_data)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(node_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::graph::GraphData;

    #[tokio::test]
    async fn test_gpu_compute_initialization() {
        let graph = GraphData::default();
        let gpu_compute = GPUCompute::new(&graph).await;
        assert!(gpu_compute.is_ok());
    }

    #[tokio::test]
    async fn test_node_data_transfer() {
        let mut graph = GraphData::default();
        // Add test nodes...
        let gpu_compute = GPUCompute::new(&graph).await.unwrap();
        let gpu_compute = Arc::try_unwrap(gpu_compute).unwrap().into_inner();
        let node_data = gpu_compute.get_node_data().unwrap();
        assert_eq!(node_data.len(), graph.nodes.len());
    }
}
