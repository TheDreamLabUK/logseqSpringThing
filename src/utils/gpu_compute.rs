use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use std::io::Error;
use std::sync::Arc;
use log::debug;
use crate::models::graph::GraphData;
use crate::models::node::GPUNode;
use crate::models::simulation_params::SimulationParams;
use tokio::sync::RwLock;

const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const MIN_DISTANCE: f32 = 0.0001;
const FLOAT3_SIZE: u32 = 12;
const FLOAT_SIZE: u32 = 4;
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * (FLOAT3_SIZE + FLOAT_SIZE);

const FORCE_KERNEL: &str = r#"
extern "C" __global__ void compute_forces(
    float* positions,
    float* velocities,
    unsigned char* masses,
    int num_nodes,
    float spring_strength,
    float repulsion,
    float damping
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int pos_idx = idx * 3;
    float3 pos_i = make_float3(
        positions[pos_idx],
        positions[pos_idx + 1],
        positions[pos_idx + 2]
    );
    float mass_i = (float)masses[idx];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    __shared__ float3 shared_positions[BLOCK_SIZE];
    __shared__ float shared_masses[BLOCK_SIZE];

    for (int tile = 0; tile < (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        int shared_idx = tile * BLOCK_SIZE + threadIdx.x;
        if (shared_idx < num_nodes) {
            shared_positions[threadIdx.x] = make_float3(
                positions[shared_idx * 3],
                positions[shared_idx * 3 + 1],
                positions[shared_idx * 3 + 2]
            );
            shared_masses[threadIdx.x] = (float)masses[shared_idx];
        }
        __syncthreads();

        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE && tile * BLOCK_SIZE + j < num_nodes; j++) {
            if (tile * BLOCK_SIZE + j == idx) continue;
            
            float3 pos_j = shared_positions[j];
            float mass_j = shared_masses[j];
            float3 diff = make_float3(
                pos_i.x - pos_j.x,
                pos_i.y - pos_j.y,
                pos_i.z - pos_j.z
            );
            
            float dist = fmaxf(sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z), MIN_DISTANCE);
            float force_mag = repulsion * mass_i * mass_j / (dist * dist);
            force.x += force_mag * diff.x / dist;
            force.y += force_mag * diff.y / dist;
            force.z += force_mag * diff.z / dist;
        }
        __syncthreads();
    }

    int vel_idx = idx * 3;
    float3 vel = make_float3(
        velocities[vel_idx],
        velocities[vel_idx + 1],
        velocities[vel_idx + 2]
    );
    
    vel.x = (vel.x + force.x) * damping;
    vel.y = (vel.y + force.y) * damping;
    vel.z = (vel.z + force.z) * damping;

    pos_i.x += vel.x;
    pos_i.y += vel.y;
    pos_i.z += vel.z;

    positions[pos_idx] = pos_i.x;
    positions[pos_idx + 1] = pos_i.y;
    positions[pos_idx + 2] = pos_i.z;

    velocities[vel_idx] = vel.x;
    velocities[vel_idx + 1] = vel.y;
    velocities[vel_idx + 2] = vel.z;
}
"#;

pub struct GPUCompute {
    device: Arc<CudaDevice>,
    force_kernel: CudaFunction,
    positions: CudaSlice<f32>,
    velocities: CudaSlice<f32>,
    masses: CudaSlice<u8>,
    num_nodes: u32,
    simulation_params: SimulationParams,
}

impl GPUCompute {
    pub async fn new(graph: &GraphData) -> Result<Arc<RwLock<Self>>, Error> {
        let num_nodes = graph.nodes.len() as u32;
        if num_nodes > MAX_NODES {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Node count exceeds limit: {}", MAX_NODES),
            ));
        }

        debug!("Initializing CUDA device");
        let device = Arc::new(CudaDevice::new(0)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?);

        debug!("Compiling and loading force computation kernel");
        let ptx = Ptx::from_src(FORCE_KERNEL);

        device.load_ptx(ptx, "compute_forces", &["compute_forces"])
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    
        let force_kernel = device.get_func("compute_forces", "compute_forces")
            .ok_or_else(|| Error::new(std::io::ErrorKind::Other, "Function compute_forces not found"))?;
    

        debug!("Allocating device memory for {} nodes", num_nodes);
        let positions = device.alloc_zeros::<f32>((num_nodes * 3) as usize)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let velocities = device.alloc_zeros::<f32>((num_nodes * 3) as usize)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let masses = device.alloc_zeros::<u8>(num_nodes as usize)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        debug!("Creating GPU compute instance");
        let mut instance = Self {
            device: Arc::clone(&device),
            force_kernel,
            positions,
            velocities,
            masses,
            num_nodes,
            simulation_params: SimulationParams::default(),
        };

        debug!("Copying initial graph data to device memory");
        instance.update_graph_data(graph)?;

        Ok(Arc::new(RwLock::new(instance)))
    }

    pub fn update_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        debug!("Updating graph data for {} nodes", graph.nodes.len());

        let mut positions = Vec::with_capacity(graph.nodes.len() * 3);
        let mut velocities = Vec::with_capacity(graph.nodes.len() * 3);
        let mut masses = Vec::with_capacity(graph.nodes.len());

        for node in &graph.nodes {
            positions.extend_from_slice(&[node.x, node.y, node.z]);
            velocities.extend_from_slice(&[node.vx, node.vy, node.vz]);
            masses.push(node.to_gpu_node().mass);
        }

        self.device.htod_sync_copy_into(&positions, &mut self.positions)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.device.htod_sync_copy_into(&velocities, &mut self.velocities)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.device.htod_sync_copy_into(&masses, &mut self.masses)
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
            shared_mem_bytes: SHARED_MEM_SIZE as u32,
        };

        let params = &self.simulation_params;
        unsafe {
            self.force_kernel.clone().launch(cfg, (
                &mut self.positions,
                &mut self.velocities,
                &mut self.masses,
                self.num_nodes as i32,
                params.spring_strength,
                params.repulsion,
                params.damping,
            )).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }
        Ok(())
    }

    pub fn get_node_positions(&self) -> Result<Vec<GPUNode>, Error> {
        let mut positions = vec![0.0f32; (self.num_nodes as usize) * 3];
        let mut velocities = vec![0.0f32; (self.num_nodes as usize) * 3];
        let mut masses = vec![0u8; self.num_nodes as usize];

        self.device.dtoh_sync_copy_into(&self.positions, &mut positions)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.device.dtoh_sync_copy_into(&self.velocities, &mut velocities)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.device.dtoh_sync_copy_into(&self.masses, &mut masses)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let mut nodes = Vec::with_capacity(self.num_nodes as usize);
        for i in 0..(self.num_nodes as usize) {
            nodes.push(GPUNode {
                x: positions[i * 3],
                y: positions[i * 3 + 1],
                z: positions[i * 3 + 2],
                vx: velocities[i * 3],
                vy: velocities[i * 3 + 1],
                vz: velocities[i * 3 + 2],
                mass: masses[i],
                flags: 0,
                padding: [0; 2],
            });
        }

        Ok(nodes)
    }
}
