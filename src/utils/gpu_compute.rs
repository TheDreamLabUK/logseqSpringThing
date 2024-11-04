use wgpu::{Device, Queue, Buffer, BindGroup, ComputePipeline, InstanceDescriptor};
use wgpu::util::DeviceExt;
use std::io::Error;
use std::collections::HashMap;
use log::{debug, info, warn, error};
use crate::models::graph::GraphData;
use crate::models::node::{Node, GPUNode};
use crate::models::edge::GPUEdge;
use crate::models::simulation_params::SimulationParams;
use rand::Rng;
use futures::channel::oneshot;

// Constants for buffer management and computation
const WORKGROUP_SIZE: u32 = 256;
const INITIAL_BUFFER_SIZE: u64 = 1024 * 1024;  // 1MB initial size
const BUFFER_ALIGNMENT: u64 = 256;  // Required GPU memory alignment
const EDGE_SIZE: u64 = 32;  // Size of Edge struct (must match WGSL)
const NODE_SIZE: u64 = 48;  // Size of Node struct in WGSL (vec3 alignment + padding)
const MAX_NODES: u32 = 1_000_000;  // Safety limit for number of nodes
const MAX_EDGES: u32 = 5_000_000;  // Safety limit for number of edges

/// Represents adjacency information for graph nodes
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Adjacency {
    offset: u32,
    count: u32,
}

/// Parameters for fisheye distortion effect
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FisheyeParams {
    pub enabled: u32,
    pub strength: f32,
    pub focus_point: [f32; 3],
    pub radius: f32,
}

impl Default for FisheyeParams {
    fn default() -> Self {
        Self {
            enabled: 0,
            strength: 0.5,
            focus_point: [0.0, 0.0, 0.0],
            radius: 100.0,
        }
    }
}

/// Main struct for GPU-accelerated graph computations
pub struct GPUCompute {
    device: Device,
    queue: Queue,
    nodes_buffer: Buffer,
    edges_buffer: Buffer,
    adjacency_buffer: Buffer,
    adjacency_list_buffer: Buffer,
    simulation_params_buffer: Buffer,
    fisheye_params_buffer: Buffer,
    force_bind_group: BindGroup,
    fisheye_bind_group: BindGroup,
    force_pipeline: ComputePipeline,
    fisheye_pipeline: ComputePipeline,
    num_nodes: u32,
    num_edges: u32,
    simulation_params: SimulationParams,
    fisheye_params: FisheyeParams,
}

impl GPUCompute {
    /// Creates a new instance of GPUCompute with initialized GPU resources
    pub async fn new() -> Result<Self, Error> {
        debug!("Initializing GPU compute capabilities");
        
        // Initialize GPU instance with high performance preference
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::new(std::io::ErrorKind::Other, "Failed to find an appropriate GPU adapter"))?;

        info!("Selected GPU adapter: {:?}", adapter.get_info().name);

        // Log device limits for debugging
        let limits = adapter.limits();
        debug!("Device limits: {:?}", limits);

        // Request device with default limits
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Primary Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Create shader modules
        let force_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Force Calculation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("force_calculation.wgsl").into()),
        });

        let fisheye_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fisheye Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fisheye.wgsl").into()),
        });

        // Create bind group layouts
        let force_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Force Compute Bind Group Layout"),
            entries: &[
                // Nodes buffer (read/write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None, // Remove min_binding_size for array buffer
                    },
                    count: None,
                },
                // Edges buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(wgpu::BufferSize::new(EDGE_SIZE).unwrap()),
                    },
                    count: None,
                },
                // Adjacency buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Adjacency list buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation parameters (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fisheye_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fisheye Bind Group Layout"),
            entries: &[
                // Nodes buffer (read/write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Fisheye parameters (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create parameter buffers
        let simulation_params = SimulationParams::default();
        let simulation_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params Buffer"),
            contents: bytemuck::cast_slice(&[simulation_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fisheye_params = FisheyeParams::default();
        let fisheye_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fisheye Params Buffer"),
            contents: bytemuck::cast_slice(&[fisheye_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute pipelines
        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Force Directed Graph Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Force Pipeline Layout"),
                bind_group_layouts: &[&force_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &force_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fisheye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fisheye Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fisheye Pipeline Layout"),
                bind_group_layouts: &[&fisheye_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &fisheye_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create aligned buffers
        let aligned_initial_size = (INITIAL_BUFFER_SIZE + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1);
        
        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes Buffer"),
            size: aligned_initial_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let edges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edges Buffer"),
            size: aligned_initial_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let adjacency_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjacency Buffer"),
            size: aligned_initial_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let adjacency_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjacency List Buffer"),
            size: aligned_initial_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind groups
        let force_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Force Compute Bind Group"),
            layout: &force_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adjacency_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adjacency_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: simulation_params_buffer.as_entire_binding(),
                },
            ],
        });

        let fisheye_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fisheye Bind Group"),
            layout: &fisheye_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fisheye_params_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            nodes_buffer,
            edges_buffer,
            adjacency_buffer,
            adjacency_list_buffer,
            simulation_params_buffer,
            fisheye_params_buffer,
            force_bind_group,
            fisheye_bind_group,
            force_pipeline,
            fisheye_pipeline,
            num_nodes: 0,
            num_edges: 0,
            simulation_params,
            fisheye_params,
        })
    }

    /// Sets the graph data with optimized adjacency list structure and enhanced validation
    pub fn set_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        debug!("Setting graph data: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        
        // Validate input sizes
        if graph.nodes.is_empty() {
            return Err(Error::new(std::io::ErrorKind::InvalidInput, "Graph contains no nodes"));
        }
        if graph.edges.is_empty() {
            warn!("Graph contains no edges");
        }
        if graph.nodes.len() > MAX_NODES as usize {
            return Err(Error::new(std::io::ErrorKind::InvalidInput, 
                format!("Number of nodes ({}) exceeds maximum allowed ({})", 
                    graph.nodes.len(), MAX_NODES)));
        }
        if graph.edges.len() > MAX_EDGES as usize {
            return Err(Error::new(std::io::ErrorKind::InvalidInput, 
                format!("Number of edges ({}) exceeds maximum allowed ({})", 
                    graph.edges.len(), MAX_EDGES)));
        }

        self.num_nodes = graph.nodes.len() as u32;
        self.num_edges = graph.edges.len() as u32;

        // Validate maximum buffer sizes
        let max_buffer_size = self.device.limits().max_buffer_size;
        let required_edge_buffer_size = (self.num_edges as u64) * EDGE_SIZE;
        let required_node_buffer_size = (self.num_nodes as u64) * NODE_SIZE;

        if required_edge_buffer_size > max_buffer_size || required_node_buffer_size > max_buffer_size {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Buffer size exceeds device limit ({} bytes). Nodes: {} bytes, Edges: {} bytes",
                    max_buffer_size, required_node_buffer_size, required_edge_buffer_size)
            ));
        }

        // Convert nodes to GPU representation with initial random positions
        let mut rng = rand::thread_rng();
        let gpu_nodes: Vec<GPUNode> = graph.nodes.iter().enumerate().map(|(_i, _)| {
            GPUNode {
                x: rng.gen_range(-75.0..75.0),
                y: rng.gen_range(-75.0..75.0),
                z: rng.gen_range(-75.0..75.0),
                _padding0: 0.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                _padding1: 0.0,
                mass: 1.0,
                _padding2: 0.0,
                _padding3: 0.0,
                _padding4: 0.0,
            }
        }).collect();

        // Convert edges to GPU representation with validation
        let gpu_edges: Vec<GPUEdge> = graph.edges.iter()
            .map(|edge| edge.to_gpu_edge(&graph.nodes))
            .collect();

        // Validate edge data
        debug!("GPU Edge validation:");
        debug!("Number of edges: {}", gpu_edges.len());
        debug!("Size of GPUEdge struct: {} bytes", std::mem::size_of::<GPUEdge>());
        debug!("Total edge buffer size: {} bytes", gpu_edges.len() * std::mem::size_of::<GPUEdge>());
        debug!("Required aligned size: {} bytes", 
            ((gpu_edges.len() * std::mem::size_of::<GPUEdge>()) as u64 + BUFFER_ALIGNMENT - 1) 
            & !(BUFFER_ALIGNMENT - 1));

        // Verify edge size
        if std::mem::size_of::<GPUEdge>() != EDGE_SIZE as usize {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                format!("GPUEdge size mismatch: expected {} bytes, got {} bytes",
                    EDGE_SIZE, std::mem::size_of::<GPUEdge>())
            ));
        }

        // Build optimized adjacency list with validation
        let mut adjacency = vec![Adjacency { offset: 0, count: 0 }; self.num_nodes as usize];
        let mut adjacency_indices = Vec::with_capacity(self.num_edges as usize);

        // Count edges per node with validation
        for (i, edge) in gpu_edges.iter().enumerate() {
            if edge.source >= self.num_nodes {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Edge {} has invalid source node index: {}", i, edge.source)
                ));
            }
            if edge.target_idx >= self.num_nodes {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Edge {} has invalid target node index: {}", i, edge.target_idx)
                ));
            }
            adjacency[edge.source as usize].count += 1;
        }

        // Calculate offsets with validation
        let mut current_offset = 0u32;
        for adj in adjacency.iter_mut() {
            // Check for overflow
            if current_offset.checked_add(adj.count).is_none() {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Adjacency list offset overflow"
                ));
            }
            adj.offset = current_offset;
            current_offset += adj.count;
            adj.count = 0; // Reset count for the second pass
        }

        // Build adjacency list with validation
        adjacency_indices.resize(self.num_edges as usize, 0);
        for edge in gpu_edges.iter() {
            let adj = &mut adjacency[edge.source as usize];
            let index = (adj.offset + adj.count) as usize;
            if index >= adjacency_indices.len() {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid adjacency list index: {} >= {}", index, adjacency_indices.len())
                ));
            }
            adjacency_indices[index] = edge.target_idx;
            adj.count += 1;
        }

        // Update buffers with validation
        match self.update_buffers(&gpu_nodes, &gpu_edges, &adjacency, &adjacency_indices) {
            Ok(_) => {
                debug!("Graph data set successfully with optimized adjacency structure");
                Ok(())
            },
            Err(e) => {
                error!("Failed to update buffers: {}", e);
                Err(e)
            }
        }
    }

    /// Updates the force-directed graph parameters
    pub fn set_force_directed_params(&mut self, params: &SimulationParams) -> Result<(), Error> {
        debug!("Updating force-directed parameters: {:?}", params);
        self.simulation_params = *params;
        self.queue.write_buffer(
            &self.simulation_params_buffer,
            0,
            bytemuck::cast_slice(&[self.simulation_params])
        );
        Ok(())
    }

    /// Updates the fisheye parameters
    pub fn set_fisheye_params(&mut self, enabled: bool, strength: f32, focus_point: [f32; 3], radius: f32) -> Result<(), Error> {
        self.fisheye_params = FisheyeParams {
            enabled: if enabled { 1 } else { 0 },
            strength,
            focus_point,
            radius,
        };
        self.queue.write_buffer(
            &self.fisheye_params_buffer,
            0,
            bytemuck::cast_slice(&[self.fisheye_params])
        );
        Ok(())
    }

    /// Computes forces with optimized dispatch
    pub fn compute_forces(&self) -> Result<(), Error> {
        if self.num_nodes == 0 {
            debug!("No nodes to compute forces for");
            return Ok(());
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Force Computation Encoder"),
        });

        // Optimized force calculation pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Force Computation Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.force_pipeline);
            cpass.set_bind_group(0, &self.force_bind_group, &[]);
            
            let workgroup_count = (self.num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            debug!("Dispatching force computation with {} workgroups", workgroup_count);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Fisheye distortion pass (only if enabled)
        if self.fisheye_params.enabled == 1 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fisheye Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.fisheye_pipeline);
            cpass.set_bind_group(0, &self.fisheye_bind_group, &[]);
            
            let workgroup_count = (self.num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            debug!("Dispatching fisheye computation with {} workgroups", workgroup_count);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Updates the GPU buffers with optimized memory alignment
    fn update_buffers(
        &mut self,
        gpu_nodes: &[GPUNode],
        gpu_edges: &[GPUEdge],
        adjacency: &[Adjacency],
        adjacency_indices: &[u32],
    ) -> Result<(), Error> {
        // Helper function to create a buffer with proper alignment and padding
        let create_aligned_buffer = |device: &Device, data: &[u8], element_size: u64, label: &str, usage: wgpu::BufferUsages| -> Buffer {
            // Calculate total size needed with proper alignment
            let data_size = data.len() as u64;
            let num_elements = (data_size + element_size - 1) / element_size;
            let aligned_size = ((num_elements * element_size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1)) as usize;
            
            debug!("Creating aligned buffer '{}': data_size={}, aligned_size={}", 
                label, data_size, aligned_size);
            
            // Create a new buffer with the aligned size
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: aligned_size as u64,
                usage,
                mapped_at_creation: true,
            });

            // Write data to the buffer with padding
            {
                let mut view = buffer.slice(..).get_mapped_range_mut();
                // Zero out the entire buffer first
                view.fill(0);
                // Copy actual data
                view[..data.len()].copy_from_slice(data);
            }
            buffer.unmap();
            buffer
        };

        // Calculate element sizes for proper alignment
        let node_size = std::mem::size_of::<GPUNode>() as u64;
        let edge_size = std::mem::size_of::<GPUEdge>() as u64;
        let adjacency_size = std::mem::size_of::<Adjacency>() as u64;
        let index_size = std::mem::size_of::<u32>() as u64;

        debug!("Buffer element sizes - Node: {}, Edge: {}, Adjacency: {}, Index: {}", 
            node_size, edge_size, adjacency_size, index_size);

        // Ensure edge_size is 32 bytes
        assert_eq!(edge_size, EDGE_SIZE, "GPUEdge size must be 32 bytes");

        // Update all buffers with proper alignment
        self.nodes_buffer = create_aligned_buffer(
            &self.device,
            bytemuck::cast_slice(gpu_nodes),
            node_size,
            "Nodes Buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );

        self.edges_buffer = create_aligned_buffer(
            &self.device,
            bytemuck::cast_slice(gpu_edges),
            edge_size,
            "Edges Buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        self.adjacency_buffer = create_aligned_buffer(
            &self.device,
            bytemuck::cast_slice(adjacency),
            adjacency_size,
            "Adjacency Buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        self.adjacency_list_buffer = create_aligned_buffer(
            &self.device,
            bytemuck::cast_slice(adjacency_indices),
            index_size,
            "Adjacency List Buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // Recreate bind groups with the new buffers
        self.force_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Force Compute Bind Group"),
            layout: &self.force_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.edges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.adjacency_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.adjacency_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.simulation_params_buffer.as_entire_binding(),
                },
            ],
        });

        self.fisheye_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fisheye Bind Group"),
            layout: &self.fisheye_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.fisheye_params_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(())
    }

    /// Gets the updated node positions with optimized memory mapping
    pub async fn get_updated_positions(&self) -> Result<Vec<Node>, Error> {
        if self.num_nodes == 0 {
            return Ok(Vec::new());
        }

        let buffer_size = (self.num_nodes as u64) * std::mem::size_of::<GPUNode>() as u64;
        let aligned_size = (buffer_size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1);
        
        debug!("Creating staging buffer for position readback: size={}, aligned_size={}", 
            buffer_size, aligned_size);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Node Position Staging Buffer"),
            size: aligned_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Position Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.nodes_buffer, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        match rx.await {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range();
                let gpu_nodes: &[GPUNode] = bytemuck::cast_slice(&data);

                let nodes: Vec<Node> = gpu_nodes.iter().enumerate().map(|(i, gpu_node)| {
                    Node {
                        id: i.to_string(),
                        label: format!("Node {}", i),
                        metadata: HashMap::new(),
                        x: gpu_node.x,
                        y: gpu_node.y,
                        z: gpu_node.z,
                        vx: gpu_node.vx,
                        vy: gpu_node.vy,
                        vz: gpu_node.vz,
                    }
                }).collect();

                drop(data);
                staging_buffer.unmap();

                Ok(nodes)
            },
            Ok(Err(e)) => Err(Error::new(std::io::ErrorKind::Other, format!("Buffer mapping failed: {:?}", e))),
            Err(e) => Err(Error::new(std::io::ErrorKind::Other, format!("Channel receive failed: {}", e))),
        }
    }
}
