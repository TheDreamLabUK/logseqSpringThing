// Previous imports remain the same...
use wgpu::{Device, Queue, Buffer, BindGroup, ComputePipeline, InstanceDescriptor};
use wgpu::util::DeviceExt;
use std::io::Error;
use std::collections::HashMap;
use log::{debug, info};
use crate::models::graph::GraphData;
use crate::models::node::{Node, GPUNode};
use crate::models::edge::GPUEdge;
use crate::models::simulation_params::SimulationParams;
use rand::Rng;
use futures::channel::oneshot;

// Constants and struct definitions remain the same...
const WORKGROUP_SIZE: u32 = 256;
const INITIAL_BUFFER_SIZE: u64 = 1024 * 1024;
const BUFFER_ALIGNMENT: u64 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Adjacency {
    offset: u32,
    count: u32,
}

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

// Main struct definition remains the same...
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
    pub async fn new() -> Result<Self, Error> {
        debug!("Initializing GPU compute capabilities");
        
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::new(std::io::ErrorKind::Other, "Failed to find an appropriate adapter"))?;

        info!("Selected GPU adapter: {:?}", adapter.get_info().name);

        let limits = adapter.limits();
        debug!("Device limits: {:?}", limits);

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

        let force_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Force Calculation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("force_calculation.wgsl").into()),
        });

        let fisheye_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fisheye Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fisheye.wgsl").into()),
        });

        // Create optimized bind group layouts with adjacency buffers
        let force_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Force Compute Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        // Create initial parameters with proper alignment
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

        // Create compute pipelines with Some("main") for entry points
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

        // Create optimized bind groups
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

    /// Sets the graph data with optimized adjacency list structure
    pub fn set_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        debug!("Setting graph data: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        
        self.num_nodes = graph.nodes.len() as u32;
        self.num_edges = graph.edges.len() as u32;

        // Convert nodes to GPU representation with initial random positions
        let mut rng = rand::thread_rng();
        let gpu_nodes: Vec<GPUNode> = graph.nodes.iter().enumerate().map(|(_i, _)| {
            GPUNode {
                x: rng.gen_range(-75.0..75.0),
                y: rng.gen_range(-75.0..75.0),
                z: rng.gen_range(-75.0..75.0),
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                mass: 1.0,
                padding1: 0,
            }
        }).collect();

        // Convert edges to GPU representation
        let gpu_edges: Vec<GPUEdge> = graph.edges.iter()
            .map(|edge| edge.to_gpu_edge(&graph.nodes))
            .collect();

        // Build optimized adjacency list
        let mut adjacency = vec![Adjacency { offset: 0, count: 0 }; self.num_nodes as usize];
        let mut adjacency_indices = Vec::with_capacity(self.num_edges as usize);

        // Count edges per node
        for edge in gpu_edges.iter() {
            adjacency[edge.source as usize].count += 1;
        }

        // Calculate offsets
        let mut current_offset = 0;
        for adj in adjacency.iter_mut() {
            adj.offset = current_offset;
            current_offset += adj.count;
            adj.count = 0; // Reset count for the second pass
        }

        // Build adjacency list
        for edge in gpu_edges.iter() {
            let adj = &mut adjacency[edge.source as usize];
            let index = (adj.offset + adj.count) as usize;
            if index < adjacency_indices.len() {
                adjacency_indices[index] = edge.target_idx;
            } else {
                adjacency_indices.push(edge.target_idx);
            }
            adj.count += 1;
        }

        // Update buffers with optimized data
        self.update_buffers(&gpu_nodes, &gpu_edges, &adjacency, &adjacency_indices)?;

        debug!("Graph data set successfully with optimized adjacency structure");
        Ok(())
    }

    /// Updates the force-directed graph parameters
    pub fn set_force_directed_params(&mut self, params: &SimulationParams) -> Result<(), Error> {
        debug!("Updating force-directed parameters: {:?}", params);
        self.simulation_params = *params;
        self.queue.write_buffer(&self.simulation_params_buffer, 0, bytemuck::cast_slice(&[self.simulation_params]));
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
        self.queue.write_buffer(&self.fisheye_params_buffer, 0, bytemuck::cast_slice(&[self.fisheye_params]));
        Ok(())
    }

    /// Computes forces with optimized dispatch
    pub fn compute_forces(&self) -> Result<(), Error> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Layout Computation Encoder"),
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
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Fisheye distortion pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fisheye Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.fisheye_pipeline);
            cpass.set_bind_group(0, &self.fisheye_bind_group, &[]);
            
            let workgroup_count = (self.num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
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
            let aligned_size = ((num_elements * element_size + 255) & !255) as usize; // Align to 256 bytes
            
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
                for byte in view.iter_mut() {
                    *byte = 0;
                }
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

        debug!("Buffer sizes - Node: {}, Edge: {}, Adjacency: {}, Index: {}", 
               node_size, edge_size, adjacency_size, index_size);

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
