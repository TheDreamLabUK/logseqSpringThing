use wgpu::{Device, Queue, Buffer, BindGroup, ComputePipeline, InstanceDescriptor};
use wgpu::util::DeviceExt;
use std::io::Error;
use log::{debug, info};
use crate::models::graph::GraphData;
use crate::models::node::GPUNode;
use crate::models::simulation_params::SimulationParams;
use crate::models::position_update::NodePositionVelocity;
use futures::channel::oneshot;

// Constants for buffer management and computation
const WORKGROUP_SIZE: u32 = 256;
const INITIAL_BUFFER_SIZE: u64 = 4 * 1024 * 1024;  // Increased to 4MB for larger quantized values
const BUFFER_ALIGNMENT: u64 = 256;  // Required GPU memory alignment
const EDGE_SIZE: u64 = 32;  // Size of Edge struct (must match WGSL)
const NODE_SIZE: u64 = 32;  // Increased from 28 to 32 for better alignment with larger values
const MAX_NODES: u32 = 1_000_000;  // Safety limit for number of nodes

// Position update constants
const POSITION_BUFFER_SIZE: u64 = 32;  // Increased from 24 to 32 for better alignment

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
    nodes_staging_buffer: Buffer,
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
    is_initialized: bool,
    position_update_buffer: Buffer,
    position_staging_buffer: Buffer,
    position_pipeline: ComputePipeline,
    position_bind_group: BindGroup,
}

impl GPUCompute {
    /// Creates a new instance of GPUCompute with initialized GPU resources
    pub async fn new(graph: &GraphData) -> Result<Self, Error> {
        debug!("Initializing GPU compute capabilities with {} nodes", graph.nodes.len());
        
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
                        min_binding_size: None,
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
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation parameters (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        // Create compute pipelines with updated descriptors
        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Force Directed Graph Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Force Pipeline Layout"),
                bind_group_layouts: &[&force_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &force_module,
            entry_point: Some("compute_main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let fisheye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fisheye Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fisheye Pipeline Layout"),
                bind_group_layouts: &[&fisheye_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &fisheye_module,
            entry_point: Some("compute_main"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Create buffers
        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let nodes_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes Staging Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edges Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let adjacency_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjacency Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let adjacency_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjacency List Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let simulation_params = SimulationParams::default();
        let gpu_params = simulation_params.to_gpu_params();
        let simulation_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params Buffer"),
            contents: bytemuck::cast_slice(&[gpu_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fisheye_params = FisheyeParams::default();
        let fisheye_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fisheye Params Buffer"),
            contents: bytemuck::cast_slice(&[fisheye_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create dedicated position buffers
        let position_update_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Update Buffer"),
            size: (MAX_NODES as u64) * POSITION_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let position_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Staging Buffer"),
            size: (MAX_NODES as u64) * POSITION_BUFFER_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create position update shader module
        let position_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Position Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("update_positions.wgsl").into()),
        });

        // Create position bind group layout
        let position_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Position Update Layout"),
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
            ],
        });

        // Create position pipeline
        let position_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Position Update Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Position Pipeline Layout"),
                bind_group_layouts: &[&position_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &position_module,
            entry_point: Some("update_positions"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Create position bind group
        let position_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Position Update Bind Group"),
            layout: &position_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_update_buffer.as_entire_binding(),
                },
            ],
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

        let num_nodes = graph.nodes.len() as u32;
        if num_nodes == 0 {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot initialize GPU compute with 0 nodes"
            ));
        }

        // Create a mutable instance
        let mut instance = Self {
            device,
            queue,
            nodes_buffer,
            nodes_staging_buffer,
            edges_buffer,
            adjacency_buffer,
            adjacency_list_buffer,
            simulation_params_buffer,
            fisheye_params_buffer,
            force_bind_group,
            fisheye_bind_group,
            force_pipeline,
            fisheye_pipeline,
            num_nodes,
            num_edges: graph.edges.len() as u32,
            simulation_params,
            fisheye_params,
            is_initialized: false,
            position_update_buffer,
            position_staging_buffer,
            position_pipeline,
            position_bind_group,
        };

        // Initialize with graph data
        instance.update_graph_data(graph)?;

        Ok(instance)
    }

    /// Updates the simulation parameters
    pub fn update_simulation_params(&mut self, params: &SimulationParams) -> Result<(), Error> {
        debug!("Updating simulation parameters: {:?}", params);
        
        self.simulation_params = params.clone();
        
        // Convert to GPU-compatible parameters
        let gpu_params = params.to_gpu_params();
        
        self.queue.write_buffer(
            &self.simulation_params_buffer,
            0,
            bytemuck::cast_slice(&[gpu_params])
        );

        debug!("Simulation parameters updated successfully");
        Ok(())
    }

    /// Updates the graph data in GPU buffers
    pub fn update_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        debug!("Updating GPU graph data: {} nodes, {} edges", 
            graph.nodes.len(), graph.edges.len());

        // Convert nodes to GPU format
        let gpu_nodes: Vec<GPUNode> = graph.nodes.iter()
            .map(|node| node.to_gpu_node())
            .collect();

        // Write nodes to buffer
        self.queue.write_buffer(
            &self.nodes_buffer,
            0,
            bytemuck::cast_slice(&gpu_nodes)
        );

        // Update counts
        self.num_nodes = graph.nodes.len() as u32;
        self.num_edges = graph.edges.len() as u32;
        self.is_initialized = true;

        debug!("Graph data updated successfully");
        Ok(())
    }

    /// Updates node positions from client data
    pub async fn update_positions(&mut self, nodes: &[NodePositionVelocity]) -> Result<(), Error> {
        debug!("Updating node positions for {} nodes", nodes.len());

        // Write position data to buffer
        self.queue.write_buffer(
            &self.position_update_buffer,
            0,
            bytemuck::cast_slice(nodes)
        );

        // Run position validation shader
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Position Update Encoder"),
            }
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Position Validation Pass"),
                    timestamp_writes: None,
                }
            );

            compute_pass.set_pipeline(&self.position_pipeline);
            compute_pass.set_bind_group(0, &self.position_bind_group, &[]);
            
            let workgroups = (self.num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            debug!("Dispatching position validation shader: {} workgroups", workgroups);
            
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy validated positions and velocities to node buffer
        encoder.copy_buffer_to_buffer(
            &self.position_update_buffer,
            0,
            &self.nodes_buffer,
            0,
            (self.num_nodes as u64) * std::mem::size_of::<NodePositionVelocity>() as u64,
        );

        self.queue.submit(Some(encoder.finish()));
        debug!("Position update complete");

        Ok(())
    }

    /// Fast path for position updates from client
    pub async fn update_positions_binary(&mut self, binary_data: &[u8], is_initial_layout: bool) -> Result<(), Error> {
        // Verify data length (24 bytes per node - position + velocity)
        let expected_size = self.num_nodes as usize * 24;
        if binary_data.len() != expected_size {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid position data length: expected {}, got {}", 
                    expected_size, binary_data.len())
            ));
        }

        // Update simulation params based on layout type
        let mut gpu_params = self.simulation_params.to_gpu_params();
        if is_initial_layout {
            // Adjust parameters for initial layout
            gpu_params.spring_strength *= 2.0;
            gpu_params.repulsion *= 2.0;
        }
        
        self.queue.write_buffer(
            &self.simulation_params_buffer,
            0,
            bytemuck::cast_slice(&[gpu_params])
        );

        // Write position data to buffer
        self.queue.write_buffer(
            &self.position_update_buffer,
            0,
            binary_data
        );

        debug!("Position data written to GPU buffer");

        // Run position validation shader
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Position Update Encoder"),
            }
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Position Validation Pass"),
                    timestamp_writes: None,
                }
            );

            compute_pass.set_pipeline(&self.position_pipeline);
            compute_pass.set_bind_group(0, &self.position_bind_group, &[]);
            
            let workgroups = (self.num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            debug!("Dispatching position validation shader: {} workgroups", workgroups);
            
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy validated positions and velocities to node buffer
        encoder.copy_buffer_to_buffer(
            &self.position_update_buffer,
            0,
            &self.nodes_buffer,
            0,
            (self.num_nodes as u64) * 24,
        );

        self.queue.submit(Some(encoder.finish()));
        debug!("Position update complete");

        Ok(())
    }

    /// Performs one step of the force-directed layout computation
    pub fn step(&mut self) -> Result<(), Error> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Force Compute Encoder"),
        });

        debug!("Starting force computation step: {} nodes, {} edges", 
            self.num_nodes, self.num_edges);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Force Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.force_pipeline);
            compute_pass.set_bind_group(0, &self.force_bind_group, &[]);
            
            let workgroups = (self.num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            debug!("Dispatching force computation: {} workgroups", workgroups);
            
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        debug!("Force computation step complete");
        
        Ok(())
    }

    /// Retrieves current node positions from GPU
    pub async fn get_node_positions(&self) -> Result<Vec<GPUNode>, Error> {
        debug!("Reading node positions from GPU: {} nodes", self.num_nodes);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Node Position Readback"),
        });

        let buffer_size = (self.num_nodes as u64) * std::mem::size_of::<GPUNode>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.nodes_buffer,
            0,
            &self.nodes_staging_buffer,
            0,
            buffer_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.nodes_staging_buffer.slice(..);
        let (sender, receiver) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        receiver.await.unwrap().map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let data = buffer_slice.get_mapped_range();
        let nodes: Vec<GPUNode> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.nodes_staging_buffer.unmap();

        debug!("Node positions read successfully. Sample positions: {:?}", 
            nodes.iter().take(3).map(|n| (n.x, n.y, n.z)).collect::<Vec<_>>());

        Ok(nodes)
    }

    /// Updates fisheye distortion parameters
    pub fn update_fisheye_params(&mut self, enabled: bool, strength: f32, focus_point: [f32; 3], radius: f32) -> Result<(), Error> {
        debug!("Updating fisheye parameters: enabled={}, strength={}, focus={:?}, radius={}", 
            enabled, strength, focus_point, radius);

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

        debug!("Fisheye parameters updated successfully");
        Ok(())
    }
}
