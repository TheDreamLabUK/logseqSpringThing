use wgpu::{Device, Queue, Buffer, BindGroup, ComputePipeline, InstanceDescriptor};
use wgpu::util::DeviceExt; // Needed for create_buffer_init
use std::io::Error;
use log::{error, info}; // For logging
use crate::models::graph::GraphData;
use crate::models::node::{Node, GPUNode};
use crate::models::edge::GPUEdge; // Import GPUEdge
use crate::models::simulation_params::SimulationParams; // Import SimulationParams
use rand::Rng; // Import rand for random number generation

const INITIAL_BUFFER_SIZE: u64 = 1024; // 1KB initial size

/// Manages GPU computations for force-directed graph layout using WebGPU.
///
/// This struct encapsulates the WebGPU resources and logic for performing
/// force calculations and updating node positions on the GPU, accelerating
/// the graph layout process.
pub struct GPUCompute {
    /// The WebGPU device representing the GPU.
    device: Device,
    /// The WebGPU queue for submitting commands to the GPU.
    queue: Queue,
    /// Buffer storing node data (positions, velocities) on the GPU.
    nodes_buffer: Buffer,
    /// Buffer storing edge data (source, target_node, weight) on the GPU.
    edges_buffer: Buffer,
    /// Buffer storing simulation parameters (repulsion, attraction, etc.).
    simulation_params_buffer: Buffer,
    /// Bind group linking the buffers to the compute shader.
    bind_group: BindGroup,
    /// Compute pipeline for the force calculation shader.
    compute_pipeline: ComputePipeline,
    /// Number of nodes in the graph.
    num_nodes: u32,
    /// Number of edges in the graph.
    num_edges: u32,
}

impl GPUCompute {
    /// Creates a new instance of GPUCompute.
    ///
    /// Initializes the WebGPU environment, loads shaders, creates buffers,
    /// and sets up the compute pipeline for force calculations.
    pub async fn new() -> Result<Self, Error> {
        // Instantiates instance of an adapter
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::new(std::io::ErrorKind::Other, "Request adapter error"))?;

        // Request device and queue from adapter
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Load the force calculation shader from the embedded WGSL source code
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Force Calculation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("force_calculation.wgsl").into()),
        });

        // Create a bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create simulation parameters buffer
        let simulation_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params Buffer"),
            contents: bytemuck::cast_slice(&[SimulationParams::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create a compute pipeline for force calculations
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &cs_module,
            entry_point: "main",
        });

        // Create nodes and edges buffers with initial size
        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let edges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edges Buffer"),
            size: INITIAL_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
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

        Ok(Self {
            device,
            queue,
            nodes_buffer,
            edges_buffer,
            simulation_params_buffer,
            bind_group,
            compute_pipeline,
            num_nodes: 0,
            num_edges: 0,
        })
    }

    /// Sets the graph data for GPU computation.
    ///
    /// Updates the nodes and edges buffers on the GPU with the provided
    /// graph data. Converts the graph data to a GPU-friendly format
    /// (using `GPUNode` and `GPUEdge`).
    pub fn set_graph_data(&mut self, graph: &GraphData) -> Result<(), Error> {
        self.num_nodes = graph.nodes.len() as u32;
        self.num_edges = graph.edges.len() as u32;

        // Convert Node to GPUNode with random initial positions
        let mut rng = rand::thread_rng();
        let gpu_nodes: Vec<GPUNode> = graph.nodes.iter().map(|node| {
            let mut gpu_node = node.to_gpu_node();
            gpu_node.x = rng.gen_range(-1.0..1.0);
            gpu_node.y = rng.gen_range(-1.0..1.0);
            gpu_node.z = rng.gen_range(-1.0..1.0);
            gpu_node
        }).collect();

        // Convert Edge to GPUEdge
        let gpu_edges: Vec<GPUEdge> = graph.edges.iter().map(|edge| edge.to_gpu_edge(&graph.nodes)).collect();

        // Update nodes buffer
        let nodes_data = bytemuck::cast_slice(&gpu_nodes);
        if (nodes_data.len() as u64) > self.nodes_buffer.size() {
            // If the new data is larger than the current buffer, create a new buffer
            self.nodes_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Nodes Buffer"),
                contents: nodes_data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            });
        } else {
            // Otherwise, just update the existing buffer
            self.queue.write_buffer(&self.nodes_buffer, 0, nodes_data);
        }

        // Update edges buffer
        let edges_data = bytemuck::cast_slice(&gpu_edges);
        if (edges_data.len() as u64) > self.edges_buffer.size() {
            // If the new data is larger than the current buffer, create a new buffer
            self.edges_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Edges Buffer"),
                contents: edges_data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            // Otherwise, just update the existing buffer
            self.queue.write_buffer(&self.edges_buffer, 0, edges_data);
        }

        // Update bind group with new buffers
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.compute_pipeline.get_bind_group_layout(0),
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
                    resource: self.simulation_params_buffer.as_entire_binding(),
                },
            ],
        });

        info!("Updated GPU buffers: {} nodes, {} edges", self.num_nodes, self.num_edges);
        Ok(())
    }

    /// Performs force calculations on the GPU.
    ///
    /// Dispatches the force calculation compute shader, which updates the
    /// velocities of the nodes based on the forces acting on them.
    pub fn compute_forces(&self) -> Result<(), Error> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(
                (self.num_nodes + 63) / 64,
                1,
                1,
            );
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Retrieves the updated node positions from the GPU.
    ///
    /// Copies the updated node positions from the GPU buffer to a staging
    /// buffer, maps the staging buffer to CPU memory, and returns the node
    /// data as a vector of `Node` structs.
    pub async fn get_updated_positions(&self) -> Result<Vec<Node>, Error> {
        let buffer_size = std::mem::size_of::<GPUNode>() * self.num_nodes as usize;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Retrieval Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.nodes_buffer, 0, &staging_buffer, 0, buffer_size as u64);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        // Await the receiver to get the mapping result.
        receiver.receive().await.unwrap().map_err(|e| {
            error!("Failed to map staging buffer: {:?}", e);
            Error::new(std::io::ErrorKind::Other, "Failed to map staging buffer")
        })?;

        let data = buffer_slice.get_mapped_range();
        let gpu_nodes: Vec<GPUNode> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Convert GPUNode back to Node
        let nodes: Vec<Node> = gpu_nodes.iter().map(|gpu_node| {
            let mut node = Node::default();
            node.update_from_gpu_node(gpu_node);
            node
        }).collect();

        Ok(nodes)
    }
}