// Structure representing a node with position, velocity, and mass
struct Node {
    position: vec3<f32>,  // 12 bytes
    velocity: vec3<f32>,  // 12 bytes
    mass: f32,           // 4 bytes
    padding1: u32,        // 4 bytes
}

// Structure representing an edge between two nodes
struct Edge {
    source: u32,      // 4 bytes
    target_idx: u32,  // 4 bytes
    weight: f32,      // 4 bytes
    padding1: u32,    // 4 bytes
    padding2: u32,    // 4 bytes
    padding3: u32,    // 4 bytes
    padding4: u32,    // 4 bytes
    padding5: u32,    // 4 bytes
}

// Structure representing adjacency information
struct Adjacency {
    offset: u32,
    count: u32,
}

// Buffer containing all nodes
struct NodesBuffer {
    nodes: array<Node>,
}

// Buffer containing all edges
struct EdgesBuffer {
    edges: array<Edge>,
}

// Buffer containing adjacency information
struct AdjacencyBuffer {
    adjacency: array<Adjacency>,
}

// Buffer containing adjacency list indices
struct AdjacencyListBuffer {
    indices: array<u32>,
}

// Parameters for the simulation
struct SimulationParams {
    iterations: u32,           // 4 bytes
    repulsion_strength: f32,   // 4 bytes
    attraction_strength: f32,  // 4 bytes
    damping: f32,             // 4 bytes
    padding1: u32,            // 4 bytes
    padding2: u32,            // 4 bytes
    padding3: u32,            // 4 bytes
    padding4: u32,            // 4 bytes - Total: 32 bytes, aligned to 16
}

// Bind groups for data access
@group(0) @binding(0) var<storage, read_write> nodes_buffer: NodesBuffer;
@group(0) @binding(1) var<storage, read> edges_buffer: EdgesBuffer;
@group(0) @binding(2) var<storage, read> adjacency_buffer: AdjacencyBuffer;
@group(0) @binding(3) var<storage, read> adjacency_list_buffer: AdjacencyListBuffer;
@group(0) @binding(4) var<uniform> params: SimulationParams;

// Constants for simulation
const WORKGROUP_SIZE: u32 = 256;
const MAX_FORCE: f32 = 100.0;
const MIN_DISTANCE: f32 = 0.1;
const GRID_DIM: u32 = 16;
const TOTAL_GRID_SIZE: u32 = 4096; // 16^3
const CENTER_FORCE_STRENGTH: f32 = 0.05;
const CENTER_RADIUS: f32 = 50.0;

// Grid cell structure for spatial partitioning
struct GridCell {
    start_idx: u32,
    count: u32,
}

// Shared workgroup memory for grid
var<workgroup> grid: array<GridCell, 4096>; // Using TOTAL_GRID_SIZE constant
var<workgroup> node_counts: array<atomic<u32>, 4096>; // Using TOTAL_GRID_SIZE constant

// Utility functions
fn get_grid_index(position: vec3<f32>) -> u32 {
    let grid_pos = vec3<u32>(
        u32(clamp((position.x + 100.0) * f32(GRID_DIM) / 200.0, 0.0, f32(GRID_DIM - 1))),
        u32(clamp((position.y + 100.0) * f32(GRID_DIM) / 200.0, 0.0, f32(GRID_DIM - 1))),
        u32(clamp((position.z + 100.0) * f32(GRID_DIM) / 200.0, 0.0, f32(GRID_DIM - 1)))
    );
    return grid_pos.x + grid_pos.y * GRID_DIM + grid_pos.z * GRID_DIM * GRID_DIM;
}

fn calculate_repulsion(pos1: vec3<f32>, pos2: vec3<f32>, mass1: f32, mass2: f32) -> vec3<f32> {
    let direction = pos1 - pos2;
    let distance_sq = dot(direction, direction);
    
    if (distance_sq < MIN_DISTANCE * MIN_DISTANCE) {
        return vec3<f32>(0.0);
    }
    
    let distance = sqrt(distance_sq);
    let force_magnitude = params.repulsion_strength * mass1 * mass2 / (distance_sq + MIN_DISTANCE);
    return normalize(direction) * min(force_magnitude, MAX_FORCE);
}

fn calculate_attraction(pos1: vec3<f32>, pos2: vec3<f32>, weight: f32) -> vec3<f32> {
    let direction = pos2 - pos1;
    let distance_sq = dot(direction, direction);
    
    if (distance_sq < MIN_DISTANCE * MIN_DISTANCE) {
        return vec3<f32>(0.0);
    }
    
    let distance = sqrt(distance_sq);
    let force_magnitude = params.attraction_strength * weight * (distance - CENTER_RADIUS);
    return normalize(direction) * min(force_magnitude, MAX_FORCE);
}

// Main compute shader
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&nodes_buffer.nodes);

    if (node_id >= n_nodes) {
        return;
    }

    var node = nodes_buffer.nodes[node_id];
    var force = vec3<f32>(0.0);

    // Calculate grid cell for current node
    let grid_idx = get_grid_index(node.position);
    
    // Calculate repulsive forces from nearby grid cells
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_x = i32(grid_idx % GRID_DIM) + dx;
                let neighbor_y = i32((grid_idx / GRID_DIM) % GRID_DIM) + dy;
                let neighbor_z = i32(grid_idx / (GRID_DIM * GRID_DIM)) + dz;
                
                if (neighbor_x < 0 || neighbor_x >= i32(GRID_DIM) ||
                    neighbor_y < 0 || neighbor_y >= i32(GRID_DIM) ||
                    neighbor_z < 0 || neighbor_z >= i32(GRID_DIM)) {
                    continue;
                }
                
                let neighbor_idx = u32(neighbor_x) + 
                                 u32(neighbor_y) * GRID_DIM + 
                                 u32(neighbor_z) * GRID_DIM * GRID_DIM;
                
                // Calculate repulsion with nodes in neighboring cells
                for (var i = 0u; i < n_nodes; i = i + 1u) {
                    if (i == node_id || get_grid_index(nodes_buffer.nodes[i].position) != neighbor_idx) {
                        continue;
                    }
                    let other = nodes_buffer.nodes[i];
                    force += calculate_repulsion(node.position, other.position, node.mass, other.mass);
                }
            }
        }
    }

    // Calculate attractive forces using adjacency list
    let adj = adjacency_buffer.adjacency[node_id];
    for (var i = 0u; i < adj.count; i = i + 1u) {
        let target_idx = adjacency_list_buffer.indices[adj.offset + i];
        let other = nodes_buffer.nodes[target_idx];
        force += calculate_attraction(node.position, other.position, 1.0); // Using default weight of 1.0
    }

    // Apply centering force
    let to_center = -node.position;
    let center_distance = length(to_center);
    if (center_distance > CENTER_RADIUS) {
        force += normalize(to_center) * CENTER_FORCE_STRENGTH * (center_distance - CENTER_RADIUS);
    }

    // Update velocity with damping
    node.velocity = (node.velocity + force / node.mass) * params.damping;
    
    // Limit velocity magnitude
    let velocity_magnitude_sq = dot(node.velocity, node.velocity);
    if (velocity_magnitude_sq > MAX_FORCE * MAX_FORCE) {
        node.velocity = normalize(node.velocity) * MAX_FORCE;
    }

    // Update position
    node.position = node.position + node.velocity;

    // Store updated node
    nodes_buffer.nodes[node_id] = node;
}
