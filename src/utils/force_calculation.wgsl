struct Node {
    position: vec3<f32>,  // 12 bytes
    velocity: vec3<f32>,  // 12 bytes
    mass: f32,           // 4 bytes
    padding1: u32,        // 4 bytes
}

struct Edge {
    source: u32,      // 4 bytes
    target_idx: u32,  // 4 bytes
    weight: f32,      // 4 bytes
    padding1: u32,    // 4 bytes
}

struct NodesBuffer {
    nodes: array<Node>,
}

struct EdgesBuffer {
    edges: array<Edge>,
}

struct SimulationParams {
    iterations: u32,           // 4 bytes
    spring_strength: f32,      // 4 bytes
    damping: f32,             // 4 bytes
    is_initial_layout: u32,   // 4 bytes - Flag for initial vs interactive mode
    time_step: f32,           // 4 bytes - For smooth animation
    padding2: u32,            // 4 bytes
    padding3: u32,            // 4 bytes
    padding4: u32,            // 4 bytes - Total: 32 bytes, aligned to 16
}

// Constants optimized for stability and performance
const WORKGROUP_SIZE: u32 = 256;
const MAX_FORCE: f32 = 50.0;
const MIN_DISTANCE: f32 = 1.0;
const GRID_DIM: u32 = 32;
const TOTAL_GRID_SIZE: u32 = GRID_DIM * GRID_DIM * GRID_DIM;
const CENTER_FORCE_STRENGTH: f32 = 0.05;  // Reduced for more stability
const CENTER_RADIUS: f32 = 50.0;  // Reduced to match CPU implementation
const MAX_VELOCITY: f32 = 10.0;  // Matches CPU implementation
const NATURAL_LENGTH: f32 = 30.0;  // Reduced to match initial distribution
const REPULSION_SCALE: f32 = 10000.0;  // Matches CPU implementation

@group(0) @binding(0) var<storage, read_write> nodes_buffer: NodesBuffer;
@group(0) @binding(1) var<storage, read> edges_buffer: EdgesBuffer;
@group(0) @binding(2) var<uniform> params: SimulationParams;

// Utility functions
fn is_valid_float(x: f32) -> bool {
    return x == x && abs(x) < 1e10;
}

fn is_valid_vec3(v: vec3<f32>) -> bool {
    return is_valid_float(v.x) && is_valid_float(v.y) && is_valid_float(v.z);
}

fn clamp_vector(v: vec3<f32>, max_magnitude: f32) -> vec3<f32> {
    let magnitude_sq = dot(v, v);
    if (magnitude_sq > max_magnitude * max_magnitude) {
        return normalize(v) * max_magnitude;
    }
    return v;
}

fn clamp_position(pos: vec3<f32>) -> vec3<f32> {
    let max_coord = 100.0;  // Matches CPU implementation
    return vec3<f32>(
        clamp(pos.x, -max_coord, max_coord),
        clamp(pos.y, -max_coord, max_coord),
        clamp(pos.z, -max_coord, max_coord)
    );
}

fn get_grid_index(position: vec3<f32>) -> u32 {
    let grid_pos = vec3<u32>(
        u32(clamp((position.x + CENTER_RADIUS) * f32(GRID_DIM) / (2.0 * CENTER_RADIUS), 0.0, f32(GRID_DIM - 1))),
        u32(clamp((position.y + CENTER_RADIUS) * f32(GRID_DIM) / (2.0 * CENTER_RADIUS), 0.0, f32(GRID_DIM - 1))),
        u32(clamp((position.z + CENTER_RADIUS) * f32(GRID_DIM) / (2.0 * CENTER_RADIUS), 0.0, f32(GRID_DIM - 1)))
    );
    return grid_pos.x + grid_pos.y * GRID_DIM + grid_pos.z * GRID_DIM * GRID_DIM;
}

fn calculate_spring_force(pos1: vec3<f32>, pos2: vec3<f32>, mass1: f32, mass2: f32, is_connected: bool, weight: f32) -> vec3<f32> {
    if (!is_valid_vec3(pos1) || !is_valid_vec3(pos2)) {
        return vec3<f32>(0.0);
    }
    
    let direction = pos2 - pos1;
    let distance_sq = dot(direction, direction);
    
    if (distance_sq < MIN_DISTANCE * MIN_DISTANCE) {
        return normalize(direction) * MAX_FORCE;
    }
    
    let distance = sqrt(distance_sq);
    var force_magnitude: f32;
    
    if (is_connected) {
        // Connected nodes: Hooke's law with natural length
        force_magnitude = params.spring_strength * (distance - NATURAL_LENGTH) * weight;
    } else {
        // Unconnected nodes: Inverse square repulsion
        let repulsion_scale = select(1.0, 0.5, params.is_initial_layout == 0u);
        force_magnitude = -params.spring_strength * REPULSION_SCALE * repulsion_scale * mass1 * mass2 / (distance_sq + MIN_DISTANCE);
    }
    
    return clamp_vector(normalize(direction) * force_magnitude, MAX_FORCE);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&nodes_buffer.nodes);

    if (node_id >= n_nodes) {
        return;
    }

    var node = nodes_buffer.nodes[node_id];
    
    if (!is_valid_vec3(node.position) || !is_valid_vec3(node.velocity)) {
        node.position = vec3<f32>(0.0);
        node.velocity = vec3<f32>(0.0);
        nodes_buffer.nodes[node_id] = node;
        return;
    }

    var force = vec3<f32>(0.0);
    let grid_idx = get_grid_index(node.position);
    
    // Calculate forces with nearby nodes only
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
                
                for (var i = 0u; i < n_nodes; i = i + 1u) {
                    if (i == node_id || get_grid_index(nodes_buffer.nodes[i].position) != neighbor_idx) {
                        continue;
                    }
                    let other = nodes_buffer.nodes[i];
                    if (is_valid_vec3(other.position)) {
                        force += calculate_spring_force(node.position, other.position, node.mass, other.mass, false, 1.0);
                    }
                }
            }
        }
    }

    // Calculate forces with connected nodes
    let n_edges = arrayLength(&edges_buffer.edges);
    for (var i = 0u; i < n_edges; i = i + 1u) {
        let edge = edges_buffer.edges[i];
        if (edge.source == node_id) {
            let target_node = nodes_buffer.nodes[edge.target_idx];
            if (is_valid_vec3(target_node.position)) {
                force += calculate_spring_force(node.position, target_node.position, node.mass, target_node.mass, true, edge.weight);
            }
        }
    }

    // Apply centering force
    let to_center = -node.position;
    let center_distance = length(to_center);
    if (center_distance > CENTER_RADIUS) {
        force += normalize(to_center) * CENTER_FORCE_STRENGTH * (center_distance - CENTER_RADIUS);
    }

    // Update velocity and position with time step
    node.velocity = clamp_vector((node.velocity + force / node.mass) * params.damping, MAX_VELOCITY);
    node.position = clamp_position(node.position + node.velocity * params.time_step);

    if (!is_valid_vec3(node.position) || !is_valid_vec3(node.velocity)) {
        node.position = vec3<f32>(0.0);
        node.velocity = vec3<f32>(0.0);
    }

    nodes_buffer.nodes[node_id] = node;
}
