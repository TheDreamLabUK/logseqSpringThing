// Structure representing a node with position, velocity, and mass
struct Node {
    position: vec3<f32>,  // 12 bytes
    velocity: vec3<f32>,  // 12 bytes
    mass: f32,           // 4 bytes
    padding1: u32,        // 4 bytes
}

// Structure representing an edge between two nodes
struct Edge {
    source: u32,
    target_idx: u32,
    weight: f32,
    padding1: u32,
}

// Buffer containing all nodes
struct NodesBuffer {
    nodes: array<Node>,
}

// Buffer containing all edges
struct EdgesBuffer {
    edges: array<Edge>,
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
@group(0) @binding(2) var<uniform> params: SimulationParams;

// Constants for force calculations
const MIN_DISTANCE: f32 = 0.1;
const MAX_FORCE: f32 = 100.0;
const WORKGROUP_SIZE: u32 = 256;
const CENTER_FORCE_STRENGTH: f32 = 0.05;  // Strength of centering force
const CENTER_RADIUS: f32 = 50.0;         // Radius within which centering force is reduced

// Utility functions
fn is_nan(x: f32) -> bool {
    return x != x;
}

fn is_inf(x: f32) -> bool {
    let max_float = 3.402823466e+38;
    return abs(x) >= max_float;
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return !(is_nan(v.x) || is_nan(v.y) || is_nan(v.z) || 
             is_inf(v.x) || is_inf(v.y) || is_inf(v.z));
}

fn clamp_magnitude(v: vec3<f32>, max_magnitude: f32) -> vec3<f32> {
    let magnitude_sq = dot(v, v);
    if (magnitude_sq > max_magnitude * max_magnitude) {
        return normalize(v) * max_magnitude;
    }
    return v;
}

// Calculate center of mass for all nodes
fn calculate_center_of_mass() -> vec3<f32> {
    var com = vec3<f32>(0.0);
    var total_mass = 0.0;
    let n_nodes = arrayLength(&nodes_buffer.nodes);

    for (var i = 0u; i < n_nodes; i = i + 1u) {
        let node = nodes_buffer.nodes[i];
        com += node.position * node.mass;
        total_mass += node.mass;
    }

    if (total_mass > 0.0) {
        return com / total_mass;
    }
    return vec3<f32>(0.0);
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

    // Validate position
    if (!is_valid_float3(node.position)) {
        node.position = vec3<f32>(0.0);
    }

    // Calculate center of mass and apply centering force
    let com = calculate_center_of_mass();
    let to_origin = -com;
    let com_distance = length(to_origin);
    
    // Apply stronger centering force when center of mass is far from origin
    if (com_distance > 0.0001) {
        let center_force = normalize(to_origin) * CENTER_FORCE_STRENGTH * com_distance;
        force += center_force;
    }

    // Add individual node centering force (weaker, to maintain graph structure)
    let node_to_origin = -node.position;
    let node_distance = length(node_to_origin);
    if (node_distance > CENTER_RADIUS) {
        force += normalize(node_to_origin) * (node_distance - CENTER_RADIUS) * 0.01;
    }

    // Calculate repulsive forces
    for (var i = 0u; i < n_nodes; i = i + 1u) {
        if (i == node_id) {
            continue;
        }

        let other = nodes_buffer.nodes[i];
        let direction = node.position - other.position;
        let distance = length(direction);
        
        if (distance > 0.0001) {
            let repulsion = params.repulsion_strength * node.mass * other.mass * exp(-distance / 10.0);
            force += normalize(direction) * repulsion;
        }
    }

    // Calculate attractive forces
    let n_edges = arrayLength(&edges_buffer.edges);
    for (var i = 0u; i < n_edges; i = i + 1u) {
        let edge = edges_buffer.edges[i];
        if (edge.source == node_id || edge.target_idx == node_id) {
            let other_id = select(edge.source, edge.target_idx, edge.source == node_id);
            let other = nodes_buffer.nodes[other_id];
            let direction = other.position - node.position;
            let distance = length(direction);
            
            if (distance > 0.0001) {
                let attraction = params.attraction_strength * edge.weight * log(distance + 1.0);
                force += normalize(direction) * attraction;
            }
        }
    }

    // Limit force magnitude
    force = clamp_magnitude(force, MAX_FORCE);

    // Update velocity with damping
    node.velocity = (node.velocity + force / node.mass) * params.damping;
    node.velocity = clamp_magnitude(node.velocity, 2.0);

    // Update position
    node.position = node.position + node.velocity;

    // Validate final state
    if (!is_valid_float3(node.position)) {
        node.position = vec3<f32>(0.0);
    }
    if (!is_valid_float3(node.velocity)) {
        node.velocity = vec3<f32>(0.0);
    }

    // Store updated node
    nodes_buffer.nodes[node_id] = node;
}
