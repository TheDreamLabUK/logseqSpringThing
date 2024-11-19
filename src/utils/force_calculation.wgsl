// Node structure exactly matching Rust GPUNode memory layout (28 bytes total)
struct Node {
    x: f32, y: f32, z: f32,      // position (12 bytes)
    vx: f32, vy: f32, vz: f32,   // velocity (12 bytes)
    mass: u32,                    // mass in lower byte + flags + padding (4 bytes)
}

// Edge structure matching Rust GPUEdge layout
struct Edge {
    source: u32,      // 4 bytes
    target_idx: u32,  // 4 bytes (renamed from 'target' as it's a reserved keyword)
    weight: f32,      // 4 bytes
}

struct NodesBuffer {
    nodes: array<Node>,
}

struct EdgesBuffer {
    edges: array<Edge>,
}

// Matches Rust SimulationParams exactly
struct SimulationParams {
    iterations: u32,           // Range: 1-500
    spring_strength: f32,      // Range: 0.001-1.0
    repulsion_strength: f32,   // Range: 1.0-10000.0
    attraction_strength: f32,  // Range: 0.001-1.0
    damping: f32,             // Range: 0.5-0.95
    is_initial_layout: u32,   // bool converted to u32
    time_step: f32,           // Range: 0.1-1.0
    padding: u32,             // Explicit padding for alignment
}

@group(0) @binding(0) var<storage, read_write> nodes_buffer: NodesBuffer;
@group(0) @binding(1) var<storage, read> edges_buffer: EdgesBuffer;
@group(0) @binding(2) var<uniform> params: SimulationParams;

// Physics constants
const WORKGROUP_SIZE: u32 = 256;
const MAX_FORCE: f32 = 50.0;
const MIN_DISTANCE: f32 = 1.0;
const CENTER_RADIUS: f32 = 50.0;
const MAX_VELOCITY: f32 = 10.0;
const NATURAL_LENGTH: f32 = 30.0;

// Convert quantized mass (0-255 in lower byte) to float (0.0-2.0)
fn decode_mass(mass_packed: u32) -> f32 {
    return f32(mass_packed & 0xFFu) / 127.5;
}

// Get node position as vec3
fn get_position(node: Node) -> vec3<f32> {
    return vec3<f32>(node.x, node.y, node.z);
}

// Get node velocity as vec3
fn get_velocity(node: Node) -> vec3<f32> {
    return vec3<f32>(node.vx, node.vy, node.vz);
}

// Calculate spring force between connected nodes
fn calculate_spring_force(pos1: vec3<f32>, pos2: vec3<f32>, mass1: f32, mass2: f32, weight: f32) -> vec3<f32> {
    let displacement = pos2 - pos1;
    let distance = length(displacement);
    
    if (distance < MIN_DISTANCE) {
        return normalize(displacement) * MAX_FORCE;
    }
    
    // Combined spring and attraction forces
    let spring_force = params.spring_strength * weight * (distance - NATURAL_LENGTH);
    let attraction_force = params.attraction_strength * weight * distance;
    
    return normalize(displacement) * (spring_force + attraction_force);
}

// Calculate repulsion force between nodes
fn calculate_repulsion_force(pos1: vec3<f32>, pos2: vec3<f32>, mass1: f32, mass2: f32) -> vec3<f32> {
    let displacement = pos2 - pos1;
    let distance_sq = dot(displacement, displacement);
    
    if (distance_sq < MIN_DISTANCE * MIN_DISTANCE) {
        return normalize(displacement) * -MAX_FORCE;
    }
    
    // Coulomb-like repulsion scaled by masses
    let force_magnitude = -params.repulsion_strength * mass1 * mass2 / distance_sq;
    return normalize(displacement) * min(abs(force_magnitude), MAX_FORCE) * sign(force_magnitude);
}

// Calculate center gravity force
fn calculate_center_force(position: vec3<f32>) -> vec3<f32> {
    let to_center = -position;
    let distance = length(to_center);
    
    if (distance > CENTER_RADIUS) {
        // Stronger centering force during initial layout
        let center_strength = select(0.05, 0.1, params.is_initial_layout == 1u);
        return normalize(to_center) * center_strength * (distance - CENTER_RADIUS);
    }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&nodes_buffer.nodes);

    if (node_id >= n_nodes) {
        return;
    }

    var node = nodes_buffer.nodes[node_id];
    var total_force = vec3<f32>(0.0);
    let node_mass = decode_mass(node.mass);
    let node_pos = get_position(node);

    // Calculate forces from edges (bi-directional)
    let n_edges = arrayLength(&edges_buffer.edges);
    for (var i = 0u; i < n_edges; i = i + 1u) {
        let edge = edges_buffer.edges[i];
        if (edge.source == node_id || edge.target_idx == node_id) {
            let other_id = select(edge.source, edge.target_idx, edge.source == node_id);
            let other_node = nodes_buffer.nodes[other_id];
            let other_mass = decode_mass(other_node.mass);
            let other_pos = get_position(other_node);
            
            // Accumulate spring force
            total_force += calculate_spring_force(
                node_pos,
                other_pos,
                node_mass,
                other_mass,
                edge.weight
            );
        }
    }

    // Calculate repulsion forces with all other nodes
    for (var i = 0u; i < n_nodes; i = i + 1u) {
        if (i != node_id) {
            let other_node = nodes_buffer.nodes[i];
            let other_mass = decode_mass(other_node.mass);
            let other_pos = get_position(other_node);
            
            total_force += calculate_repulsion_force(
                node_pos,
                other_pos,
                node_mass,
                other_mass
            );
        }
    }

    // Add center gravity force
    total_force += calculate_center_force(node_pos);

    // Scale forces based on layout phase
    let force_scale = select(1.0, 2.0, params.is_initial_layout == 1u);
    total_force *= force_scale;

    // Update velocity with damping
    var velocity = get_velocity(node);
    velocity = (velocity + total_force * params.time_step) * params.damping;

    // Apply velocity limits
    let speed = length(velocity);
    if (speed > MAX_VELOCITY) {
        velocity = (velocity / speed) * MAX_VELOCITY;
    }

    // Update position
    let new_pos = node_pos + velocity * params.time_step;

    // Apply position bounds
    let bounded_pos = clamp(
        new_pos,
        vec3<f32>(-CENTER_RADIUS * 2.0),
        vec3<f32>(CENTER_RADIUS * 2.0)
    );

    // Update node with new values
    node.x = bounded_pos.x;
    node.y = bounded_pos.y;
    node.z = bounded_pos.z;
    node.vx = velocity.x;
    node.vy = velocity.y;
    node.vz = velocity.z;

    nodes_buffer.nodes[node_id] = node;
}
