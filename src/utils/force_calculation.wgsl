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

// Physics constants - aligned with settings.toml
const WORKGROUP_SIZE: u32 = 256;
const MAX_FORCE: f32 = 100.0;          // Increased for stronger forces
const MIN_DISTANCE: f32 = 5.0;         // Increased minimum distance
const CENTER_RADIUS: f32 = 250.0;      // Matches target_radius from settings
const MAX_VELOCITY: f32 = 20.0;        // Increased for faster movement
const NATURAL_LENGTH: f32 = 120.0;     // Matches natural_length from settings
const BOUNDARY_LIMIT: f32 = 600.0;     // Matches boundary_limit from settings

// Validation functions
fn is_valid_float(x: f32) -> bool {
    return x == x && abs(x) < 1e10;  // Check for NaN and infinity
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return is_valid_float(v.x) && is_valid_float(v.y) && is_valid_float(v.z);
}

fn clamp_force(force: vec3<f32>) -> vec3<f32> {
    let magnitude = length(force);
    if (magnitude > MAX_FORCE) {
        return (force / magnitude) * MAX_FORCE;
    }
    return force;
}

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
    
    // Combined spring and attraction forces with weight scaling
    let spring_force = params.spring_strength * weight * (distance - NATURAL_LENGTH);
    let attraction_force = params.attraction_strength * weight * distance;
    
    let total_force = normalize(displacement) * (spring_force + attraction_force);
    return clamp_force(total_force);
}

// Calculate repulsion force between nodes
fn calculate_repulsion_force(pos1: vec3<f32>, pos2: vec3<f32>, mass1: f32, mass2: f32) -> vec3<f32> {
    let displacement = pos2 - pos1;
    let distance_sq = dot(displacement, displacement);
    
    if (distance_sq < MIN_DISTANCE * MIN_DISTANCE) {
        return normalize(displacement) * -MAX_FORCE;
    }
    
    // Coulomb-like repulsion scaled by masses and adjusted for graph size
    let force_magnitude = -params.repulsion_strength * mass1 * mass2 / max(distance_sq, 0.1);
    let force = normalize(displacement) * min(abs(force_magnitude), MAX_FORCE) * sign(force_magnitude);
    return clamp_force(force);
}

// Calculate center gravity force
fn calculate_center_force(position: vec3<f32>) -> vec3<f32> {
    let to_center = -position;
    let distance = length(to_center);
    
    if (distance > CENTER_RADIUS) {
        // Stronger centering force during initial layout
        let center_strength = select(0.1, 0.2, params.is_initial_layout == 1u);
        let force = normalize(to_center) * center_strength * (distance - CENTER_RADIUS);
        return clamp_force(force);
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
    
    // Validate input node data
    if (!is_valid_float3(get_position(node)) || !is_valid_float3(get_velocity(node))) {
        // Reset invalid node to origin
        node.x = 0.0;
        node.y = 0.0;
        node.z = 0.0;
        node.vx = 0.0;
        node.vy = 0.0;
        node.vz = 0.0;
        nodes_buffer.nodes[node_id] = node;
        return;
    }

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
            
            // Validate other node
            if (!is_valid_float3(get_position(other_node))) {
                continue;
            }
            
            let other_mass = decode_mass(other_node.mass);
            let other_pos = get_position(other_node);
            
            // Accumulate spring force
            let spring_force = calculate_spring_force(
                node_pos,
                other_pos,
                node_mass,
                other_mass,
                edge.weight
            );
            total_force += spring_force;
        }
    }

    // Calculate repulsion forces with all other nodes
    for (var i = 0u; i < n_nodes; i = i + 1u) {
        if (i != node_id) {
            let other_node = nodes_buffer.nodes[i];
            
            // Validate other node
            if (!is_valid_float3(get_position(other_node))) {
                continue;
            }
            
            let other_mass = decode_mass(other_node.mass);
            let other_pos = get_position(other_node);
            
            let repulsion_force = calculate_repulsion_force(
                node_pos,
                other_pos,
                node_mass,
                other_mass
            );
            total_force += repulsion_force;
        }
    }

    // Add center gravity force
    let center_force = calculate_center_force(node_pos);
    total_force += center_force;

    // Scale forces based on layout phase
    let force_scale = select(1.0, 2.0, params.is_initial_layout == 1u);
    total_force *= force_scale;
    total_force = clamp_force(total_force);

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
        vec3<f32>(-BOUNDARY_LIMIT),
        vec3<f32>(BOUNDARY_LIMIT)
    );

    // Validate final values
    if (!is_valid_float3(bounded_pos) || !is_valid_float3(velocity)) {
        // Reset to origin if invalid
        node.x = 0.0;
        node.y = 0.0;
        node.z = 0.0;
        node.vx = 0.0;
        node.vy = 0.0;
        node.vz = 0.0;
    } else {
        // Update node with new values
        node.x = bounded_pos.x;
        node.y = bounded_pos.y;
        node.z = bounded_pos.z;
        node.vx = velocity.x;
        node.vy = velocity.y;
        node.vz = velocity.z;
    }

    nodes_buffer.nodes[node_id] = node;
}
