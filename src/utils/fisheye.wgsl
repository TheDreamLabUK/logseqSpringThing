// Structure representing a node with position, velocity, mass and padding
struct Node {
    position: vec3<f32>,  // 12 bytes
    velocity: vec3<f32>,  // 12 bytes
    mass: f32,           // 4 bytes
    padding1: u32,       // 4 bytes
}

// Buffer containing all nodes
struct NodesBuffer {
    nodes: array<Node>,
}

// Fisheye distortion parameters
struct FisheyeParams {
    enabled: u32,
    strength: f32,
    focus_point: vec3<f32>,
    radius: f32,
}

// Nodes buffer for reading and writing node data
@group(0) @binding(0) var<storage, read_write> nodes_buffer: NodesBuffer;

// Uniform buffer containing fisheye parameters
@group(0) @binding(1) var<uniform> fisheye_params: FisheyeParams;

// Constants
const PI: f32 = 3.14159265359;

// Apply fisheye distortion to a position
fn apply_fisheye(position: vec3<f32>) -> vec3<f32> {
    if (fisheye_params.enabled == 0u) {
        return position;
    }

    // Calculate vector from focus point to position
    let offset = position - fisheye_params.focus_point;
    let distance = length(offset);
    
    if (distance == 0.0 || distance > fisheye_params.radius) {
        return position;
    }

    // Normalize distance to [0,1] range within radius
    let normalized_distance = distance / fisheye_params.radius;
    
    // Calculate distortion factor using atan function
    // This creates a smooth falloff that preserves detail in the center
    let distortion = atan(normalized_distance * fisheye_params.strength) / 
                    (normalized_distance * fisheye_params.strength);
    
    // Apply distortion
    return fisheye_params.focus_point + offset * distortion;
}

// Main compute shader function
@compute @workgroup_size(256)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&nodes_buffer.nodes);
    
    if (node_id >= n_nodes) {
        return;
    }

    var node = nodes_buffer.nodes[node_id];
    
    // Apply fisheye distortion to node position
    node.position = apply_fisheye(node.position);
    
    // Write back to buffer
    nodes_buffer.nodes[node_id] = node;
}
