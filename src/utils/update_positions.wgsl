struct Node {
    position: vec3<f32>,  // 12 bytes
    velocity: vec3<f32>,  // 12 bytes
    mass: f32,           // 4 bytes
    padding1: u32,       // 4 bytes
}

struct NodesBuffer {
    nodes: array<Node>,
}

@group(0) @binding(1) var<uniform> delta_time: f32;
@group(0) @binding(0) var<storage, read_write> nodes_buffer: NodesBuffer;

const MIN_DISTANCE: f32 = 0.1;

fn is_nan(x: f32) -> bool {
    return x != x;
}

fn is_inf(x: f32) -> bool {
    return abs(x) >= 3.402823466e+38;
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return !(is_nan(v.x) || is_nan(v.y) || is_nan(v.z) || 
             is_inf(v.x) || is_inf(v.y) || is_inf(v.z));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&nodes_buffer.nodes);

    if (node_id >= n_nodes) { return; }

    var node = nodes_buffer.nodes[node_id];
    
    // Validate position and velocity
    if (!is_valid_float3(node.position)) {
        node.position = vec3<f32>(0.0);
    }
    if (!is_valid_float3(node.velocity)) {
        node.velocity = vec3<f32>(0.0);
    }

    // Update position with velocity and delta time
    node.position = node.position + node.velocity * delta_time;

    // Write back to buffer
    nodes_buffer.nodes[node_id] = node;
}
