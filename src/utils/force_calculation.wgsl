// Structure representing a node with position and velocity.
struct Node {
    position: vec3<f32>,
    velocity: vec3<f32>,
}

// Structure representing an edge between two nodes.
struct Edge {
    source: u32,
    target_node: u32,
    weight: f32,
}

// Buffer containing all nodes.
struct NodesBuffer {
    nodes: array<Node>,
}

// Buffer containing all edges.
struct EdgesBuffer {
    edges: array<Edge>,
}

// Parameters for the simulation.
struct SimulationParams {
    iterations: u32,
    repulsion_strength: f32,
    attraction_strength: f32,
    damping: f32,
    padding: u32,
}

// Uniform buffer containing simulation parameters.
@group(0) @binding(2) var<uniform> simulation_params: SimulationParams;

// Nodes buffer for reading and writing node data.
@group(0) @binding(0) var<storage, read_write> nodes_buffer: NodesBuffer;

// Edges buffer for reading edge data.
@group(0) @binding(1) var<storage, read> edges_buffer: EdgesBuffer;

fn is_nan(x: f32) -> bool {
    return x != x;
}

fn is_inf(x: f32) -> bool {
    let max_float = 3.402823466e+38;  // Maximum finite value for f32
    return abs(x) >= max_float;
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return !(is_nan(v.x) || is_nan(v.y) || is_nan(v.z) || is_inf(v.x) || is_inf(v.y) || is_inf(v.z));
}

fn clamp_magnitude(v: vec3<f32>, max_magnitude: f32) -> vec3<f32> {
    let magnitude_sq = dot(v, v);
    if (magnitude_sq > max_magnitude * max_magnitude) {
        return normalize(v) * max_magnitude;
    }
    return v;
}

// Main compute shader function.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&nodes_buffer.nodes);

    if (node_id < n_nodes) {
        var node = nodes_buffer.nodes[node_id];
        var force = vec3<f32>(0.0, 0.0, 0.0);

        if (!is_valid_float3(node.position)) {
            node.position = vec3<f32>(0.0, 0.0, 0.0);
        }

        // Add a weak centering force
        let center_direction = -node.position;
        let center_distance = length(center_direction);
        if (center_distance > 10.0) {
            force = force + normalize(center_direction) * (center_distance - 10.0) * 0.01;
        }

        // Repulsive force between nodes with smoother falloff
        for (var i = 0u; i < n_nodes; i = i + 1u) {
            if (i != node_id) {
                let other_node = nodes_buffer.nodes[i];
                let direction = node.position - other_node.position;
                let distance = length(direction);
                
                // Check for zero distance to avoid division by zero
                if (distance > 0.0001) {
                    // Use smoother repulsion falloff
                    let repulsive_force = simulation_params.repulsion_strength * exp(-distance / 10.0);
                    force = force + normalize(direction) * repulsive_force;
                }
            }
        }

        // Attractive force along edges with distance limiting
        let n_edges = arrayLength(&edges_buffer.edges);
        for (var j = 0u; j < n_edges; j = j + 1u) {
            let edge = edges_buffer.edges[j];
            if (edge.source == node_id || edge.target_node == node_id) {
                let other_id = select(edge.source, edge.target_node, edge.source == node_id);
                let other_node = nodes_buffer.nodes[other_id];
                let direction = other_node.position - node.position;
                let distance = length(direction);
                if (distance > 0.0001) {
                    // Use logarithmic attraction to prevent excessive force at large distances
                    let attractive_force = simulation_params.attraction_strength * edge.weight * log(distance + 1.0);
                    force = force + normalize(direction) * attractive_force;
                }
            }
        }

        // Clamp maximum force magnitude
        force = clamp_magnitude(force, 1.0);

        // Apply damping to velocity
        node.velocity = (node.velocity + force) * simulation_params.damping;
        
        // Clamp maximum velocity
        node.velocity = clamp_magnitude(node.velocity, 2.0);

        // Update node's position
        node.position = node.position + node.velocity;

        // Ensure final position and velocity are valid
        if (!is_valid_float3(node.position)) {
            node.position = vec3<f32>(0.0, 0.0, 0.0);
        }
        if (!is_valid_float3(node.velocity)) {
            node.velocity = vec3<f32>(0.0, 0.0, 0.0);
        }

        // Write back to the buffer
        nodes_buffer.nodes[node_id] = node;
    }
}
