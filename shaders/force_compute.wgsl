struct Node {
    position: vec3<f32>,
    velocity: vec3<f32>,
    mass: f32,
    padding: f32,
}

struct Edge {
    source_index: u32,
    target_index: u32,
    strength: f32,
    padding: f32,
}

struct SimParams {
    spring_k: f32,
    repulsion: f32,
    damping: f32,
    delta_time: f32,
}

@group(0) @binding(0) var<storage, read> nodes_in: array<Node>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read_write> nodes_out: array<Node>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&nodes_in)) {
        return;
    }

    var force = vec3<f32>(0.0, 0.0, 0.0);
    let node = nodes_in[index];

    // Calculate repulsion forces
    for (var i = 0u; i < arrayLength(&nodes_in); i++) {
        if (i == index) {
            continue;
        }

        let other = nodes_in[i];
        let diff = node.position - other.position;
        let dist = length(diff);
        let dist_squared = dist * dist + 0.1; // Add small offset to prevent division by zero
        
        // Coulomb's law for repulsion
        force += normalize(diff) * params.repulsion / dist_squared;
    }

    // Calculate spring forces
    for (var i = 0u; i < arrayLength(&edges); i++) {
        let edge = edges[i];
        if (edge.source_index == index || edge.target_index == index) {
            let other_index = select(edge.source_index, edge.target_index, edge.source_index == index);
            let other = nodes_in[other_index];
            
            let diff = node.position - other.position;
            let dist = length(diff);
            
            // Hooke's law for springs
            force -= normalize(diff) * params.spring_k * dist * edge.strength;
        }
    }

    // Update velocity and position using Verlet integration
    var new_node = node;
    new_node.velocity = node.velocity + force * params.delta_time;
    new_node.velocity *= params.damping;
    new_node.position += new_node.velocity * params.delta_time;

    nodes_out[index] = new_node;
} 