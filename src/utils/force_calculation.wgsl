// force_calculation.wgsl
[[block]]
struct Graph {
    [[offset(0)]] nodes: array<vec4<f32>>;
    [[offset(0)]] edges: array<vec4<f32>>;
};

[[group(0), binding(0)]] var<storage, read_write> graph: Graph;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&graph.nodes);

    if (node_id < n_nodes) {
        let node = graph.nodes[node_id];
        var force = vec2<f32>(0.0, 0.0);

        // Repulsive force
        for (var i = 0u; i < n_nodes; i = i + 1u) {
            if (i != node_id) {
                let other_node = graph.nodes[i];
                let direction = vec2<f32>(other_node.x - node.x, other_node.y - node.y);
                let distance_sq = dot(direction, direction) + 0.01;
                let repulsive_force = 1.0 / distance_sq;
                force = force + normalize(direction) * repulsive_force;
            }
        }

        // Attractive force (edges)
        let n_edges = arrayLength(&graph.edges);
        for (var j = 0u; j < n_edges; j = j + 1u) {
            let edge = graph.edges[j];
            if (edge.x == f32(node_id) || edge.y == f32(node_id)) {
                let other_id = if edge.x == f32(node_id) { u32(edge.y) } else { u32(edge.x) };
                let other_node = graph.nodes[other_id];
                let direction = vec2<f32>(other_node.x - node.x, other_node.y - node.y);
                force = force + normalize(direction) * edge.z; // edge.z represents weight/strength
            }
        }
        
        graph.nodes[node_id].z = force.x;
        graph.nodes[node_id].w = force.y;
    }
}