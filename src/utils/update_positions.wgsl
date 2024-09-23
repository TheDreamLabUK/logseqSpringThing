// update_positions.wgsl
[[block]]
struct Graph {
    [[offset(0)]] nodes: array<vec4<f32>>;
};

[[group(0), binding(0)]] var<storage, read_write> graph: Graph;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&graph.nodes);

    if (node_id < n_nodes) {
        graph.nodes[node_id].x = graph.nodes[node_id].x + graph.nodes[node_id].z;
        graph.nodes[node_id].y = graph.nodes[node_id].y + graph.nodes[node_id].w;
    }
}