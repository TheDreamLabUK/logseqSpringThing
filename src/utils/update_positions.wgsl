struct PositionUpdate {
    position: vec3<f32>,  // 12 bytes
}

@group(0) @binding(0) var<storage, read_write> position_updates: array<PositionUpdate>;

// Utility functions
fn is_valid_float(x: f32) -> bool {
    return x == x && abs(x) < 1e10;
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return is_valid_float(v.x) && is_valid_float(v.y) && is_valid_float(v.z);
}

@compute @workgroup_size(256)  // Increased workgroup size for better throughput
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&position_updates);

    if (node_id >= n_nodes) { return; }

    var update = position_updates[node_id];
    
    // Only validate position
    if (!is_valid_float3(update.position)) {
        update.position = vec3<f32>(0.0);
    }

    position_updates[node_id] = update;
}
