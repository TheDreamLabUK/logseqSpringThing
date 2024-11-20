struct PositionUpdate {
    position: vec3<f32>,  // 12 bytes
    velocity: vec3<f32>,  // 12 bytes
    mass: f32,           // 4 bytes for mass based on node size
    padding: vec3<f32>,  // 12 bytes padding for alignment
}

@group(0) @binding(0) var<storage, read_write> position_updates: array<PositionUpdate>;

// Utility functions
fn is_valid_float(x: f32) -> bool {
    return x == x && abs(x) < 1e10;
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return is_valid_float(v.x) && is_valid_float(v.y) && is_valid_float(v.z);
}

@compute @workgroup_size(256)
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&position_updates);

    if (node_id >= n_nodes) { return; }

    var update = position_updates[node_id];
    
    // Validate position, velocity, and mass
    if (!is_valid_float3(update.position)) {
        update.position = vec3<f32>(0.0);
    }
    if (!is_valid_float3(update.velocity)) {
        update.velocity = vec3<f32>(0.0);
    }
    
    // Ensure mass is positive and reasonable
    if (!is_valid_float(update.mass) || update.mass <= 0.0) {
        update.mass = 1.0; // Default mass if invalid
    }

    // Scale velocity based on mass (heavier nodes have more momentum)
    update.velocity = update.velocity * (1.0 / sqrt(update.mass));
    
    position_updates[node_id] = update;
}
