struct PositionUpdate {
    position: vec3<f32>,  // 12 bytes (x, y, z)
    velocity: vec3<f32>,  // 12 bytes (vx, vy, vz)
}

@group(0) @binding(0) var<storage, read_write> position_updates: array<PositionUpdate>;

// Constants
const MAX_VELOCITY: f32 = 100.0;
const MAX_POSITION: f32 = 1000.0;  // Maximum distance from origin

// Utility functions
fn is_valid_float(x: f32) -> bool {
    return x == x && abs(x) < 1e10;  // Check for NaN and infinity
}

fn is_valid_float3(v: vec3<f32>) -> bool {
    return is_valid_float(v.x) && is_valid_float(v.y) && is_valid_float(v.z);
}

fn clamp_position(pos: vec3<f32>) -> vec3<f32> {
    return clamp(pos, vec3<f32>(-MAX_POSITION), vec3<f32>(MAX_POSITION));
}

fn clamp_velocity(vel: vec3<f32>) -> vec3<f32> {
    let speed = length(vel);
    if (speed > MAX_VELOCITY) {
        return (vel / speed) * MAX_VELOCITY;
    }
    return vel;
}

@compute @workgroup_size(256)
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;
    let n_nodes = arrayLength(&position_updates);

    if (node_id >= n_nodes) { return; }

    var update = position_updates[node_id];
    
    // Validate and clamp position
    if (!is_valid_float3(update.position)) {
        update.position = vec3<f32>(0.0);
    } else {
        update.position = clamp_position(update.position);
    }
    
    // Validate and clamp velocity
    if (!is_valid_float3(update.velocity)) {
        update.velocity = vec3<f32>(0.0);
    } else {
        update.velocity = clamp_velocity(update.velocity);
    }
    
    position_updates[node_id] = update;
}
