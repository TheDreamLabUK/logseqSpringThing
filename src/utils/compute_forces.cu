// Node data structure matching Rust's NodeData
struct NodeData {
    float position[3];    // 12 bytes
    unsigned char mass;   // 1 byte
    unsigned char flags;  // 1 byte
    unsigned char padding[2]; // 2 bytes padding
};

// Velocity data structure matching Rust's VelocityData
struct VelocityData {
    float x;
    float y;
    float z;
};

extern "C" __global__ void compute_forces(
    NodeData* nodes,
    VelocityData* velocities,
    unsigned long long unused,
    unsigned int num_nodes,
    float spring_strength,
    float spring_length,
    float repulsion,
    float attraction,
    float damping
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    // Load node data
    NodeData node_i = nodes[idx];
    float3 pos_i = make_float3(
        node_i.position[0],
        node_i.position[1],
        node_i.position[2]
    );
    float mass_i = (float)node_i.mass;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    __shared__ float3 shared_positions[256];
    __shared__ float shared_masses[256];

    // Process nodes in tiles to maximize shared memory usage
    for (int tile = 0; tile < (num_nodes + blockDim.x - 1) / blockDim.x; tile++) {
        int shared_idx = tile * blockDim.x + threadIdx.x;
        
        // Load tile into shared memory
        if (shared_idx < num_nodes) {
            NodeData shared_node = nodes[shared_idx];
            shared_positions[threadIdx.x] = make_float3(
                shared_node.position[0],
                shared_node.position[1],
                shared_node.position[2]
            );
            shared_masses[threadIdx.x] = (float)shared_node.mass;
        }
        __syncthreads();

        // Compute forces between current node and all nodes in tile
        #pragma unroll 8
        for (int j = 0; j < blockDim.x && tile * blockDim.x + j < num_nodes; j++) {
            if (tile * blockDim.x + j == idx) continue;

            // Skip nodes with inactive flag
            if ((nodes[tile * blockDim.x + j].flags & 0x1) == 0) continue;

            float3 pos_j = shared_positions[j];
            float mass_j = shared_masses[j];
            
            // Calculate displacement vector from j to i for repulsion
            float3 diff = make_float3(
                pos_i.x - pos_j.x,
                pos_i.y - pos_j.y,
                pos_i.z - pos_j.z
            );

            // Calculate distance with minimum clamp
            float dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            float dist = fmaxf(sqrtf(dist2), 0.1f); // Smaller min distance for more dynamic movement
            
            // Normalize direction vector
            float inv_dist = 1.0f / dist;
            float3 dir = make_float3(
                diff.x * inv_dist,
                diff.y * inv_dist,
                diff.z * inv_dist
            );

            // Calculate repulsion force (inverse square law)
            float mass_factor = sqrtf(mass_i * mass_j);
            float repulsion_mag = repulsion * mass_factor / dist2;
            
            // Add repulsion force
            force.x += dir.x * repulsion_mag;
            force.y += dir.y * repulsion_mag;
            force.z += dir.z * repulsion_mag;

            // Add spring force if nodes are connected
            if ((node_i.flags & 0x2) && (nodes[tile * blockDim.x + j].flags & 0x2)) {
                // Spring force points opposite to displacement if too far, along it if too close
                float spring_displacement = dist - spring_length;
                float spring_mag = spring_strength * spring_displacement * attraction;
                
                // Spring force opposes displacement
                force.x -= dir.x * spring_mag;
                force.y -= dir.y * spring_mag;
                force.z -= dir.z * spring_mag;
            }
        }
        __syncthreads();
    }

    // Clamp maximum force magnitude
    float force_mag = sqrtf(force.x * force.x + force.y * force.y + force.z * force.z);
    if (force_mag > 1000.0f) {
        float scale = 1000.0f / force_mag;
        force.x *= scale;
        force.y *= scale;
        force.z *= scale;
    }

    // Load current velocity
    float3 vel = make_float3(
        velocities[idx].x,
        velocities[idx].y,
        velocities[idx].z
    );

    // Time step for integration (adjust this to control simulation speed)
    const float dt = 0.016f; // 60 fps

    // Semi-implicit Euler integration
    // First update velocity (v = v + a*dt)
    vel.x = (vel.x + force.x * dt) * damping;
    vel.y = (vel.y + force.y * dt) * damping;
    vel.z = (vel.z + force.z * dt) * damping;

    // Then update position (p = p + v*dt)
    pos_i.x += vel.x * dt;
    pos_i.y += vel.y * dt;
    pos_i.z += vel.z * dt;

    // Store updated position and velocity
    nodes[idx].position[0] = pos_i.x;
    nodes[idx].position[1] = pos_i.y;
    nodes[idx].position[2] = pos_i.z;
    velocities[idx].x = vel.x;
    velocities[idx].y = vel.y;
    velocities[idx].z = vel.z;
}
