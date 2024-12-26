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
            
            // Calculate displacement vector
            float3 diff = make_float3(
                pos_i.x - pos_j.x,
                pos_i.y - pos_j.y,
                pos_i.z - pos_j.z
            );

    // Calculate distance with larger minimum clamp to prevent instability
    float dist = fmaxf(sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z), 1.0f);
    
    // Scale repulsion force by sqrt of masses to reduce excessive force
    float mass_factor = sqrtf(mass_i * mass_j);
    float force_mag = repulsion * mass_factor / (dist * dist);

    // Add spring force if nodes are connected (check flags)
    if ((node_i.flags & 0x2) && (nodes[tile * blockDim.x + j].flags & 0x2)) {
        // Use spring_length parameter for natural length
        float spring_force = spring_strength * (dist - spring_length);
        // Add attraction component
        force_mag += spring_force * attraction;
    }

    // Clamp maximum force to prevent instability
    force_mag = fminf(force_mag, 1000.0f);

            // Accumulate force
            force.x += force_mag * diff.x / dist;
            force.y += force_mag * diff.y / dist;
            force.z += force_mag * diff.z / dist;
        }
        __syncthreads();
    }

    // Load current velocity
    float3 vel = make_float3(
        velocities[idx].x,
        velocities[idx].y,
        velocities[idx].z
    );

    // Update velocity with damping
    vel.x = (vel.x + force.x) * damping;
    vel.y = (vel.y + force.y) * damping;
    vel.z = (vel.z + force.z) * damping;

    // Update position
    pos_i.x += vel.x;
    pos_i.y += vel.y;
    pos_i.z += vel.z;

    // Store updated position and velocity
    nodes[idx].position[0] = pos_i.x;
    nodes[idx].position[1] = pos_i.y;
    nodes[idx].position[2] = pos_i.z;
    velocities[idx].x = vel.x;
    velocities[idx].y = vel.y;
    velocities[idx].z = vel.z;

    // Flags and mass remain unchanged
}
