// Aligned Vec3 struct matching Three.js Vector3 memory layout
struct alignas(16) Vec3 {
    float x, y, z;

    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __device__ Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    __device__ Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    __device__ Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    
    __device__ Vec3 normalized() const {
        float len = length();
        if (len < 1e-6f) return Vec3();
        float inv_len = 1.0f / len;
        return Vec3(x * inv_len, y * inv_len, z * inv_len);
    }
    
    __device__ float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
};

// Node data structure with Vec3
struct NodeData {
    Vec3 position;       // 12 bytes
    Vec3 velocity;       // 12 bytes
    unsigned char mass;  // 1 byte
    unsigned char flags; // 1 byte
    unsigned char padding[2]; // 2 bytes padding for alignment
};

extern "C" __global__ void compute_forces(
    NodeData* nodes,
    int num_nodes,
    float spring_strength,
    float repulsion,
    float damping,
    float max_repulsion_distance,
    float viewport_bounds
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    // Load node data
    NodeData node_i = nodes[idx];
    Vec3 pos_i = node_i.position;
    float mass_i = static_cast<float>(node_i.mass) / 255.0f; // Normalize mass to [0,1]
    Vec3 force;

    __shared__ Vec3 shared_positions[256];
    __shared__ float shared_masses[256];

    // Process nodes in tiles to maximize shared memory usage
    for (int tile = 0; tile < (num_nodes + blockDim.x - 1) / blockDim.x; tile++) {
        int shared_idx = tile * blockDim.x + threadIdx.x;
        
        // Load tile into shared memory
        if (shared_idx < num_nodes) {
            // Load directly from global memory using Vec3 members
            shared_positions[threadIdx.x].x = nodes[shared_idx].position.x;
            shared_positions[threadIdx.x].y = nodes[shared_idx].position.y;
            shared_positions[threadIdx.x].z = nodes[shared_idx].position.z;
            
            // Load mass and normalize
            NodeData shared_node = nodes[shared_idx];  // Only load once
            shared_masses[threadIdx.x] = static_cast<float>(shared_node.mass) / 255.0f;
        }
        __syncthreads();

        // Compute forces between current node and all nodes in tile
        #pragma unroll 8
        for (int j = 0; j < blockDim.x && tile * blockDim.x + j < num_nodes; j++) {
            if (tile * blockDim.x + j == idx) continue;

            // Skip nodes with inactive flag
            if ((nodes[tile * blockDim.x + j].flags & 0x1) == 0) continue;

            const Vec3& pos_j = shared_positions[j];  // Use const reference
            float mass_j = shared_masses[j];
            
            // Calculate displacement vector
            Vec3 diff = pos_i - pos_j;
            
            // Calculate force magnitude with minimum distance clamp
            float dist = fmaxf(diff.length(), 0.0001f);
            
            // Calculate bounded repulsion force
            float force_mag = 0.0f;
            if (dist < max_repulsion_distance) {
                // Use square root of mass product to prevent excessive forces
                float mass_factor = sqrtf(mass_i * mass_j);
                force_mag = -repulsion * mass_factor / (dist * dist);
                
                // Clamp maximum repulsion force
                force_mag = fmaxf(force_mag, -10.0f);
                
                if (idx == 0 && j == 0) {
                    printf("Node 0: dist=%f, mass_i=%f, mass_j=%f, repulsion_force=%f\n",
                           dist, mass_i, mass_j, force_mag);
                }
            }

            // Add spring force if nodes are connected (check flags)
            if ((node_i.flags & 0x2) && (nodes[tile * blockDim.x + j].flags & 0x2)) {
                // Simple spring force with fixed rest length
                float rest_length = 1.0f; // Fixed rest length for all edges
                // Add non-linear spring force with clamping
                float displacement = dist - rest_length;
                float spring_force = spring_strength * displacement * (1.0f + 0.5f * tanhf(fabsf(displacement)));
                // Clamp maximum spring force
                spring_force = fmaxf(fminf(spring_force, 10.0f), -10.0f);
                force_mag += spring_force;
                
                if (idx == 0 && j == 0) {
                    printf("Node 0: spring_force=%f\n", spring_force);
                }
            }
            
            // Clamp total force magnitude
            force_mag = fmaxf(fminf(force_mag, 15.0f), -15.0f);

            // Accumulate force using normalized direction
            const Vec3 force_dir = diff.normalized();  // Use const
            force = force + force_dir * force_mag;
            
            // Improved momentum damping
            float velocity_alignment = node_i.velocity.dot(force_dir);
            if (velocity_alignment > 0) {
                // Scale damping with mass and clamp it
                float momentum_damping = -velocity_alignment * sqrtf(mass_i) * 0.1f;
                momentum_damping = fmaxf(momentum_damping, -5.0f);
                force = force + force_dir * momentum_damping;
            }
        }
        __syncthreads();
    }

    // Clamp total force magnitude before velocity update
    float force_mag = force.length();
    if (force_mag > 20.0f) {
        force = force * (20.0f / force_mag);
    }

    // Update velocity with damping
    Vec3 new_velocity = (node_i.velocity + force) * damping;

    // Debug print velocity for first node
    if (idx == 0) {
        printf("Node 0: force_mag=%f, force=(%f, %f, %f), new_vel=(%f, %f, %f)\n",
               force_mag,
               force.x, force.y, force.z,
               new_velocity.x, new_velocity.y, new_velocity.z);
    }

    // Apply mass-based velocity limiting
    float max_velocity = 2.0f / (0.5f + mass_i); // Heavier nodes move slower
    float velocity_mag = new_velocity.length();
    if (velocity_mag > max_velocity) {
        new_velocity = new_velocity * (max_velocity / velocity_mag);
    }

    // Update position
    Vec3 new_position = pos_i + new_velocity;

    // Apply viewport bounds
    if (viewport_bounds > 0.0f) {
        new_position.x = fmaxf(fminf(new_position.x, viewport_bounds), -viewport_bounds);
        new_position.y = fmaxf(fminf(new_position.y, viewport_bounds), -viewport_bounds);
        new_position.z = fmaxf(fminf(new_position.z, viewport_bounds), -viewport_bounds);
    }

    // Apply additional damping for nodes near bounds
    if (viewport_bounds > 0.0f && (fabsf(new_position.x) > viewport_bounds * 0.9f ||
                                  fabsf(new_position.y) > viewport_bounds * 0.9f ||
                                  fabsf(new_position.z) > viewport_bounds * 0.9f)) {
        new_velocity = new_velocity * 0.9f; // Extra damping near bounds
    }

    // Store updated position and velocity
    nodes[idx].position = new_position;
    nodes[idx].velocity = new_velocity;

    // Flags and mass remain unchanged
}
