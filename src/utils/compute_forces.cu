#include <cuda_runtime.h>

extern "C" {
    struct BinaryNodeData {
        float position[3];    // 12 bytes - matches Rust [f32; 3]
        float velocity[3];    // 12 bytes - matches Rust [f32; 3]
        unsigned char mass;   // 1 byte - matches Rust u8
        unsigned char flags;  // 1 byte - matches Rust u8
        unsigned char padding[2]; // 2 bytes - matches Rust padding
    };

    __global__ void compute_forces_kernel(
        BinaryNodeData* nodes,
        int num_nodes,
        float spring_k,
        float damping,
        float repel_k,
        float dt,
        float max_repulsion_dist,
        float viewport_bounds
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nodes) return;

        float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos = make_float3(nodes[idx].position[0], nodes[idx].position[1], nodes[idx].position[2]);
        float3 vel = make_float3(nodes[idx].velocity[0], nodes[idx].velocity[1], nodes[idx].velocity[2]);
        
        // Convert mass from u8 to float (0-1 range)
        float mass = (nodes[idx].mass + 1.0f) / 256.0f; // Add 1 to avoid zero mass
        bool is_active = (nodes[idx].flags & 0x1) != 0;
        
        if (!is_active) return; // Skip inactive nodes
        
        // Process all node interactions
        for (int j = 0; j < num_nodes; j++) {
            if (j == idx) continue;
            
            if (!(nodes[j].flags & 0x1)) continue; // Skip inactive nodes
            
            float other_mass = (nodes[j].mass + 1.0f) / 256.0f;
            float3 other_pos = make_float3(
                nodes[j].position[0],
                nodes[j].position[1],
                nodes[j].position[2]
            );
            
            float3 diff = make_float3(
                other_pos.x - pos.x,
                other_pos.y - pos.y,
                other_pos.z - pos.z
            );
            
            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            if (dist > 0.0001f) {
                float3 dir = make_float3(
                    diff.x / dist,
                    diff.y / dist,
                    diff.z / dist
                );
                
                // Spring forces - apply only if both nodes have the connected flag
                if ((nodes[idx].flags & 0x2) && (nodes[j].flags & 0x2)) {
                    // Use natural length of 1.0 to match world units
                    float spring_force = spring_k * (dist - 1.0f);
                    float spring_scale = mass * other_mass;
                    total_force.x += dir.x * spring_force * spring_scale;
                    total_force.y += dir.y * spring_force * spring_scale;
                    total_force.z += dir.z * spring_force * spring_scale;
                }
                
                // Repulsion forces
                if (dist < max_repulsion_dist) {
                    float repel_scale = repel_k * mass * other_mass;
                    float repel_force = repel_scale / (dist * dist);
                    total_force.x -= dir.x * repel_force;
                    total_force.y -= dir.y * repel_force;
                    total_force.z -= dir.z * repel_force;
                }
            }
        }

        // Apply damping to velocity
        vel.x = vel.x * (1.0f - damping) + total_force.x * dt;
        vel.y = vel.y * (1.0f - damping) + total_force.y * dt;
        vel.z = vel.z * (1.0f - damping) + total_force.z * dt;

        // Update position
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        // Constrain to viewport bounds if enabled (bounds > 0)
        if (viewport_bounds > 0.0f) {
            pos.x = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.x));
            pos.y = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.y));
            pos.z = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.z));
        }

        // Store results back
        nodes[idx].position[0] = pos.x;
        nodes[idx].position[1] = pos.y;
        nodes[idx].position[2] = pos.z;
        nodes[idx].velocity[0] = vel.x;
        nodes[idx].velocity[1] = vel.y;
        nodes[idx].velocity[2] = vel.z;

        // Debug output for first node
        if (idx == 0) {
            float force_mag = sqrtf(
                total_force.x * total_force.x +
                total_force.y * total_force.y +
                total_force.z * total_force.z
            );
            printf("Node %d: force_mag=%f, pos=(%f,%f,%f), vel=(%f,%f,%f)\n",
                idx, force_mag,
                pos.x, pos.y, pos.z,
                vel.x, vel.y, vel.z);
        }
    }
}
