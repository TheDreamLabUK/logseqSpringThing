#include <cuda_runtime.h>

extern "C" {
    // This struct matches the Rust BinaryNodeData struct
    struct BinaryNodeData {
        float position[3];    // 12 bytes - matches Rust [f32; 3]
        float velocity[3];    // 12 bytes - matches Rust [f32; 3]
        // These fields are used internally but not transmitted over the wire
        // The binary_protocol.rs sets default values when decoding
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
        float viewport_bounds,
        int iteration_count
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nodes) return;

        const float MAX_FORCE = 3.0f; // Reduced maximum force magnitude
        const float MAX_VELOCITY = 0.02f; // Stricter velocity cap to prevent momentum buildup
        const float MIN_DISTANCE = 0.15f; // Slightly increased minimum distance
        
        // Progressive force application parameters
        // First 100 iterations use a ramp-up factor
        const int WARMUP_ITERATIONS = 100;
        float ramp_up_factor = 1.0f;
        
        if (iteration_count < WARMUP_ITERATIONS) {
            // Gradually increase from 0.01 to 1.0 over WARMUP_ITERATIONS
            ramp_up_factor = 0.01f + (iteration_count / (float)WARMUP_ITERATIONS) * 0.99f;
            
            // Also use higher damping in initial iterations to stabilize the system
            damping = fmaxf(damping, 0.9f - 0.4f * (iteration_count / (float)WARMUP_ITERATIONS));
        }
        
        float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos = make_float3(nodes[idx].position[0], nodes[idx].position[1], nodes[idx].position[2]);
        float3 vel = make_float3(nodes[idx].velocity[0], nodes[idx].velocity[1], nodes[idx].velocity[2]);

        // Zero out velocity in the very first iterations to prevent explosion
        if (iteration_count < 5) {
            vel = make_float3(0.0f, 0.0f, 0.0f);
        }
        
        // Convert mass from u8 to float (approximately 0-1 range)
        float mass;
        if (nodes[idx].mass == 0) {
            mass = 0.5f; // Default mid-range mass value
        } else {
            mass = (nodes[idx].mass + 1.0f) / 256.0f; // Add 1 to avoid zero mass
        }
        
        bool is_active = true; // All nodes are active by default
        
        if (!is_active) return; // Skip inactive nodes
        
        // Process all node interactions
        for (int j = 0; j < num_nodes; j++) {
            if (j == idx) continue;
            
            // All nodes are considered active by default
            // We no longer check the flags since all nodes are treated as active
            
            // Handle other node's mass the same way
            float other_mass = (nodes[j].mass == 0) ? 0.5f : (nodes[j].mass + 1.0f) / 256.0f;
            
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
            // Only process if nodes are at a meaningful distance apart
            if (dist > MIN_DISTANCE) {
                float3 dir = make_float3(
                    diff.x / dist,
                    diff.y / dist,
                    diff.z / dist
                );
                
                // Apply spring forces to all nodes by default
                {
                    // Use natural length of 1.0 to match world units
                    float natural_length = 1.0f;
                    
                    // Progressive spring forces - stronger when further apart
                    // Apply the ramp_up_factor to gradually increase spring forces
                    float spring_force = spring_k * ramp_up_factor * (dist - natural_length);
                    
                    // Apply progressively stronger springs for very distant nodes
                    if (dist > natural_length * 3.0f) {
                        spring_force *= (1.0f + (dist - natural_length * 3.0f) * 0.1f);
                    }
                    
                    
                    float spring_scale = mass * other_mass;
                    float force_magnitude = spring_force * spring_scale;
                    
                    // Repulsion forces - only apply at close distances
                    if (dist < max_repulsion_dist) {
                        float repel_scale = repel_k * mass * other_mass;
                        // Apply the ramp_up_factor to gradually increase repulsion forces
                        float dist_sq = fmaxf(dist * dist, MIN_DISTANCE);
                        // Cap maximum repulsion force to prevent explosion
                        float repel_force = fminf(repel_scale / dist_sq, repel_scale * 2.0f);
                        total_force.x -= dir.x * repel_force;
                        total_force.y -= dir.y * repel_force;
                        total_force.z -= dir.z * repel_force;
                    } else {
                        // Always apply spring forces
                        total_force.x += dir.x * force_magnitude;
                        total_force.y += dir.y * force_magnitude;
                        total_force.z += dir.z * force_magnitude;
                    }
                }
            }
        }
        
        // Stronger center gravity to prevent nodes from drifting too far
        float center_strength = 0.015f * mass * ramp_up_factor; // Apply ramp_up to center gravity too
        float center_dist = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
        if (center_dist > 3.0f) { // Apply at shorter distances
            float center_factor = center_strength * (center_dist - 3.0f) / center_dist;
            total_force.x -= pos.x * center_factor;
            total_force.y -= pos.y * center_factor;
            total_force.z -= pos.z * center_factor;
        }

        // Calculate total force magnitude
        float force_magnitude = sqrtf(
            total_force.x*total_force.x + 
            total_force.y*total_force.y + 
            total_force.z*total_force.z);
        
        // Scale down excessive forces to prevent explosion
        if (force_magnitude > MAX_FORCE) {
            float scale_factor = MAX_FORCE / force_magnitude;
            total_force.x *= scale_factor;
            total_force.y *= scale_factor;
            total_force.z *= scale_factor;
            
            // Additional logging to help debug extreme forces after randomization
            if (idx == 0 && iteration_count < 5)
                printf("Force clamped from %f to %f (iteration %d)\n", force_magnitude, MAX_FORCE, iteration_count);
        }

        // Apply damping and bounded forces to velocity
        vel.x = vel.x * (1.0f - damping) + fminf(MAX_FORCE, fmaxf(-MAX_FORCE, total_force.x)) * dt;
        vel.y = vel.y * (1.0f - damping) + fminf(MAX_FORCE, fmaxf(-MAX_FORCE, total_force.y)) * dt;
        vel.z = vel.z * (1.0f - damping) + fminf(MAX_FORCE, fmaxf(-MAX_FORCE, total_force.z)) * dt;
        
        // Apply STRICT velocity cap to prevent runaway momentum
        float vel_magnitude = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        if (vel_magnitude > MAX_VELOCITY) {
            float scale_factor = MAX_VELOCITY / vel_magnitude;
            vel.x *= scale_factor;
            vel.y *= scale_factor;
            vel.z *= scale_factor;
        }
        
        // Update position
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        // Progressive boundary approach - stronger the further you go
        if (viewport_bounds > 0.0f && iteration_count > 10) { // Only apply boundary after initial stabilization
            float soft_margin = 0.3f * viewport_bounds; // 30% soft boundary
            float bound_with_margin = viewport_bounds - soft_margin;

            // Apply progressively stronger boundary forces
            if (fabsf(pos.x) > bound_with_margin) {
                pos.x *= 0.92f; // Pull back by 8%
                // Also add dampening to velocity in this direction
                vel.x *= 0.85f;
            }
            if (fabsf(pos.y) > bound_with_margin) {
                pos.y *= 0.92f; // Pull back by 8%
                vel.y *= 0.85f;
            }
            if (fabsf(pos.z) > bound_with_margin) {
                pos.z *= 0.92f; // Pull back by 8%
                vel.z *= 0.85f;
            }
        }

        // Store results back
        nodes[idx].position[0] = pos.x;
        nodes[idx].position[1] = pos.y;
        nodes[idx].position[2] = pos.z;
        nodes[idx].velocity[0] = vel.x;
        nodes[idx].velocity[1] = vel.y;
        nodes[idx].velocity[2] = vel.z;

        // Debug output for first node
        if (idx == 0 && (iteration_count < 5 || iteration_count % 20 == 0)) {
            float force_mag = sqrtf(
                total_force.x * total_force.x +
                total_force.y * total_force.y +
                total_force.z * total_force.z
            );
            printf("Node %d: force_mag=%f, pos=(%f,%f,%f), vel=(%f,%f,%f)\n",
                idx, force_mag, 
                pos.x, pos.y, pos.z,
                vel.x, vel.y, vel.z);
                
            // More detailed logging during initialization
            if (iteration_count < WARMUP_ITERATIONS)
                printf("Node %d: iteration=%d, ramp_up=%f, damping=%f\n", idx, iteration_count, ramp_up_factor, damping);
        }
    }
}
