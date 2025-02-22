#include <cuda_runtime.h>

// Constants
#define MAX_CONNECTIONS 32
#define MAX_REPULSION_DISTANCE 5.0f
#define VIEWPORT_BOUNDS 100.0f
#define SPRING_STRENGTH 0.5f
#define REPULSION 1.0f

extern "C" {
    struct Node {
        float x, y, z;           // Position
        float vel_x, vel_y, vel_z; // Velocity
        float mass;              // Node mass
        unsigned int connected[(MAX_CONNECTIONS + 31) / 32]; // Bitfield for connections
        int connections[MAX_CONNECTIONS];
        int num_connections;
    };

    __global__ void compute_forces_kernel(
        Node* nodes,
        int num_nodes,
        float spring_k,
        float damping,
        float repel_k,
        float dt
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_nodes) {
            float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
            
            float mass = nodes[idx].mass / 255.0f;
            float3 pos = make_float3(nodes[idx].x, nodes[idx].y, nodes[idx].z);
            float3 vel = make_float3(nodes[idx].vel_x, nodes[idx].vel_y, nodes[idx].vel_z);
            
            for (int j = 0; j < num_nodes; j++) {
                if (j != idx) {
                    float other_mass = nodes[j].mass / 255.0f;
                    float3 other_pos = make_float3(nodes[j].x, nodes[j].y, nodes[j].z);
                    
                    float3 diff = make_float3(
                        other_pos.x - pos.x,
                        other_pos.y - pos.y,
                        other_pos.z - pos.z
                    );
                    
                    float dist = sqrtf(
                        diff.x * diff.x +
                        diff.y * diff.y +
                        diff.z * diff.z
                    );
                    
                    if (dist > 0.0001f) {
                        // Spring forces for connected nodes
                        if (nodes[idx].connected[j / 32] & (1u << (j % 32))) {
                            float spring_scale = SPRING_STRENGTH * mass * other_mass;
                            float3 spring = make_float3(
                                diff.x * spring_k * spring_scale,
                                diff.y * spring_k * spring_scale,
                                diff.z * spring_k * spring_scale
                            );
                            total_force.x += spring.x;
                            total_force.y += spring.y;
                            total_force.z += spring.z;
                        }
                        
                        // Repulsion forces
                        if (dist < MAX_REPULSION_DISTANCE) {
                            float repel_strength = REPULSION * mass * other_mass / (dist * dist);
                            float3 repel = make_float3(
                                -diff.x * repel_k * repel_strength / dist,
                                -diff.y * repel_k * repel_strength / dist,
                                -diff.z * repel_k * repel_strength / dist
                            );
                            total_force.x += repel.x;
                            total_force.y += repel.y;
                            total_force.z += repel.z;
                        }
                    }
                }
            }
            
            // Update velocity with damping
            nodes[idx].vel_x = vel.x * (1.0f - damping) + total_force.x * dt;
            nodes[idx].vel_y = vel.y * (1.0f - damping) + total_force.y * dt;
            nodes[idx].vel_z = vel.z * (1.0f - damping) + total_force.z * dt;
            
            // Update position
            nodes[idx].x += nodes[idx].vel_x * dt;
            nodes[idx].y += nodes[idx].vel_y * dt;
            nodes[idx].z += nodes[idx].vel_z * dt;
            
            // Constrain to viewport bounds
            nodes[idx].x = fmaxf(-VIEWPORT_BOUNDS, fminf(VIEWPORT_BOUNDS, nodes[idx].x));
            nodes[idx].y = fmaxf(-VIEWPORT_BOUNDS, fminf(VIEWPORT_BOUNDS, nodes[idx].y));
            nodes[idx].z = fmaxf(-VIEWPORT_BOUNDS, fminf(VIEWPORT_BOUNDS, nodes[idx].z));
        }
    }
}
