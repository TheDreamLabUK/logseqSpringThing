#include <cuda_runtime.h>
#include <math.h>

extern "C" {
    // Matches Rust Vec3Data memory layout
    struct NodeData {
        float x, y, z;        // position (12 bytes)
        float vx, vy, vz;     // velocity (12 bytes)
        float mass;           // mass derived from file size (4 bytes)
    };

    struct EdgeData {
        int source_idx;      // Index of source node (4 bytes)
        int target_idx;      // Index of target node (4 bytes)
        float weight;        // Edge weight from bi-directional links (4 bytes)
    };

    // Constants for validation
    const float MAX_FORCE = 1000.0f;
    const float MAX_VELOCITY = 100.0f;
    const float MIN_DIST = 0.0001f;

    __device__ bool is_valid_float3(float3 v) {
        return !isnan(v.x) && !isnan(v.y) && !isnan(v.z) &&
               isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
    }

    __device__ float3 clamp_force(float3 force) {
        float mag = sqrtf(force.x * force.x + force.y * force.y + force.z * force.z);
        if (mag > MAX_FORCE) {
            float scale = MAX_FORCE / mag;
            force.x *= scale;
            force.y *= scale;
            force.z *= scale;
        }
        return force;
    }

    __device__ float3 clamp_velocity(float3 vel) {
        float mag = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        if (mag > MAX_VELOCITY) {
            float scale = MAX_VELOCITY / mag;
            return make_float3(vel.x * scale, vel.y * scale, vel.z * scale);
        }
        return vel;
    }

    __global__ void compute_forces_kernel(
        NodeData* nodes,
        int num_nodes,
        EdgeData* edges,
        int num_edges,
        float spring_k,
        float damping,
        float repel_k,
        float dt,
        float max_repulsion_dist,
        float viewport_bounds
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nodes) return;

        // Validate simulation parameters
        if (spring_k < 0.0f || damping < 0.0f || damping > 1.0f || 
            repel_k < 0.0f || dt <= 0.0f || max_repulsion_dist <= 0.0f) {
            if (idx == 0) {
                printf("Invalid simulation parameters: spring_k=%f, damping=%f, repel_k=%f, dt=%f, max_repulsion_dist=%f\n",
                    spring_k, damping, repel_k, dt, max_repulsion_dist);
            }
            return;
        }

        float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos = make_float3(nodes[idx].x, nodes[idx].y, nodes[idx].z);
        float3 vel = make_float3(nodes[idx].vx, nodes[idx].vy, nodes[idx].vz);

        // Validate input position and velocity
        if (!is_valid_float3(pos) || !is_valid_float3(vel)) {
            if (idx == 0) {
                printf("Node %d: Invalid input detected - pos=(%f,%f,%f), vel=(%f,%f,%f)\n",
                    idx, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z);
            }
            // Reset to safe values
            pos = make_float3(0.0f, 0.0f, 0.0f);
            vel = make_float3(0.0f, 0.0f, 0.0f);
            nodes[idx].x = 0.0f;
            nodes[idx].y = 0.0f;
            nodes[idx].z = 0.0f;
            nodes[idx].vx = 0.0f;
            nodes[idx].vy = 0.0f;
            nodes[idx].vz = 0.0f;
        }
        
        // Process spring forces from edges first
        for (int e = 0; e < num_edges; e++) {
            if (edges[e].source_idx == idx || edges[e].target_idx == idx) {
                int other_idx = (edges[e].source_idx == idx) ? edges[e].target_idx : edges[e].source_idx;
                float3 other_pos = make_float3(nodes[other_idx].x, nodes[other_idx].y, nodes[other_idx].z);
                float3 diff = make_float3(other_pos.x - pos.x, other_pos.y - pos.y, other_pos.z - pos.z);
                float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                
                if (dist > MIN_DIST) {
                    float3 dir = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);
                    float spring_force = spring_k * (dist - 1.0f) * edges[e].weight;
                    float spring_scale = nodes[idx].mass * nodes[other_idx].mass;
                    total_force.x += dir.x * spring_force * spring_scale;
                    total_force.y += dir.y * spring_force * spring_scale;
                    total_force.z += dir.z * spring_force * spring_scale;
                }
            }
        }
        
        // Process all node interactions
        for (int j = 0; j < num_nodes; j++) {
            if (j == idx) continue;
            
            // Use mass from node data
            float other_mass = nodes[j].mass;
            float3 other_pos = make_float3(
                nodes[j].x,
                nodes[j].y,
                nodes[j].z
            );
            
            float3 diff = make_float3(
                other_pos.x - pos.x,
                other_pos.y - pos.y,
                other_pos.z - pos.z
            );
            
            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            if (dist > MIN_DIST) {
                float3 dir = make_float3(
                    diff.x / dist,
                    diff.y / dist,
                    diff.z / dist
                );
                
                // Repulsion forces
                if (dist < max_repulsion_dist) {
                    float repel_scale = repel_k * nodes[idx].mass * other_mass;
                    float repel_force = repel_scale / (dist * dist);
                    total_force.x -= dir.x * repel_force;
                    total_force.y -= dir.y * repel_force;
                    total_force.z -= dir.z * repel_force;
                }
            }
        }

        // Clamp total force to prevent instability
        total_force = clamp_force(total_force);

        // Apply damping to velocity
        vel.x = vel.x * (1.0f - damping) + total_force.x * dt;
        vel.y = vel.y * (1.0f - damping) + total_force.y * dt;
        vel.z = vel.z * (1.0f - damping) + total_force.z * dt;

        // Update position
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        // Clamp velocity to prevent runaway values
        vel = clamp_velocity(vel);

        // Constrain to viewport bounds if enabled (bounds > 0)
        if (viewport_bounds > 0.0f) {
            pos.x = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.x));
            pos.y = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.y));
            pos.z = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.z));
        }

        // Store results back
        nodes[idx].x = pos.x;
        nodes[idx].y = pos.y;
        nodes[idx].z = pos.z;
        nodes[idx].vx = vel.x;
        nodes[idx].vy = vel.y;
        nodes[idx].vz = vel.z;

        // Debug output for first node
        if (idx == 0) {
            float force_mag = sqrtf(
                total_force.x * total_force.x +
                total_force.y * total_force.y +
                total_force.z * total_force.z
            );
            float vel_mag = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            printf("Node %d: force_mag=%f (max=%f), vel_mag=%f (max=%f), pos=(%f,%f,%f), vel=(%f,%f,%f)\n",
                idx, force_mag,
                MAX_FORCE, vel_mag, MAX_VELOCITY,
                pos.x, pos.y, pos.z,
                vel.x, vel.y, vel.z);
        }
    }
}
