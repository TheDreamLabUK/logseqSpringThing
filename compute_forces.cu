extern "C" __global__ void compute_forces(
    float* positions,
    float* velocities,
    unsigned char* masses,
    int num_nodes,
    float spring_strength,
    float repulsion,
    float damping
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int pos_idx = idx * 3;
    float3 pos_i = make_float3(
        positions[pos_idx],
        positions[pos_idx + 1],
        positions[pos_idx + 2]
    );
    float mass_i = (float)masses[idx];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    __shared__ float3 shared_positions[256];
    __shared__ float shared_masses[256];

    for (int tile = 0; tile < (num_nodes + 256 - 1) / 256; tile++) {
        int shared_idx = tile * 256 + threadIdx.x;
        if (shared_idx < num_nodes) {
            shared_positions[threadIdx.x] = make_float3(
                positions[shared_idx * 3],
                positions[shared_idx * 3 + 1],
                positions[shared_idx * 3 + 2]
            );
            shared_masses[threadIdx.x] = (float)masses[shared_idx];
        }
        __syncthreads();

        #pragma unroll 8
        for (int j = 0; j < 256 && tile * 256 + j < num_nodes; j++) {
            if (tile * 256 + j == idx) continue;

            float3 pos_j = shared_positions[j];
            float mass_j = shared_masses[j];
            float3 diff = make_float3(
                pos_i.x - pos_j.x,
                pos_i.y - pos_j.y,
                pos_i.z - pos_j.z
            );

            float dist = fmaxf(sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z), 0.0001f);
            float force_mag = repulsion * mass_i * mass_j / (dist * dist);
            force.x += force_mag * diff.x / dist;
            force.y += force_mag * diff.y / dist;
            force.z += force_mag * diff.z / dist;
        }
        __syncthreads();
    }

    int vel_idx = idx * 3;
    float3 vel = make_float3(
        velocities[vel_idx],
        velocities[vel_idx + 1],
        velocities[vel_idx + 2]
    );

    vel.x = (vel.x + force.x) * damping;
    vel.y = (vel.y + force.y) * damping;
    vel.z = (vel.z + force.z) * damping;

    pos_i.x += vel.x;
    pos_i.y += vel.y;
    pos_i.z += vel.z;

    positions[pos_idx] = pos_i.x;
    positions[pos_idx + 1] = pos_i.y;
    positions[pos_idx + 2] = pos_i.z;

    velocities[vel_idx] = vel.x;
    velocities[vel_idx + 1] = vel.y;
    velocities[vel_idx + 2] = vel.z;
}
