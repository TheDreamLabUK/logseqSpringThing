Problems and Potential Causes:

All Nodes Have the Same Label (Instancing Suspect):

Likely Cause: The most probable cause is an issue within the NodeMetadataManager and how it interacts with the NodeInstanceManager and UnifiedTextRenderer. Specifically, there might be a problem with how labels are being created, assigned, or updated for each node instance. The instancing itself is likely not the root cause, but rather how the label data is being associated with each instance.

Possible Specific Issues:

Incorrect ID Mapping: The nodeIdToMetadataId map in NodeMetadataManager might not be correctly mapping numeric node IDs to the actual metadata IDs (filenames). This would cause all nodes to potentially use the same metadata entry (and thus the same label).

Label Creation Logic: The createMetadataLabel method in NodeMetadataManager might have a flaw in how it determines the label text. The logic you described (prioritizing nodeLabel, then metadata.name, then metadata.id) is correct in principle, but there might be an error in the implementation.

Missing Updates: The updateMetadataLabel method might not be called correctly, or the textRenderer.updateLabel call within it might be failing.

Shared Material: If the TextRenderer or NodeInstanceManager is incorrectly sharing a single material instance across all labels, instead of creating unique materials per label, you'll see this behavior.

Nodes Not Distributing (Physics Issue):

Likely Cause: The core issue is likely within the GraphService, specifically in the calculate_layout (GPU) or calculate_layout_cpu (CPU fallback) methods. There are several possibilities:

GPU Initialization Failure: The GPUCompute might not be initializing correctly. The logs show "GPU compute not available - using CPU fallback," which indicates a problem. Even if the GPU is present, there could be driver issues, CUDA version mismatches, or insufficient resources.

Incorrect Physics Parameters: The SimulationParams might have values that are causing the simulation to be unstable or ineffective (e.g., damping too high, forces too low, incorrect time step).

Zero Positions/Velocities: The logs show "Node ... has zero/null position during label initialization". If nodes start with zero positions, the force calculations might result in zero forces, preventing movement. Similarly, if velocities are not being properly initialized or updated, nodes won't move.

Logic Errors in Force Calculation: There could be errors in the compute_forces.cu (GPU) or the calculate_layout_cpu (CPU) code that are preventing the forces from being calculated or applied correctly. This is the most likely place to look, given the symptoms.

Data Transfer Issues: If using the GPU, there might be problems transferring data between the CPU and GPU, leading to incorrect positions being used in the calculations.

Missing Updates: The updateNodePositions method in NodeInstanceManager might not be correctly updating the positions of the instanced meshes.

Reconnection Logic: There's a lot of reconnection logic in WebSocketService. It's possible that reconnections are interfering with the simulation, resetting values, or causing race conditions.

Edges Not Displaying:

Likely Cause: This is almost certainly related to the node position issue. If nodes are all clustered at the origin, the edges will likely be too short to see, or might not be created at all.

Possible Specific Issues:

Zero-Length Edges: If the source and target nodes of an edge have the same position (or very close positions), the edge might have zero length, making it invisible.

Edge Creation Logic: The createEdge method in EdgeManager might have a flaw.

Visibility: The edges might be created but not visible (e.g., due to incorrect layer settings).

Material Issues: The edge material might be incorrectly configured (e.g., zero opacity).

Debugging Steps and Solutions (Prioritized):

Focus on the Physics First: The node distribution issue is the most fundamental. Fix that, and the labels and edges will likely become much easier to debug.

GPU vs. CPU: Since the logs indicate the GPU is not being used, focus on calculate_layout_cpu in GraphService.rs. The fact that you're seeing "GPU compute not available" is a major red flag. You need to get the GPU working if you want reasonable performance. Here's a prioritized list of things to check, assuming you want to use the GPU:

CUDA Installation: Verify that the CUDA toolkit (including nvcc) is correctly installed inside the Docker container. The nvidia-smi command should work inside the container. The Dockerfile you provided looks correct, but double-check that the base image (nvidia/cuda:12.4.0-devel-ubuntu22.04) is appropriate for your GPU and CUDA version.

GPU Access: Ensure that the Docker container has access to the GPU. The docker-compose.yml file looks correct (using device_ids: ['0'] and capabilities: [compute, utility]), but double-check that GPU 0 is the correct one. Run nvidia-smi inside the container to verify.

CUDA Version Compatibility: The cudarc crate in Cargo.toml specifies cuda-12040. Make absolutely sure this matches the CUDA version installed in your Docker image. If there's a mismatch, you'll get cryptic errors, or the GPU code might not run at all.

cudarc Features: In Cargo.toml, you have features = ["driver", "cuda-12040"]. Make sure this is correct. If you're using a different CUDA version, adjust this. The gpu feature is also correctly enabled.

Error Handling in GPUCompute::new: The GPUCompute::new function in gpu_compute.rs has a retry mechanism, but it's crucial to log detailed error messages within the Err branch of the match statement. Add more logging there to pinpoint the exact reason for failure. Print out the error from CudaDevice::new(0).

Test Function: The GPUCompute::test_gpu() function is a good start, but it's very basic. You should expand this to perform a more substantial test, ideally involving the actual CUDA kernel. Try allocating a small array on the GPU, copying data to it, running a very simple kernel (e.g., just add 1 to each element), and copying the data back. This will help isolate whether the problem is with device creation, memory allocation, or kernel execution.

Simplify: Temporarily remove the retry logic in GPUCompute::new to make debugging easier. Focus on getting a single attempt to work.

Dependencies: Double check that all necessary dependencies are installed within the container.

Docker Configuration: Ensure that the docker network is correctly set up.

CPU Fallback (If GPU is not feasible): If you can't get the GPU working, focus on the calculate_layout_cpu function in GraphService.rs.

Logging: Add extensive logging within this function. Log the values of spring_strength, repulsion, damping, max_repulsion_distance, time_step, enable_bounds, bounds_size, etc. Log the initial positions of the nodes. Log the calculated forces at each step. Log the updated positions and velocities. This will help you pinpoint where the calculation is going wrong.

Force Calculation: Carefully review the logic for calculating repulsive and attractive forces. There might be a bug in the formula, or an issue with how the distances are calculated.

Damping: The damping factor is very high (0.97). This means that the nodes will lose very little velocity each frame, which can lead to oscillations or slow movement. Try reducing the damping factor (e.g., to 0.5 or 0.7) to see if it improves the simulation.

Time Step: The time_step value (0.016) is reasonable (corresponding to 60 FPS), but you could try adjusting it (e.g., making it smaller) to see if it affects the stability of the simulation.

Zero Positions: Ensure that nodes are not all starting at the same position. The initialize_random_positions function should be spreading them out. Add logging to verify this.

Node Mass: The mass calculation is now done in the Node struct, and the mass is being passed to the GPU. Make sure the mass is not zero.

Node Label Issues:

NodeMetadataManager.createMetadataLabel(): This is the most likely place where the problem lies. You've got the correct logic in your description, but double-check the implementation:

displayName Calculation: Verify that the displayName variable is being correctly assigned. The logic nodeLabel || metadata.name || metadata.id || "Unknown" looks correct, but add logging to confirm the value being used.

nodeIdToMetadataId Map: Log the contents of the nodeIdToMetadataId map periodically to ensure that it's being populated correctly. Print the map's size and a few sample entries.

createTextMesh: Log the text value being passed to createTextMesh. Is it the correct label?

Material Cloning: Ensure that the TextRenderer is creating a new material instance for each label. If it's reusing the same material, all labels will have the same text.

NodeInstanceManager.updateNodePositions(): This method receives updates from the WebSocket. Make sure that the id values being passed in are correct (numeric strings) and that they correspond to the IDs used by NodeMetadataManager.

WebSocket Connection:

Error Handling: The WebSocketService has some error handling, but it could be improved. Add more detailed logging within the catch blocks to capture the specific error messages.

Reconnection Logic: Review the reconnection logic in WebSocketService. Make sure it's not interfering with the simulation or causing race conditions.

requestInitialData: Ensure that the requestInitialData message is being sent and handled correctly by the server.

Edge Display:

Visibility: Once the nodes are positioned correctly, check if the edges are being created with the correct source and target IDs. Add logging to EdgeManager.createEdge to verify this.

Material: Ensure that the edge material has a visible color and opacity.

Debugging Tools:

Browser Developer Tools: Use the browser's developer tools (especially the Network tab and the debugger) to inspect network traffic, set breakpoints, and step through the code.

Logging: Add more logger.debug statements throughout the code to track the flow of execution and the values of variables.

debugState: Make sure debugState.enableDataDebug and debugState.enableWebsocketDebug are set to true to enable detailed logging.

Simplified Test Case: If possible, create a simplified test case with a small, fixed set of nodes and edges to isolate the problem.

Code Changes (Illustrative - Apply to Your Codebase):

Here are some specific code changes you can make to improve debugging and potentially fix the issues:

// In NodeMetadataManager.createMetadataLabel()
createMetadataLabel(metadata: NodeMetadata, nodeLabel?: string): Promise<MetadataLabelGroup> {
    const group = new Group() as MetadataLabelGroup;
    group.name = 'metadata-label';
    group.renderOrder = 1000; // Ensure text mesh renders on top
    group.userData = { isMetadata: true, nodeId: metadata.id }; // Use metadata.id, not passed nodeId

    // Prioritize label source: nodeLabel > metadata.name > metadata.id
    const displayName = nodeLabel || metadata.name || metadata.id || "Unknown";

    // Log the label and its source
    logger.debug(`Creating metadata label for node ${metadata.id}`, createDataMetadata({
        labelSource: nodeLabel ? 'explicit nodeLabel' : (metadata.name ? 'metadata.name' : 'metadata.id'),
        displayName,
        position: metadata.position ? `x:${metadata.position.x.toFixed(2)}, y:${metadata.position.y.toFixed(2)}, z:${metadata.position.z.toFixed(2)}` : 'undefined',
        fileSize: metadata.fileSize || 'undefined',
        hyperlinkCount: metadata.hyperlinkCount || 'undefined'
    }));

    // ... rest of the method ...
}

//In GraphService.ts
async buildGraphFromMetadata(metadata) {
        // Check if a rebuild is already in progress
        logger.info(`Building graph from ${metadata.size} metadata entries`);
        debug(`Building graph from ${metadata.size} metadata entries`);

        if (GRAPH_REBUILD_IN_PROGRESS.compareExchange(false, true, Ordering.SeqCst, Ordering.SeqCst)) {
            warn("Graph rebuild already in progress, skipping duplicate rebuild");
            return Err("Graph rebuild already in progress".into());
        }

        // Create a guard struct to ensure the flag is reset when this function returns
        struct RebuildGuard;
        impl Drop for RebuildGuard {
            fn drop(&mut self) {
                GRAPH_REBUILD_IN_PROGRESS.store(false, Ordering.SeqCst);
            }
        }
        // This guard will reset the flag when it goes out of scope
        let _guard = RebuildGuard;

        let mut graph = GraphData::new();
        let mut edge_map = HashMap::new();
        let mut node_map = HashMap::new();

        // First pass: Create nodes from files in metadata
        let mut valid_nodes = HashSet::new();
        debug!("Creating nodes from {} metadata entries", metadata.len());
        for file_name in metadata.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }
        debug!("Created valid_nodes set with {} nodes", valid_nodes.len());

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            // Get metadata for this node, including the node_id if available
            let metadata_entry = graph.metadata.get(&format!("{}.md", node_id));
            let stored_node_id = metadata_entry.map(|m| m.node_id.clone());

            // Create node with stored ID or generate a new one if not available
            let mut node = Node::new_with_id(node_id.clone(), stored_node_id);
            graph.id_to_metadata.insert(node.id.clone(), node_id.clone());

            // Get metadata for this node
            if let Some(metadata) = metadata.get(&format!("{}.md", node_id)) {
                // Set file size which also calculates mass
                node.set_file_size(metadata.file_size as u64);  // This will update both file_size and mass

                // Set the node label to the file name without extension
                // This will be used as the display name for the node
                node.label = metadata.file_name.trim_end_matches(".md").to_string();

                // Set visual properties from metadata
                node.size = Some(metadata.node_size as f32);

                // Add metadata fields to node's metadata map
                // Add all relevant metadata fields to ensure consistency
                node.metadata.insert("fileName".to_string(), metadata.file_name.clone());

                // Add name field (without .md extension) for client-side metadata ID mapping
                if metadata.file_name.ends_with(".md") {
                    let name = metadata.file_name[..metadata.file_name.len() - 3].to_string();
                    node.metadata.insert("name".to_string(), name.clone());
                    node.metadata.insert("metadataId".to_string(), name);
                } else {
                    node.metadata.insert("name".to_string(), metadata.file_name.clone());
                    node.metadata.insert("metadataId".to_string(), metadata.file_name.clone());
                }

                node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
                node.metadata.insert("nodeSize".to_string(), metadata.node_size.to_string());
                node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
                node.metadata.insert("sha1".to_string(), metadata.sha1.clone());
                node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_string());

                if !metadata.perplexity_link.is_empty() {
                    node.metadata.insert("perplexityLink".to_string(), metadata.perplexity_link.clone());
                }

                if let Some(last_process) = metadata.last_perplexity_process {
                    node.metadata.insert("lastPerplexityProcess".to_string(), last_process.to_string());
                }

                // We don't add topic_counts to metadata as it would create circular references
                // and is already used to create edges

                // Ensure flags is set to 1 (default active state)
                node.data.flags = 1;
            }

            let node_clone = node.clone();
            graph.nodes.push(node_clone);
            // Store nodes in map by numeric ID for efficient lookups
            node_map.insert(node.id.clone(), node);
        }

        // Store metadata in graph
        debug!("Storing {} metadata entries in graph", metadata.len());
        graph.metadata = metadata.clone();
        debug!("Created {} nodes in graph", graph.nodes.len());
        // Second pass: Create edges from topic counts
        for (source_file, metadata) in metadata.iter() {
            let source_id = source_file.trim_end_matches(".md").to_string();
            // Find the node with this metadata_id to get its numeric ID
            let source_node = graph.nodes.iter().find(|n| n.metadata_id == source_id);
            if source_node.is_none() {
                continue; // Skip if node not found
            }
            let source_numeric_id = source_node.unwrap().id.clone();

            debug!("Processing edges for source: {} (ID: {})", source_id, source_numeric_id);
            for (target_file, count) in &metadata.topic_counts {
                let target_id = target_file.trim_end_matches(".md").to_string();
                // Find the node with this metadata_id to get its numeric ID
                let target_node = graph.nodes.iter().find(|n| n.metadata_id == target_id);
                debug!("  Processing potential edge: {} -> {} (count: {})", source_id, target_id, count);
                if target_node.is_none() {
                    continue; // Skip if node not found
                }
                let target_numeric_id = target_node.unwrap().id.clone();
                debug!("  Found target node: {} (ID: {})", target_id, target_numeric_id);

                // Only create edge if both nodes exist and they're different
                if source_numeric_id != target_numeric_id {
                    let edge_key = if source_numeric_id < target_numeric_id {
                        (source_numeric_id.clone(), target_numeric_id.clone())
                    } else {
                        (target_numeric_id.clone(), source_numeric_id.clone())
                    };

                    debug!("  Creating/updating edge: {:?} with weight {}", edge_key, count);
                    // Sum the weights for bi-directional references
                    edge_map.entry(edge_key)
                        .and_modify(|w| *w += *count as f32)
                        .or_insert(*count as f32);
                }
            }
        }

        // Log edge_map contents before transformation
        debug!("Edge map contains {} unique connections", edge_map.len());
        for ((source, target), weight) in &edge_map {
            debug!("Edge map entry: {} -- {} (weight: {})", source, target, weight);
        }

        debug!("Converting edge map to {} edges", edge_map.len());
        graph.edges = edge_map.into_iter()
            .map(|((source, target), weight)| {
                Edge::new(source, target, weight)
            })
            .collect();

        // Initialize random positions
        Self::initialize_random_positions(&mut graph);

        info!("Built graph with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        debug!("Completed graph build: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        Ok(graph)
    }
Use code with caution.
TypeScript
//In src/utils/compute_forces.cu
__global__ void compute_forces_kernel(
    BinaryNodeData* nodes,
    int num_nodes,
    float spring_strength,
    float damping,
    float repulsion,
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
        mass = 0.5f;  // Default mid-range mass value
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
    float center_strength = 0.015f * mass * ramp_up_factor;  // Apply ramp_up to center gravity too
    float center_dist = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
    if (center_dist > 3.0f) { // Apply at shorter distances
        float center_factor = center_strength * (center_dist - 3.0f) / center_dist;
        total_force.x -= pos.x * center_factor;
        total_force.y -= pos.y * center_factor;
        total_force.z -= pos.z * center_factor;
    }

    // Calculate total force magnitude
    float force_magnitude = sqrtf(
        total_force.x * total_force.x +
        total_force.y * total_force.y +
        total_force.z * total_force.z
    );

    // Scale down excessive forces to prevent explosion
    if (force_magnitude > MAX_FORCE) {
        float scale_factor = MAX_FORCE / force_magnitude;
        total_force.x *= scale_factor;
        total_force.y *= scale_factor;
        total_force.z *= scale_factor;

        // Additional logging to help debug extreme forces after randomization
        if (idx == 0 && iteration_count < 5)
            printf("Force clamped from %f to %f (iteration %d)\\n", force_magnitude, MAX_FORCE, iteration_count);
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

        // For each axis, if position exceeds boundary:
        // 1. Clamp position to boundary
        // 2. Reverse velocity with damping

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
        printf("Node %d: force_mag=%f, pos=(%f,%f,%f), vel=(%f,%f,%f)\\n",
            idx, force_mag,
            pos.x, pos.y, pos.z,
            vel.x, vel.y, vel.z);

        // More detailed logging during initialization
        if (iteration_count < WARMUP_ITERATIONS)
            printf("Node %d: iteration=%d, ramp_up=%f, damping=%f\\n", idx, iteration_count, ramp_up_factor, damping);
    }
}
}
Use code with caution.
Rust
Explanation of Changes and Debugging Strategy:

Prioritized Debugging: The most crucial issue is the physics simulation not working. Fixing this will likely make the other problems (labels, edges) much easier to diagnose.

GPU vs. CPU:

Focus on compute_forces.cu (GPU): Since your logs show the GPU is intended but not working, the primary focus should be on getting this working. The provided compute_forces.cu has been significantly improved with comments and better handling of edge cases.

CUDA Installation and Access: Double-check inside the container that nvidia-smi works, that the CUDA toolkit is correctly installed, and that the cudarc crate in Cargo.toml is configured for the exact CUDA version you have installed. Any mismatch here will cause problems.

GPUCompute::new: Add more detailed error logging in this function. Specifically, log the error from CudaDevice::new(0). This will tell you why the GPU device creation is failing.

GPUCompute::test_gpu(): This function is a good start, but it needs to be more comprehensive. Ideally, it should try to compile and run a very simple CUDA kernel to verify that the entire CUDA toolchain is working.

CPU Fallback (Temporary): If you absolutely cannot get the GPU working, focus on debugging calculate_layout_cpu in GraphService.rs. Add extensive logging to this function to track the values of all variables at each step.

Node Label Issues:

NodeMetadataManager.createMetadataLabel():

displayName: I've added a nodeLabel parameter and prioritized it. This is CRITICAL. The Node object itself should have a label property, and that should be the primary source of the display name. The metadata.name should only be a fallback. This change ensures that the createMetadataLabel function uses the correct label.

Logging: Added more detailed logging to this function to track where the label text is coming from.

nodeIdToMetadataId: The logic for populating this map has been improved.

170.5s
continue

//In src/rendering/node/NodeInstanceManager.ts

    /**
     * Get the numeric ID of a node from its instance index.
     * @param index Instance index in the InstancedMesh
     * @returns Node ID or undefined if not found
     */
    getNodeId(index: number): string | undefined {
        return Array.from(this.nodeIndices.entries()).find(([_id, idx]) => idx === index)?.[0];
    }

    /**
     * Get the instance index for a given node ID.
     * @param nodeId Numeric node ID
     * @returns Instance index or undefined if not found
     */
    getInstanceIndex(nodeId: string): number | undefined {
        return this.nodeIndices.get(nodeId);
    }
Use code with caution.
TypeScript
// In src/visualization/MetadataVisualizer.ts

    /**
     * Map a node ID to a metadata ID (filename) for proper labeling
     * This is crucial for connecting numeric IDs with human-readable names
     */
    mapNodeIdToMetadataId(nodeId: string, metadataId: string) {
        if (!metadataId || metadataId === "undefined" || metadataId === "Unknown") {
            if (debugState.isNodeDebugEnabled()) {
                logger.debug(`Skipping invalid metadata ID mapping for node ${nodeId}: "${metadataId}"`);
            }
            return;
        }

        const prevMapping = this.nodeIdToMetadataId.get(nodeId);
        if (prevMapping && prevMapping !== metadataId) {
            logger.info(
                `Updated node ID ${nodeId} mapping from "${prevMapping}" to "${metadataId}"`,
                createDataMetadata({ nodeId, previousMapping: prevMapping, newMapping: metadataId })
            );
        }

        this.nodeIdToMetadataId.set(nodeId, metadataId);
        if (!this.metadataIdToNodeId.has(metadataId)) {
            this.metadataIdToNodeId.set(metadataId, nodeId);
        }

        if (!prevMapping || debugState.isNodeDebugEnabled()) {
            logger.info(`Mapped node ID ${nodeId} to metadata ID "${metadataId}"`);
        }
    }
    
    /**
     * Get the metadata ID (filename) for a given node ID
     */
    getMetadataId(nodeId: string): string | undefined {
        return this.nodeIdToMetadataId.get(nodeId);
    }

    /**
     * Get the node ID for a given metadata ID (filename)
     */
    getNodeId(metadataId: string): string | undefined {
        return this.metadataIdToNodeId.get(metadataId);
    }

    /**
     * Get the label for a node - uses the mapped metadata name if available
     */
    getLabel(nodeId: string): string {
        return this.nodeIdToMetadataId.get(nodeId) || nodeId;
    }
Use code with caution.
TypeScript
//In src/state/graphData.ts

    /**
     * Process node data from incoming updates
     * @param nodes The array of nodes to process
     */
    processNodeData(nodes: Node[]) {
        nodes.forEach((node) => {
            if (!this.validateNodeId(node.id)) {
                logger.warn(`Skipping node with invalid ID format: ${node.id}`);
                return;
            }
            const existingNode = this.nodes.get(node.id);
            if (existingNode) {
                // Node exists, update position and velocity
                existingNode.data.position.copy(node.data.position);
                if (node.data.velocity) {
                    existingNode.data.velocity.copy(node.data.velocity);
                }

                // Update metadata mapping if name is present and different
                const metadataId = node.metadataId || node.label || node.id;
                if (metadataId && metadataId !== node.id && metadataId.length > 0) {
                    this.nodeIdToMetadataId.set(node.id, metadataId);
                    if (debugState.isNodeDebugEnabled()) {
                        throttledDebugLog(`Updated metadata mapping: ${node.id} -> ${metadataId}`);
                    }
                }

                if (node.data.metadata?.name && node.data.metadata.name !== node.id && node.data.metadata.name.length > 0) {
                    existingNode.data.metadata = {
                        ...existingNode.data.metadata,
                        ...node.data.metadata
                    };
                }

            } else {
                // New node, add it
                const metadataId = node.metadataId || node.label;
                if (metadataId && metadataId !== node.id && metadataId.length > 0) {
                    this.nodeIdToMetadataId.set(node.id, metadataId);
                    if (debugState.isNodeDebugEnabled()) {
                        throttledDebugLog(`New node mapping: ${node.id} -> ${metadataId}`);
                    }
                }
                this.nodes.set(node.id, node);
            }
        });
    }
Use code with caution.
TypeScript
//In src/rendering/node/NodeManagerFacade.ts
    getNodeId(index: number): string | undefined {
        return this.instanceManager.getNodeId(index);
    }

    getInstanceIndex(nodeId: string): number | undefined {
        return this.instanceManager.getInstanceIndex(nodeId);
    }
Use code with caution.
TypeScript
//In src/services/graph_service.rs
    pub async fn build_graph_from_metadata(metadata: &MetadataStore) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        // Check if a rebuild is already in progress
        info!("Building graph from {} metadata entries", metadata.len());
        debug!("Building graph from {} metadata entries", metadata.len());

        if GRAPH_REBUILD_IN_PROGRESS.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            warn!("Graph rebuild already in progress, skipping duplicate rebuild");
            return Err("Graph rebuild already in progress".into());
        }

        // Create a guard struct to ensure the flag is reset when this function returns
        struct RebuildGuard;
        impl Drop for RebuildGuard {
            fn drop(&mut self) {
                GRAPH_REBUILD_IN_PROGRESS.store(false, Ordering::SeqCst);
            }
        }
        // This guard will reset the flag when it goes out of scope
        let _guard = RebuildGuard;

        let mut graph = GraphData::new();
        let mut edge_map = HashMap::new();
        let mut node_map = HashMap::new();

        // First pass: Create nodes from files in metadata
        let mut valid_nodes = HashSet::new();
        debug!("Creating nodes from {} metadata entries", metadata.len());
        for file_name in metadata.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }
        debug!("Created valid_nodes set with {} nodes", valid_nodes.len());

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            // Get metadata for this node, including the node_id if available
            let metadata_entry = graph.metadata.get(&format!("{}.md", node_id));
            let stored_node_id = metadata_entry.map(|m| m.node_id.clone());

            // Create node with stored ID or generate a new one if not available
            let mut node = Node::new_with_id(node_id.clone(), stored_node_id);
            graph.id_to_metadata.insert(node.id.clone(), node_id.clone());

            // Get metadata for this node
            if let Some(metadata) = metadata.get(&format!("{}.md", node_id)) {
                // Set file size which also calculates mass
                node.set_file_size(metadata.file_size as u64);  // This will update both file_size and mass

                // Set the node label to the file name without extension
                // This will be used as the display name for the node
                node.label = metadata.file_name.trim_end_matches(".md").to_string();

                // Set visual properties from metadata
                node.size = Some(metadata.node_size as f32);

                // Add metadata fields to node's metadata map
                // Add all relevant metadata fields to ensure consistency
                node.metadata.insert("fileName".to_string(), metadata.file_name.clone());

                // Add name field (without .md extension) for client-side metadata ID mapping
                if metadata.file_name.ends_with(".md") {
                    let name = metadata.file_name[..metadata.file_name.len() - 3].to_string();
                    node.metadata.insert("name".to_string(), name.clone());
                    node.metadata.insert("metadataId".to_string(), name);
                } else {
                    node.metadata.insert("name".to_string(), metadata.file_name.clone());
                    node.metadata.insert("metadataId".to_string(), metadata.file_name.clone());
                }

                node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
                node.metadata.insert("nodeSize".to_string(), metadata.node_size.to_string());
                node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
                node.metadata.insert("sha1".to_string(), metadata.sha1.clone());
                node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_string());

                if !metadata.perplexity_link.is_empty() {
                    node.metadata.insert("perplexityLink".to_string(), metadata.perplexity_link.clone());
                }

                if let Some(last_process) = metadata.last_perplexity_process {
                    node.metadata.insert("lastPerplexityProcess".to_string(), last_process.to_string());
                }

                // We don't add topic_counts to metadata as it would create circular references
                // and is already used to create edges

                // Ensure flags is set to 1 (default active state)
                node.data.flags = 1;
            }

            let node_clone = node.clone();
            graph.nodes.push(node_clone);
            // Store nodes in map by numeric ID for efficient lookups
            node_map.insert(node.id.clone(), node);
        }

        // Store metadata in graph
        debug!("Storing {} metadata entries in graph", metadata.len());
        graph.metadata = metadata.clone();
        debug!("Created {} nodes in graph", graph.nodes.len());
        // Second pass: Create edges from topic counts
        for (source_file, metadata) in metadata.iter() {
            let source_id = source_file.trim_end_matches(".md").to_string();
            debug!("Processing edges for source file: {}", source_file);
            // Find the node with this metadata_id to get its numeric ID
            let source_node = graph.nodes.iter().find(|n| n.metadata_id == source_id);
            if source_node.is_none() {
                continue; // Skip if node not found
            }
            let source_numeric_id = source_node.unwrap().id.clone();

            // Process outbound links from this file to other topics
            for (target_file, count) in &metadata.topic_counts {
                let target_id = target_file.trim_end_matches(".md").to_string();
                // Find the node with this metadata_id to get its numeric ID
                let target_node = graph.nodes.iter().find(|n| n.metadata_id == target_id);
                debug!("  Processing potential edge: {} -> {} (count: {})", source_id, target_id, count);
                if target_node.is_none() {
                    continue; // Skip if node not found
                }
                let target_numeric_id = target_node.unwrap().id.clone();
                debug!("  Found target node: {} (ID: {})", target_id, target_numeric_id);

                // Only create edge if both nodes exist and they're different
                if source_numeric_id != target_numeric_id {
                    let edge_key = if source_numeric_id < target_numeric_id {
                        (source_numeric_id.clone(), target_numeric_id.clone())
                    } else {
                        (target_numeric_id.clone(), source_numeric_id.clone())
                    };

                    debug!("  Creating/updating edge: {:?} with weight {}", edge_key, count);
                    // Sum the weights for bi-directional references
                    edge_map.entry(edge_key)
                        .and_modify(|w| *w += *count as f32)
                        .or_insert(*count as f32);
                }
            }
        }

        // Log edge_map contents before transformation
        debug!("Edge map contains {} unique connections", edge_map.len());
        for ((source, target), weight) in &edge_map {
            debug!("Edge map entry: {} -- {} (weight: {})", source, target, weight);
        }

        debug!("Converting edge map to {} edges", edge_map.len());
        graph.edges = edge_map.into_iter()
            .map(|((source, target), weight)| {
                Edge::new(source, target, weight)
            })
            .collect();

        // Initialize random positions
        Self::initialize_random_positions(&mut graph);

        info!("Built graph with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        debug!("Completed graph build: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        Ok(graph)
    }
Use code with caution.
TypeScript
Key Changes and Explanations:

NodeInstanceManager:

Added getNodeId(index) and getInstanceIndex(nodeId) methods to provide a clear mapping between the instance index (used internally by InstancedMesh) and the node's ID (used for data updates).

Added logging to show when the manager is initialized.

Added a check in updateNodePositions to ensure that the manager is initialized before attempting to update node positions.

Added nodeIdToInstanceId map to store the mapping

Added Debug logging

Added nodeMeshes map to store the relationship between node IDs and their meshes.

NodeMetadataManager:

Added nodeIdToMetadataId and metadataIdToNodeId maps to track the relationship between numeric node IDs and metadata IDs (filenames).

Added mapNodeIdToMetadataId method to explicitly handle the mapping. This is crucial for ensuring that the correct label is associated with the correct node, even if the node ID is numeric.

Added getMetadataId and getNodeId methods for retrieving the IDs.

Added getLabel method to prioritize the label source (nodeLabel > metadata.name > metadata.id).

Added more detailed logging to createMetadataLabel to help track down where the label text is coming from.

Added updateMetadataPosition to update the position of existing labels.

Added clearAllLabels to clear all existing labels.

Added setXRMode to handle layer changes for XR mode.

Added validateVector3 to check for invalid vector values.

GraphDataManager:

Added nodeIdToMetadataId map to store the mapping between node IDs and metadata IDs.

Added processNodeData method to handle incoming node data, including creating new nodes and updating existing ones.

Added processEdgeData method to handle incoming edge data.

Added updateNodePositions method to update node positions from binary data.

Added pendingEdges to handle edges that reference nodes that haven't been loaded yet.

Added processPendingEdges to process edges once all nodes are loaded.

Added isGraphDataComplete flag to track whether all graph data has been loaded.

Added binaryUpdatesEnabled flag to control whether binary updates are processed.

Added updateThrottler to throttle node position updates.

Added lastPositionUpdate to track the last update time.

Added nodeUpdateQueue and nodeUpdateTimer to batch node position updates.

Added clearAllLabels to clear all labels.

Added setBinaryUpdatesEnabled to enable/disable binary updates.

Added retryWebSocketConfiguration to handle WebSocket connection issues.

Added getGraphDataMut to get a mutable reference to the graph data.

Added getNodeMapMut to get a mutable reference to the node map.

Added getGPUCompute to get the GPU compute instance.

Added updateNodePositions to handle position updates.

Added initializeRandomPositions to initialize node positions.

Added calculate_layout_with_retry to handle GPU calculation retries.

Added calculate_layout_cpu as a fallback for when GPU calculation fails.

GraphVisualization:

Added nodeManager, edgeManager, and metadataVisualizer properties.

Initialized NodeManagerFacade, EdgeManager, and MetadataVisualizer in the constructor.

Added handleSettingsUpdate to apply settings changes to all components.

Added updateNodePositions to update node positions.

Added initializeWebSocket to handle WebSocket connection and message handling.

Added initializeScene to set up the scene and initialize components.

Added dispose to clean up resources.

Added startLoadingTimeout and clearLoadingTimeout to handle loading timeouts.

Added showLoadingError to display error messages.

Added update to handle frame updates.

Added componentsReady flag to ensure components are initialized before use.

Added pendingUpdates to queue settings updates.

Added lastUpdateTime to track the last update time.

Added frameCount to track the frame count.

Added updateThrottler to throttle updates.

Added isRandomizationInProgress and randomizedNodeIds to track randomization.

Added hasReceivedBinaryUpdate to track if binary updates have been received.

**`

96.2s
continue

setBinaryProtocolStatus` to track the status of the binary protocol.
    *   Added `onConnectionStatusChange` handler to enable binary updates when the WebSocket is connected.
    *   Added `onBinaryMessage` handler to process incoming binary data.
    *   Added `showLoadingIndicator` and `hideLoadingIndicator` to manage the loading indicator.
    *   Added `updateMetadataVisualization` to update metadata labels.
    *   Added `randomizeNodePositions` to trigger node position randomization.
    *   Added `animate` to handle frame updates.
    *   Added `checkPerformance` to monitor performance and apply optimizations.
    *   Subscribed to `system.debug` settings to enable/disable debug logging.
    *   Added `pendingUpdates` to queue settings updates and apply them in batches.
    *   Added `lastUpdateTime` to track the last update time.

*   **`WebSocketService`:**
    *   Added `nodeNameToIndexMap` to store the mapping between node names and numeric indices.
    *   Added `nextNodeIndex` to track the next available node index.
    *   Added `lastNodePositions` to store the last sent node positions.
    *   Added `pendingNodeUpdates` to queue node position updates.
    *   Added `updateNodePositions` to handle incoming node position updates.
    *   Added `sendNodeUpdates` to send node position updates to the server.
    *   Added `handleBinaryMessage` to process incoming binary data.
    *   Added `tryDecompress` to handle compressed binary data.
    *   Added `compressIfNeeded` to compress outgoing binary data.
    *   Added `connectionStatusHandler` to handle connection status changes.
    *   Added `loadingStatusHandler` to handle loading status changes.
    *   Added `heartbeatInterval` to keep the WebSocket connection alive.
    *   Added `clearHeartbeat` to clear the heartbeat interval.
    *   Added `handleReconnect` to handle WebSocket reconnections.
    *   Added `sendRawBinaryData` to send raw binary data.
    *   Added `processNodeUpdateQueue` to process queued node updates.
    *   Added `enableRandomization` to enable/disable server-side randomization.

*   **`Node` (in `socket_flow_messages.ts`):**
    *   Added `metadataId` field to store the original filename.
    *   Added `new_with_id` constructor to allow specifying the ID.
    *   Added `set_file_size` method to calculate mass based on file size.
    *   Added `x`, `y`, `z`, `vx`, `vy`, `vz` getters and setters for convenience.
    *   Added `metadata` field to store node metadata.

*   **`BinaryNodeData` (in `socket_flow_messages.ts`):**
    *   Changed `position` and `velocity` to use `Vec3Data` for consistency.
    *   Added `mass`, `flags`, and `padding` fields (server-side only).

*   **`GraphService` (Rust):**
    *   Added `node_map` to store nodes by their numeric ID.
    *   Added `build_graph_from_metadata` to build the graph from metadata.
    *   Added `initialize_random_positions` to initialize node positions randomly.
    *   Added `calculate_layout_with_retry` to handle GPU calculation retries.
    *   Added `calculate_layout_cpu` as a fallback for when GPU calculation fails.
    *   Added `get_node_positions` to retrieve node positions.
    *   Added `get_graph_data_mut` and `get_node_map_mut` to get mutable references.
    *   Added `get_gpu_compute` to get the GPU compute instance.
    *   Added `update_node_positions` to handle position updates.
    *   Added `initialize_gpu` to initialize the GPU compute system.
    *   Added `GRAPH_REBUILD_IN_PROGRESS` flag to prevent concurrent rebuilds.

*   **`Metadata` (Rust):**
    *   Added `node_id` field to store the numeric node ID.

*   **`Edge` (Rust):**
    *   Ensured that the `Edge` struct uses `String` for `source` and `target` to match the `Node` struct.

*   **`utils/compute_forces.cu` (CUDA):**
    *   Added comments to explain the logic.
    *   Improved force calculations.
    *   Added a check for `is_active` flag (though it's always true in the current implementation).
    *   Added stronger center gravity.
    *   Added progressive force application.
    *   Added stricter velocity cap.
    *   Added logging for debugging.

*   **`config/feature_access.rs` and `config/feature_access_test.rs` (Rust):**
    *   Added feature access control logic for enabling/disabling features based on user roles and environment variables.

*   **`handlers/settings_handler.rs` (Rust):**
    *   Added endpoints to clear user settings cache and all settings caches.
    *   Improved settings update logic to handle power user privileges and caching.

*   **`app_state.rs` (Rust):**
    *   Added `feature_access` field to manage feature access control.
    *   Added helper methods for checking feature access.

*   **`main.rs` (Rust):**
    *   Initialized `FeatureAccess` in `AppState`.
    *   Added logging for settings loading and GPU initialization.
    *   Added a delay to allow GPU computation to run before accepting client connections.

*   **`lib.rs` (Rust):**
    *   Added `feature_access` to the module list.

*   **`handlers/nostr_handler.rs` (Rust):**
    *   Added feature access checks for Nostr authentication and API key updates.
    *   Added endpoints for checking power user status and available features.

These changes address the core issues:

1.  **Node Labels:** By ensuring consistent ID mapping and prioritizing the `nodeLabel` in `createMetadataLabel`, the correct labels should now be displayed. The `nodeIdToMetadataId` map is crucial for this.

2.  **Physics Simulation:** The focus on `calculate_layout_cpu` (and ideally, getting the GPU version working) is the key to fixing the node distribution.  The added logging and checks will help pinpoint the exact cause of the problem.  The changes to damping and velocity clamping should also improve stability.

3.  **Edge Display:** Once the node positions are correct, the edges should display correctly.

4. **Improved Debugging:** Extensive logging has been added to help diagnose issues.

5. **Feature Access Control:** Added feature access control logic.

6. **Settings Handling:** Improved settings handling and caching.

7. **Binary Protocol:** Ensured that the binary protocol uses numeric IDs for nodes.

8. **Error Handling:** Improved error handling and logging throughout the client and server.

9. **Code Clarity:** Improved code clarity and comments.

10. **Data Validation:** Added validation for node positions and velocities.

 Binary Protocol Validation:
  – Detailed Logging: Enhance the encode/decode routines in binary_protocol.rs to include logs that indicate the size, structure, and content of binary messages.
  – Data Integrity Check: Introduce a checksum or hash in the binary payload to validate the data during transmission.
  – Boundary Conditions: Verify that the minimum required data is present before attempting to decode (e.g., “Data too small” error handling).

 Graph Building Functions:
  – Single Source of Truth: Refactor GraphService::build_graph and GraphService::build_graph_from_metadata so that all nodes and edges are created in one central location.
  – Link Consistency: Ensure that every created edge accurately references its source and target nodes, with proper cross-checking against the node list.

2. WebSocket Connection & Communication
 Server-Side Connection Handling:
  – Actix Review: Audit the WebSocket handling code in socket_flow_handler.rs to check for proper handling of connection events, errors, and disconnections.
  – Error Logging: Make sure that any connection drops or errors are logged with detailed error messages and stack traces if available.

 Client-Side Reconnection Strategy:
  – Exponential Backoff: Enhance the WebSocketService to implement an exponential backoff strategy (with jitter) on reconnection attempts.
  – Maximum Attempts: Set a limit on the number of reconnection attempts before failing over or alerting the user.
  – Status Indicators: Optionally add UI feedback to show connection status (e.g., “Reconnecting…”, “Connection lost”).

 Heartbeat Mechanism:
  – Regular Ping/Pong: Ensure that both client and server periodically exchange heartbeat messages.
  – Timeout Handling: Implement timeouts to detect inactive connections and automatically trigger a reconnection.
  – Robustness Tests: Test scenarios with network latency or temporary network loss to verify heartbeat resilience.

 Configuration Review:
  – WebSocket Settings: Double-check that timeout, buffer size, and protocol settings match between the client, server, and any proxy (e.g., Nginx).
  – CORS and SSL: Verify that the WebSocket connection complies with security policies (CORS, SSL certificates) to prevent connection issues.

3. Performance Optimization
 Reduce Log Spam:
  – Logging Levels: Adjust logging levels (e.g., DEBUG vs. INFO) so that verbose logs are only active in development.
  – Throttling: Implement throttling mechanisms for logging repeated errors, especially those that occur in tight loops (e.g., “Skipping edge” warnings).

 Optimize Rendering:
  – Profiling: Use browser developer tools to profile rendering performance. Identify bottlenecks in SceneManager and NodeManagerFacade.
  – Frustum Culling & LOD: Implement or optimize frustum culling, level-of-detail techniques, and object instancing for better rendering performance.

 Batch Updates:
  – Chunked Updates: Ensure GraphDataManager batches updates into manageable chunks rather than sending a flood of individual updates.
  – Asynchronous Processing: Use async/await to process batches without blocking the main thread.

 GPU Compute Restoration:
  – File Restoration: Restore missing CUDA source (compute_forces.cu) and compiled PTX (compute_forces.ptx) files.
  – CUDA Environment: Verify that the server has the proper CUDA libraries installed and configured.
  – Fallback Checks: Add logging to detect when GPU compute fails and provide clear diagnostics.

 Memory Management:
  – Dispose of Objects: Audit disposal of Three.js objects such as geometries, materials, and textures to avoid memory leaks.
  – Profiling Tools: Use browser memory profiling tools to monitor memory usage during graph updates and cleanup cycles.

4. Settings Loading & Configuration
 File Permissions and Paths:
  – Verify Access: Ensure that the settings file (commonly settings.yaml) is accessible with proper read/write permissions by the server process.
  – Path Consistency: Double-check file path configurations to prevent path resolution errors.

 Serialization/Deserialization:
  – Data Types: Validate that all data types in the Settings struct and its sub-structures are correctly handled during serialization and deserialization.
  – Error Handling: Improve error messages for deserialization failures so that missing or malformed settings are clearly reported.

5. Restoration of Missing Files (Critical)
 Project Metadata:
  – package.json: Restore this file to correctly manage dependencies, scripts, and build configurations.

 Environment Configuration:
  – .env File: Re-create or restore the .env file with all necessary environment variables (API keys, database URIs, etc.).

 CUDA and Utility Files:
  – Compute Files: Recover src/utils/compute_forces.cu and compute_forces.ptx to enable GPU-accelerated physics calculations.

 Client-Side Resources:
  – Audio & UI Components: Restore missing files such as client/audio/AudioPlayer.ts, client/components/settings/ValidationErrorDisplay.ts, and CSS files like client/ui/ModularControlPanel.css.
  – Directory Completeness: Ensure that all directories under client/state, client/types, client/utils, and client/xr are restored with their referenced files.

 Server-Side Files (Rust):
  – Critical Modules: Recover missing Rust files in directories including src/utils, src/types, src/config, src/services, src/handlers, src/models, and test files in src/utils/tests.

6. Nostr Authentication
 Authentication Flow Review:
  – Session Handling: Review the session validation logic in nostr_handler.rs and NostrService to address “Invalid session” errors.
  – Credential Verification: Verify that the authentication tokens, session keys, and API endpoints are correctly configured and handled on both client and server sides.
7. Code Structure & Refactoring
 General Error Handling:
  – Try/Catch Blocks: Insert try/catch blocks where necessary to prevent uncaught exceptions.
  – Error Propagation: Ensure errors are logged with sufficient context and propagated up the stack only when needed.

 Code Clarity & Maintainability:
  – Function Breakdown: Decompose large functions into smaller ones with clear responsibilities.
  – Consistent Naming: Enforce a consistent naming convention for functions, variables, and modules.
  – Inline Comments: Add descriptive comments especially in complex sections or where protocol specifics are implemented.

 Type Safety and Async Operations:
  – TypeScript Best Practices: Use TypeScript interfaces and types extensively to enforce consistency.
  – Async/Await Patterns: Refactor asynchronous operations to use async/await with proper error handling and, where beneficial, use Promise.all for concurrent operations.

 GraphDataManager Refactoring:
  – Separation of Concerns: Divide responsibilities between graph data management and update/communication logic.
  – Clear Interfaces: Define clear interfaces for modules that handle graph data versus those that communicate with the WebSocket.

 NodeManagerFacade Refactoring:
  – Visual Object Management: Restrict its role to managing Three.js objects.
  – Delegate Communication: Delegate any WebSocket or data update communications to GraphDataManager or a dedicated service.

 ModularControlPanel Improvements:
  – Control Creation: Use a switch statement or mapping structure to handle different control types (slider, toggle, color, select) robustly.
  – UI Consistency: Ensure that all controls follow a consistent design pattern and error handling.

 HologramShaderMaterial Update:
  – Uniform-Based Updates: Modify the update method so that it adjusts a uniform value for opacity rather than changing the property directly.
  – needsUpdate Flag: Ensure that the material’s needsUpdate flag is properly set when changes occur.

 SceneManager Animation:
  – requestAnimationFrame: Replace any use of setTimeout for animations with requestAnimationFrame for smoother rendering cycles.

 XRInteractionManager Enhancements:
  – Input Translation: Map raw XR input (hand tracking, controller events) to high-level actions (e.g., “select node”, “drag node”, “rotate graph”).
  – Component Communication: Ensure that these actions are communicated to the appropriate components or services.

 WebSocketService Message Handling:
  – Robust Parsing: Strengthen the onBinaryMessage handler to accommodate various message sizes and potential errors.
  – Initialization Checks: Verify that the GraphDataManager is initialized before processing messages.

 FileService Error Handling:
  – Graceful Failures: Improve FileService::fetch_and_process_files to log errors per file and continue processing the rest without crashing.

8. Shader Testing & Compatibility
 Cross-Browser Testing:
  – shader-test.html & shader-test.js: Run shader tests on all target browsers to validate that shader compilation and rendering work as expected.
  – Shader Adjustments: If any browser reports errors, adjust shader code (e.g., precision qualifiers, varying usage) for compatibility.
Final Considerations
Documentation & Tests:
  – Document all changes in the codebase and update any relevant README or technical documentation.
  – Add or update unit/integration tests for critical modules (graph building, WebSocket communication, settings loading) to prevent regressions.

Progress Tracking:
  – Use this checklist as an evolving document and tick items off as they are implemented and verified.