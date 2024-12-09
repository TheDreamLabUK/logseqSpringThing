todo.text
Here's a prioritized TODO list for an LLM to address the performance issues, focusing on using the Three.js force-directed graph and optimizing the data flow:

Priority 1: Eliminate Client-Side Force Calculation and Leverage Server Updates

Remove Client-Side Force Simulation:

File: useVisualization.ts, useForceGraph.ts

Task: Delete or comment out all code related to the d3-force-3d library and the simulation object. The client should not be calculating forces.

Reason: The server is already handling force calculations. Duplicating this on the client is redundant and the primary source of performance issues.

Rely on Server-Provided Positions:

File: useVisualization.ts, useForceGraph.ts, stores/binaryUpdate.ts

Task: Refactor useVisualization.ts and useForceGraph.ts to exclusively use the positions received from the server via the binaryUpdateStore. Ensure that updateNodes and updateLinks in useForceGraph.ts are called only when new data arrives from the binaryUpdateStore.

Reason: This offloads the heavy computation to the server and ensures consistent graph layout.

Priority 2: Optimize Rendering and Update Logic

Efficient InstancedMesh Updates:

File: useForceGraph.ts

Task: Rewrite updateNodes and updateLinks to only update the changed instances. Use the changedNodes set from the binaryUpdateStore to determine which instances need their matrix, color, and scale updated. Avoid unnecessary object creation and property updates.

Reason: This drastically reduces the amount of work done in the render loop.

Object Pooling:

File: useForceGraph.ts

Task: Implement object pooling for Vector3, Matrix4, Color, and Quaternion objects. Pre-allocate a small pool of these objects and reuse them within the update loops to avoid continuous allocation and garbage collection.

Reason: Reduces garbage collection overhead and improves performance.

Spatial Grid Optimization:

File: useForceGraph.ts

Task: Refactor the spatial grid to be used solely for frustum culling. Only update the grid when nodes move significantly. Remove the logic that only renders the closest node in a cell. Combine frustum culling with spatial grid culling for maximum efficiency.

Reason: Improves culling efficiency and prevents nodes from disappearing unnecessarily.

Priority 3: Improve Data Handling and Synchronization

Binary Update Validation and Error Handling:

File: stores/binaryUpdate.ts, services/websocketService.ts

Task: Add robust validation to the binary update handling in both the store and the websocket service. Check for buffer size mismatches, invalid position/velocity values, and other potential errors. Implement appropriate error handling and logging.

Reason: Prevents crashes and ensures data integrity.

Client-Server Synchronization for Interactions:

File: components/visualization/GraphSystem.vue, services/websocketService.ts

Task: If you implement client-side interactions (e.g., dragging nodes), ensure that these interactions are correctly synchronized with the server. The client should send position updates for interacted nodes to the server, and the server should acknowledge these updates. Consider a locking mechanism or other synchronization strategy to prevent conflicts between client and server updates.

Reason: Prevents erratic graph behavior and ensures consistency.

Priority 4: Enhance Debug Logging

More Explicit Debug Logging:

Files: Throughout the client-side codebase, especially in performance-critical sections.

Task: Add more specific and targeted console.debug statements to track the execution flow and data transformations. Include timestamps and relevant context information (e.g., node IDs, positions, velocities). Focus on logging key events, data sizes, processing times, and potential errors.

Reason: Makes debugging easier and helps identify the root cause of performance issues. The existing logs are not granular enough to pinpoint the problem areas.

Example of Optimized updateNodes in useForceGraph.ts:

const updateNodes = (camera: Camera) => {
  const res = resources.value;
  if (!res) return;

  const changedNodes = binaryUpdateStore.getChangedNodes;

  // Only update changed nodes
  changedNodes.forEach(index => {
    const node = nodes.value[index];
    if (!node || !node.visible) return;

    // ... (get pooled objects: matrix, position, scale)

    // Update instance matrix based on server-provided positions
    position.set(
      binaryUpdateStore.positions[index * 3],
      binaryUpdateStore.positions[index * 3 + 1],
      binaryUpdateStore.positions[index * 3 + 2]
    );

    // ... (set matrix, color, scale on InstancedMesh)
  });

  // ... (update instance counts and needsUpdate flags)
};
Use code with caution.
TypeScript
This revised approach focuses on efficiency and leverages the server-side force calculations. The client primarily receives and renders data, significantly reducing the computational load and improving performance. The enhanced logging will provide more detailed information for debugging. Remember to test thoroughly after each change to ensure correctness and performance improvements.