// public/js/services/graphDataManager.js

/**
 * GraphDataManager handles the management and updating of graph data received from the server.
 */
export class GraphDataManager {
    websocketService = null;
    graphData = null;
    forceDirectedParams = {
        iterations: 250,
        spring_strength: 0.1,
        repulsion_strength: 1000,
        attraction_strength: 0.01,
        damping: 0.8,
        time_step: 0.5  // Add default time_step
    };
    pendingRecalculation = false;
    initialLayoutDone = false;

    /**
     * Creates a new GraphDataManager instance.
     * @param {WebsocketService} websocketService - The WebSocket service instance.
     */
    constructor(websocketService) {
        this.websocketService = websocketService;
        console.log('GraphDataManager initialized');
        
        // Use arrow functions for event handlers to preserve this context
        this.websocketService.on('graphUpdate', this.handleGraphUpdate);
        this.websocketService.on('gpuPositions', this.handleGPUPositions);
    }

    requestInitialData = () => {
        console.log('Requesting initial data');
        this.websocketService.send({ type: 'getInitialData' });
    }

    handleGPUPositions = (update) => {
        if (!this.graphData || !this.graphData.nodes) {
            console.error('Cannot apply GPU position update: No graph data exists');
            return;
        }

        const { positions } = update;
        console.log('Received GPU position update:', positions);
        
        // Transform position array into node objects
        const updatedNodes = this.graphData.nodes.map((node, index) => {
            if (positions[index]) {
                const pos = positions[index];
                if (Array.isArray(pos) && pos.length >= 6) {
                    return {
                        ...node,
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                        vx: pos[3],
                        vy: pos[4],
                        vz: pos[5]
                    };
                }
            }
            return node;
        });

        // Update the graph data with the new nodes
        this.graphData = {
            ...this.graphData,
            nodes: updatedNodes
        };

        // Notify visualization of position updates with structured data
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { 
            detail: {
                nodes: this.graphData.nodes,
                edges: this.graphData.edges,
                metadata: this.graphData.metadata
            }
        }));
    }

    handleGraphUpdate = (data) => {
        console.log('Received graph update:', data);
        if (!data || !data.graphData) {
            console.error('Invalid graph update data received:', data);
            return;
        }
        this.updateGraphData(data.graphData);
    }

    updateGraphData = (newData) => {
        console.log('Updating graph data with:', newData);
        
        if (!newData) {
            console.error('Received null or undefined graph data');
            return;
        }

        // Preserve metadata if it exists in newData
        const metadata = newData.metadata || {};
        console.log('Received metadata:', metadata);

        // Handle the case where newData already has nodes and edges arrays
        if (Array.isArray(newData.nodes) && Array.isArray(newData.edges)) {
            // Integrate new positions with existing velocities and metadata
            const nodes = newData.nodes.map(node => {
                const existingNode = this.graphData?.nodes?.find(n => n.id === node.id);
                const nodeMetadata = metadata[`${node.id}.md`] || {};
                
                // Keep existing velocities if available, otherwise initialize to 0
                const vx = existingNode?.vx || 0;
                const vy = existingNode?.vy || 0;
                const vz = existingNode?.vz || 0;

                // Use new position if valid, otherwise keep existing or initialize to 0
                const x = (typeof node.x === 'number' && !isNaN(node.x)) ? node.x : 
                         (existingNode?.x || 0);
                const y = (typeof node.y === 'number' && !isNaN(node.y)) ? node.y :
                         (existingNode?.y || 0);
                const z = (typeof node.z === 'number' && !isNaN(node.z)) ? node.z :
                         (existingNode?.z || 0);

                return {
                    ...node,
                    x, y, z,
                    vx, vy, vz,
                    metadata: nodeMetadata
                };
            });

            this.graphData = {
                nodes,
                edges: newData.edges,
                metadata
            };
        }
        // Handle the case where we need to construct nodes from edges
        else if (Array.isArray(newData.edges)) {
            const nodeSet = new Set();
            newData.edges.forEach(edge => {
                nodeSet.add(edge.source);
                nodeSet.add(edge.target);
            });

            const nodes = Array.from(nodeSet).map(id => {
                const existingNode = this.graphData?.nodes?.find(n => n.id === id);
                const nodeMetadata = metadata[`${id}.md`] || {};
                
                return {
                    id,
                    label: id,
                    // Preserve existing position and velocity if available
                    x: existingNode?.x || 0,
                    y: existingNode?.y || 0,
                    z: existingNode?.z || 0,
                    vx: existingNode?.vx || 0,
                    vy: existingNode?.vy || 0,
                    vz: existingNode?.vz || 0,
                    metadata: nodeMetadata
                };
            });

            this.graphData = {
                nodes,
                edges: newData.edges.map(e => ({
                    source: e.source,
                    target: e.target,
                    weight: e.weight,
                    hyperlinks: e.hyperlinks
                })),
                metadata
            };
        } else {
            console.error('Received invalid graph data:', newData);
            return;
        }

        console.log(`Graph data updated: ${this.graphData.nodes.length} nodes, ${this.graphData.edges.length} edges`);
        console.log('Metadata entries:', Object.keys(this.graphData.metadata).length);
        
        // Dispatch an event to notify that the graph data has been updated with structured data
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { 
            detail: {
                nodes: this.graphData.nodes,
                edges: this.graphData.edges,
                metadata: this.graphData.metadata
            }
        }));

        // If there was a pending recalculation, do it now
        if (this.pendingRecalculation) {
            console.log('Processing pending layout recalculation');
            this.pendingRecalculation = false;
            this.recalculateLayout();
        }

        // If this is the first time we've received graph data, mark it as initial layout
        if (!this.initialLayoutDone) {
            console.log('Performing initial layout calculation');
            this.initialLayoutDone = true;
            this.recalculateLayout(true);
        }
    }

    getGraphData = () => {
        if (this.graphData) {
            console.log(`Returning graph data: ${this.graphData.nodes.length} nodes, ${this.graphData.edges.length} edges`);
            console.log('Metadata entries:', Object.keys(this.graphData.metadata).length);
        } else {
            console.warn('Graph data is null');
        }
        return {
            nodes: this.graphData?.nodes || [],
            edges: this.graphData?.edges || [],
            metadata: this.graphData?.metadata || {}
        };
    }

    isGraphDataValid = () => {
        return this.graphData && 
               Array.isArray(this.graphData.nodes) && 
               Array.isArray(this.graphData.edges) &&
               this.graphData.nodes.length > 0;
    }

    updateForceDirectedParams = (name, value) => {
        console.log(`Updating force-directed parameter: ${name} = ${value}`);
        
        // Convert from forceDirected prefixed names to server parameter names
        const paramMap = {
            'forceDirectedIterations': 'iterations',
            'forceDirectedSpring': 'spring_strength',
            'forceDirectedRepulsion': 'repulsion_strength',
            'forceDirectedAttraction': 'attraction_strength',
            'forceDirectedDamping': 'damping'
        };

        const serverParamName = paramMap[name] || name;
        if (this.forceDirectedParams.hasOwnProperty(serverParamName)) {
            this.forceDirectedParams[serverParamName] = value;
            console.log('Force-directed parameters updated:', this.forceDirectedParams);
            
            // Only recalculate if we have valid graph data, otherwise mark as pending
            if (this.isGraphDataValid()) {
                this.recalculateLayout();
            } else {
                console.log('Marking layout recalculation as pending until graph data is available');
                this.pendingRecalculation = true;
            }
        } else {
            console.warn(`Unknown force-directed parameter: ${name}`);
        }
    }

    recalculateLayout = (isInitial = false) => {
        console.log('Requesting server layout recalculation with parameters:', this.forceDirectedParams);
        if (this.isGraphDataValid()) {
            // Create binary data with multiplexed header
            const buffer = new ArrayBuffer(this.graphData.nodes.length * 24 + 4);
            const view = new Float32Array(buffer);
            
            // Pack is_initial_layout and time_step into a single float32:
            // Integer part (0 or 1) = is_initial_layout
            // Decimal part = time_step
            // Example: 0.5 = not initial layout (0) with time_step of 0.5
            //         1.5 = is initial layout (1) with time_step of 0.5
            view[0] = isInitial ? (1 + this.forceDirectedParams.time_step) : this.forceDirectedParams.time_step;

            // Fill position data
            this.graphData.nodes.forEach((node, index) => {
                const offset = index * 6 + 1; // +1 to skip the header
                // Position
                view[offset] = node.x;
                view[offset + 1] = node.y;
                view[offset + 2] = node.z;
                // Velocity
                view[offset + 3] = node.vx || 0;
                view[offset + 4] = node.vy || 0;
                view[offset + 5] = node.vz || 0;
            });

            // Send binary data directly
            this.websocketService.send(buffer);
            
            window.dispatchEvent(new CustomEvent('layoutRecalculationRequested', {
                detail: this.forceDirectedParams
            }));
        } else {
            console.error('Cannot recalculate layout: Invalid graph data');
            this.pendingRecalculation = true;
        }
    }
}
