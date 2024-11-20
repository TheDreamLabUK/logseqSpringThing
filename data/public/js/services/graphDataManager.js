// public/js/services/graphDataManager.js

/**
 * GraphDataManager handles the management and updating of graph data received from the server.
 */
export class GraphDataManager {
    /**
     * Creates a new GraphDataManager instance.
     * @param {WebsocketService} websocketService - The WebSocket service instance.
     */
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.graphData = null;
        this.forceDirectedParams = {
            forceDirectedIterations: 250,
            forceDirectedSpring: 0.1,
            forceDirectedRepulsion: 1000,
            forceDirectedAttraction: 0.01,
            forceDirectedDamping: 0.8
        };
        console.log('GraphDataManager initialized');
        
        this.websocketService.on('graphUpdate', this.handleGraphUpdate.bind(this));
        this.websocketService.on('gpuPositions', this.handleGPUPositions.bind(this));
    }

    requestInitialData() {
        console.log('Requesting initial data');
        this.websocketService.send({ type: 'getInitialData' });
    }

    handleGPUPositions(update) {
        if (!this.graphData || !this.graphData.nodes) {
            console.error('Cannot apply GPU position update: No graph data exists');
            return;
        }

        const { positions } = update;
        console.log('Received GPU position update:', positions);
        
        // Update node positions from GPU computation
        this.graphData.nodes.forEach((node, index) => {
            if (positions[index]) {
                const pos = positions[index];
                if (Array.isArray(pos) && pos.length >= 3) {
                    node.x = pos[0];
                    node.y = pos[1];
                    node.z = pos[2];
                    // Clear velocities since GPU is handling movement
                    node.vx = 0;
                    node.vy = 0;
                    node.vz = 0;
                } else if (pos && typeof pos === 'object') {
                    node.x = pos.x;
                    node.y = pos.y;
                    node.z = pos.z;
                    node.vx = 0;
                    node.vy = 0;
                    node.vz = 0;
                }
            }
        });

        // Notify visualization of position updates
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { 
            detail: this.graphData 
        }));
    }

    handleGraphUpdate(data) {
        console.log('Received graph update:', data);
        if (!data || !data.graphData) {
            console.error('Invalid graph update data received:', data);
            return;
        }
        this.updateGraphData(data.graphData);
    }

    updateGraphData(newData) {
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
        
        // Dispatch an event to notify that the graph data has been updated
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { detail: this.graphData }));
    }

    getGraphData() {
        if (this.graphData) {
            console.log(`Returning graph data: ${this.graphData.nodes.length} nodes, ${this.graphData.edges.length} edges`);
            console.log('Metadata entries:', Object.keys(this.graphData.metadata).length);
        } else {
            console.warn('Graph data is null');
        }
        return this.graphData;
    }

    isGraphDataValid() {
        return this.graphData && 
               Array.isArray(this.graphData.nodes) && 
               Array.isArray(this.graphData.edges) &&
               this.graphData.nodes.length > 0;
    }

    updateForceDirectedParams(name, value) {
        console.log(`Updating force-directed parameter: ${name} = ${value}`);
        
        // Handle full parameter names
        if (this.forceDirectedParams.hasOwnProperty(name)) {
            this.forceDirectedParams[name] = value;
            console.log('Force-directed parameters updated:', this.forceDirectedParams);
            this.recalculateLayout();
        } else {
            console.warn(`Unknown force-directed parameter: ${name}`);
        }
    }

    recalculateLayout() {
        console.log('Requesting server layout recalculation with parameters:', this.forceDirectedParams);
        if (this.isGraphDataValid()) {
            this.websocketService.send({
                type: 'recalculateLayout',
                params: {
                    iterations: this.forceDirectedParams.forceDirectedIterations,
                    spring_strength: this.forceDirectedParams.forceDirectedSpring,
                    repulsion_strength: this.forceDirectedParams.forceDirectedRepulsion,
                    attraction_strength: this.forceDirectedParams.forceDirectedAttraction,
                    damping: this.forceDirectedParams.forceDirectedDamping
                }
            });
            
            window.dispatchEvent(new CustomEvent('layoutRecalculationRequested', {
                detail: this.forceDirectedParams
            }));
        } else {
            console.error('Cannot recalculate layout: Invalid graph data');
        }
    }
}
