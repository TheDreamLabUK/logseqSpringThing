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
            iterations: 100,
            repulsionStrength: 1.0,
            attractionStrength: 0.01,
            damping: 0.9
        };
        console.log('GraphDataManager initialized');
        
        // Set up WebSocket message listeners
        this.websocketService.on('graphUpdate', this.handleGraphUpdate.bind(this));
        this.websocketService.on('gpuPositions', this.handleGPUPositions.bind(this));
    }

    /**
     * Requests the initial graph data from the server via WebSocket.
     */
    requestInitialData() {
        console.log('Requesting initial data');
        this.websocketService.send({ type: 'getInitialData' });
    }

    /**
     * Handles GPU-computed position updates from the server.
     * @param {object} update - The position update data.
     */
    handleGPUPositions(update) {
        if (!this.graphData || !this.graphData.nodes) {
            console.error('Cannot apply GPU position update: No graph data exists');
            return;
        }

        const { positions } = update;
        
        // Update node positions from GPU computation
        this.graphData.nodes.forEach((node, index) => {
            if (positions[index]) {
                const [x, y, z] = positions[index];
                node.x = x;
                node.y = y;
                node.z = z;
                // Clear velocities since GPU is handling movement
                node.vx = 0;
                node.vy = 0;
                node.vz = 0;
            }
        });

        // Notify visualization of position updates
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { 
            detail: this.graphData 
        }));
    }

    /**
     * Handles graph update messages.
     * @param {object} data - The received graph data.
     */
    handleGraphUpdate(data) {
        console.log('Received graph update:', data);
        if (!data || !data.graphData) {
            console.error('Invalid graph update data received:', data);
            return;
        }
        this.updateGraphData(data.graphData);
    }

    /**
     * Updates the internal graph data with new data received from the server.
     * @param {object} newData - The new graph data.
     */
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
                nodeSet.add(edge.target); // Fixed: using target instead of target_node
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
                    target: e.target, // Fixed: using target instead of target_node
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

    /**
     * Retrieves the current graph data.
     * @returns {object|null} The current graph data or null if not set.
     */
    getGraphData() {
        if (this.graphData) {
            console.log(`Returning graph data: ${this.graphData.nodes.length} nodes, ${this.graphData.edges.length} edges`);
            console.log('Metadata entries:', Object.keys(this.graphData.metadata).length);
        } else {
            console.warn('Graph data is null');
        }
        return this.graphData;
    }

    /**
     * Checks if the graph data is valid.
     * @returns {boolean} True if the graph data is valid, false otherwise.
     */
    isGraphDataValid() {
        return this.graphData && 
               Array.isArray(this.graphData.nodes) && 
               Array.isArray(this.graphData.edges) &&
               this.graphData.nodes.length > 0;
    }

    /**
     * Updates the force-directed graph parameters.
     * @param {string} name - The name of the parameter to update.
     * @param {number} value - The new value for the parameter.
     */
    updateForceDirectedParams(name, value) {
        console.log(`Updating force-directed parameter: ${name} = ${value}`);
        const paramMap = {
            'iterations': 'iterations',
            'repulsionStrength': 'repulsionStrength',
            'attractionStrength': 'attractionStrength'
        };

        const serverParamName = paramMap[name];
        if (serverParamName) {
            this.forceDirectedParams[serverParamName] = value;
            console.log('Force-directed parameters updated:', this.forceDirectedParams);
            this.recalculateLayout();
        } else {
            console.warn(`Unknown force-directed parameter: ${name}`);
        }
    }

    /**
     * Recalculates the graph layout using the current force-directed parameters.
     */
    recalculateLayout() {
        console.log('Requesting server layout recalculation with parameters:', this.forceDirectedParams);
        if (this.isGraphDataValid()) {
            this.websocketService.send({
                type: 'recalculateLayout',
                params: {
                    iterations: this.forceDirectedParams.iterations,
                    springStrength: this.forceDirectedParams.attractionStrength,
                    repulsionStrength: this.forceDirectedParams.repulsionStrength,
                    damping: this.forceDirectedParams.damping
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
