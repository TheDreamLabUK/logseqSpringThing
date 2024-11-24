// public/js/services/graphDataManager.js

/**
 * GraphDataManager handles the management and updating of graph data received from the server.
 */
export class GraphDataManager {
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.graphData = null;
        
        console.log('GraphDataManager initialized');
        
        // Bind methods to preserve this context
        this.handleGraphUpdate = this.handleGraphUpdate.bind(this);
        this.handleBinaryPositionUpdate = this.handleBinaryPositionUpdate.bind(this);
        
        // Set up event listeners
        if (this.websocketService) {
            this.websocketService.on('graphUpdate', this.handleGraphUpdate);
            this.websocketService.on('gpuPositions', this.handleBinaryPositionUpdate);

            // Debug listener for websocket connection state
            this.websocketService.on('connect', () => {
                console.log('GraphDataManager detected websocket connection');
                this.requestInitialData();
            });
        } else {
            console.error('GraphDataManager initialized without websocket service');
        }
    }

    requestInitialData() {
        console.log('Requesting initial data');
        if (this.websocketService) {
            this.websocketService.send({ type: 'getInitialData' });
        }
    }

    handleBinaryPositionUpdate(update) {
        if (!this.graphData || !this.graphData.nodes) {
            console.error('Cannot apply position update: No graph data exists');
            return;
        }

        const { positions } = update;
        console.log('Received position update for', positions.length, 'nodes');
        
        // Transform position array into node objects
        const updatedNodes = this.graphData.nodes.map((node, index) => {
            if (positions[index]) {
                const pos = positions[index];
                return {
                    ...node,
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    vx: pos.vx,
                    vy: pos.vy,
                    vz: pos.vz
                };
            }
            return node;
        });

        // Update the graph data with the new nodes
        this.graphData = {
            ...this.graphData,
            nodes: updatedNodes
        };

        // Notify visualization of position updates
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { 
            detail: {
                nodes: this.graphData.nodes,
                edges: this.graphData.edges,
                metadata: this.graphData.metadata
            }
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
                
                return {
                    ...node,
                    x: (typeof node.x === 'number' && !isNaN(node.x)) ? node.x : (existingNode?.x || 0),
                    y: (typeof node.y === 'number' && !isNaN(node.y)) ? node.y : (existingNode?.y || 0),
                    z: (typeof node.z === 'number' && !isNaN(node.z)) ? node.z : (existingNode?.z || 0),
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
                    x: existingNode?.x || 0,
                    y: existingNode?.y || 0,
                    z: existingNode?.z || 0,
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
        
        // Dispatch update event
        window.dispatchEvent(new CustomEvent('graphDataUpdated', { 
            detail: {
                nodes: this.graphData.nodes,
                edges: this.graphData.edges,
                metadata: this.graphData.metadata
            }
        }));
    }

    getGraphData() {
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

    isGraphDataValid() {
        return this.graphData && 
               Array.isArray(this.graphData.nodes) && 
               Array.isArray(this.graphData.edges) &&
               this.graphData.nodes.length > 0;
    }
}
