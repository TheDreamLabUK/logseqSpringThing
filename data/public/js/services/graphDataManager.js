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
        this.dispatchGraphUpdate();
    }

    handleGraphUpdate(data) {
        console.log('Received graph update:', data);
        if (!data || !data.graphData) {
            console.error('Invalid graph update data received:', data);
            return;
        }
        this.updateGraphData(data.graphData);
    }

    // Helper function to generate initial positions in a sphere
    generateInitialPositions(count) {
        const positions = [];
        const radius = 20; // Initial sphere radius
        const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle

        for (let i = 0; i < count; i++) {
            const y = 1 - (i / (count - 1)) * 2; // y goes from 1 to -1
            const radius_at_y = Math.sqrt(1 - y * y); // radius at y
            const theta = phi * i; // Golden angle increment

            const x = Math.cos(theta) * radius_at_y;
            const z = Math.sin(theta) * radius_at_y;

            positions.push({
                x: x * radius,
                y: y * radius,
                z: z * radius
            });
        }

        return positions;
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

        let nodes = [];
        let edges = [];

        // Handle the case where newData already has nodes and edges arrays
        if (Array.isArray(newData.nodes) && Array.isArray(newData.edges)) {
            console.log('Processing complete graph data with nodes and edges');
            
            // Process nodes with existing positions
            nodes = newData.nodes.map(node => {
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

            edges = newData.edges;
        }
        // Handle the case where we need to construct nodes from edges
        else if (Array.isArray(newData.edges)) {
            console.log('Constructing nodes from edges');
            
            // Extract unique node IDs from edges
            const nodeSet = new Set();
            newData.edges.forEach(edge => {
                nodeSet.add(edge.source);
                nodeSet.add(edge.target);
            });

            // Generate initial positions for nodes
            const initialPositions = this.generateInitialPositions(nodeSet.size);
            
            // Create nodes with positions
            nodes = Array.from(nodeSet).map((id, index) => {
                const existingNode = this.graphData?.nodes?.find(n => n.id === id);
                const nodeMetadata = metadata[`${id}.md`] || {};
                const position = existingNode || initialPositions[index];
                
                return {
                    id,
                    label: id,
                    x: position.x,
                    y: position.y,
                    z: position.z,
                    metadata: nodeMetadata
                };
            });

            edges = newData.edges;
        } else {
            console.error('Invalid graph data format:', newData);
            return;
        }

        // Process edges
        const processedEdges = edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            weight: edge.weight || 1,
            hyperlinks: edge.hyperlinks || []
        }));

        // Update graph data
        this.graphData = {
            nodes,
            edges: processedEdges,
            metadata
        };

        console.log(`Graph data updated: ${nodes.length} nodes, ${processedEdges.length} edges`);
        
        // Log sample of node positions
        console.log('Node positions sample:', 
            nodes.slice(0, 3).map(n => 
                `Node ${n.id}: (${n.x.toFixed(2)}, ${n.y.toFixed(2)}, ${n.z.toFixed(2)})`
            )
        );

        // Dispatch update event
        this.dispatchGraphUpdate();
    }

    dispatchGraphUpdate() {
        if (!this.graphData) return;

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
