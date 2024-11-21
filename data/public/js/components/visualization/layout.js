export class LayoutManager {
    constructor(settings = {}) {
        // Configuration
        this.initialIterations = settings.iterations || 250;
        this.updateIterations = 1;       // Single iteration for smooth continuous updates
        this.targetRadius = 200;
        this.naturalLength = 100;
        this.attraction = settings.attraction_strength || 0.01;
        this.repulsion = settings.repulsion_strength || 1000;
        this.spring = settings.spring_strength || 0.1;
        this.damping = settings.damping || 0.8;
        
        // State
        this.isInitialized = false;
        this.isSimulating = false;
        this.animationFrameId = null;
        this.lastPositions = null;       // Store previous positions for change detection
        this.updateThreshold = 0.001;    // Minimum position change to trigger update
        this.lastUpdateTime = 0;         // Last time positions were sent to server
        this.updateInterval = 16.67;     // Exactly 60fps
        this.positionBuffer = null;
        this.edges = [];                 // Store computed edges
        this.nodeCount = 0;              // Track number of nodes
        this.waitingForInitialData = true; // Wait for initial data before sending updates
    }

    initializePositions(nodes) {
        console.log('Initializing positions for nodes:', nodes);
        this.nodeCount = nodes.length;
        nodes.forEach(node => {
            // Initialize only if positions are invalid
            if (isNaN(node.x) || isNaN(node.y) || isNaN(node.z)) {
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = this.targetRadius * Math.cbrt(Math.random());
                
                node.x = r * Math.sin(phi) * Math.cos(theta);
                node.y = r * Math.sin(phi) * Math.sin(theta);
                node.z = r * Math.cos(phi);
            }
            // Always ensure velocities are initialized
            if (!node.vx) node.vx = 0;
            if (!node.vy) node.vy = 0;
            if (!node.vz) node.vz = 0;
        });

        // Initialize last positions with velocities
        this.lastPositions = nodes.map(node => ({
            x: node.x,
            y: node.y,
            z: node.z,
            vx: node.vx,
            vy: node.vy,
            vz: node.vz
        }));

        this.isInitialized = true;
        this.waitingForInitialData = false; // Initial data received
        console.log('Position initialization complete');

        // Send initial positions to server
        this.sendPositionUpdates(nodes, true);
    }

    applyForceDirectedLayout(nodes, edges) {
        if (!nodes || !Array.isArray(nodes) || nodes.length === 0) {
            console.warn('Invalid nodes array provided to force-directed layout');
            return;
        }

        if (!this.isInitialized || this.waitingForInitialData) {
            console.warn('Layout manager not initialized or waiting for initial data');
            return;
        }

        console.log('Applying force-directed layout to', nodes.length, 'nodes');
        const dt = 0.1;

        // Apply forces based on edges (topic counts)
        edges.forEach(edge => {
            const sourceNode = nodes.find(n => n.id === edge.source);
            const targetNode = nodes.find(n => n.id === edge.target);
            
            if (sourceNode && targetNode) {
                // Calculate spring force based on topic counts
                const dx = targetNode.x - sourceNode.x;
                const dy = targetNode.y - sourceNode.y;
                const dz = targetNode.z - sourceNode.z;
                
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (distance === 0) return;

                // Use edge weight (from topic counts) to scale the force
                const force = (distance - this.naturalLength) * this.spring * (edge.weight || 1);
                
                const fx = (dx / distance) * force;
                const fy = (dy / distance) * force;
                const fz = (dz / distance) * force;

                // Apply forces to both nodes
                sourceNode.vx += fx * this.attraction;
                sourceNode.vy += fy * this.attraction;
                sourceNode.vz += fz * this.attraction;
                targetNode.vx -= fx * this.attraction;
                targetNode.vy -= fy * this.attraction;
                targetNode.vz -= fz * this.attraction;
            }
        });

        // Apply repulsion between all nodes
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[j].x - nodes[i].x;
                const dy = nodes[j].y - nodes[i].y;
                const dz = nodes[j].z - nodes[i].z;
                
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (distance === 0) continue;

                const force = this.repulsion / (distance * distance);
                
                const fx = (dx / distance) * force;
                const fy = (dy / distance) * force;
                const fz = (dz / distance) * force;

                nodes[i].vx -= fx;
                nodes[i].vy -= fy;
                nodes[i].vz -= fz;
                nodes[j].vx += fx;
                nodes[j].vy += fy;
                nodes[j].vz += fz;
            }
        }

        // Update positions and apply damping
        nodes.forEach(node => {
            // Apply current velocity
            node.x += node.vx * dt;
            node.y += node.vy * dt;
            node.z += node.vz * dt;

            // Apply damping
            node.vx *= this.damping;
            node.vy *= this.damping;
            node.vz *= this.damping;

            // Bound checking
            const bound = 500;
            if (Math.abs(node.x) > bound) node.vx *= -0.5;
            if (Math.abs(node.y) > bound) node.vy *= -0.5;
            if (Math.abs(node.z) > bound) node.vz *= -0.5;
        });
    }

    updateFeature(control, value) {
        console.log(`Updating layout feature: ${control} = ${value}`);
        
        // Convert from forceDirected prefixed names to internal parameter names
        const paramMap = {
            'forceDirectedIterations': 'iterations',
            'forceDirectedSpring': 'spring_strength',
            'forceDirectedRepulsion': 'repulsion_strength',
            'forceDirectedAttraction': 'attraction_strength',
            'forceDirectedDamping': 'damping'
        };

        const paramName = paramMap[control] || control;
        switch(paramName) {
            case 'iterations':
                this.initialIterations = value;
                break;
            case 'spring_strength':
                this.spring = value;
                break;
            case 'repulsion_strength':
                this.repulsion = value;
                break;
            case 'attraction_strength':
                this.attraction = value;
                break;
            case 'damping':
                this.damping = value;
                break;
            default:
                console.warn(`Unknown layout parameter: ${control}`);
        }
    }

    performLayout(graphData) {
        if (!this.isInitialized || !graphData || this.waitingForInitialData) {
            console.warn('Cannot perform layout: not initialized, no graph data, or waiting for initial data');
            return;
        }

        const now = Date.now();
        if (now - this.lastUpdateTime >= this.updateInterval) {
            // Apply force-directed layout
            this.applyForceDirectedLayout(graphData.nodes, graphData.edges);
            
            // Send position updates
            this.sendPositionUpdates(graphData.nodes, false);
            this.lastUpdateTime = now;
        }
    }

    sendPositionUpdates(nodes, isInitialLayout = false) {
        if (!this.lastPositions || !this.isInitialized || nodes.length !== this.nodeCount || this.waitingForInitialData) {
            console.warn('Cannot send position updates: not initialized, node count mismatch, or waiting for initial data');
            return;
        }

        // Create binary buffer for all node positions and velocities (24 bytes per node)
        const buffer = new ArrayBuffer(nodes.length * 24 + 4); // Extra 4 bytes for is_initial_layout flag
        const dataView = new Float32Array(buffer);
        let hasChanges = false;

        // Set is_initial_layout flag (1.0 for true, 0.0 for false)
        dataView[0] = isInitialLayout ? 1.0 : 0.0;

        nodes.forEach((node, index) => {
            const offset = index * 6 + 1; // +1 to account for is_initial_layout flag
            const lastPos = this.lastPositions[index];

            if (!lastPos || 
                Math.abs(node.x - lastPos.x) > this.updateThreshold ||
                Math.abs(node.y - lastPos.y) > this.updateThreshold ||
                Math.abs(node.z - lastPos.z) > this.updateThreshold ||
                Math.abs(node.vx - lastPos.vx) > this.updateThreshold ||
                Math.abs(node.vy - lastPos.vy) > this.updateThreshold ||
                Math.abs(node.vz - lastPos.vz) > this.updateThreshold) {
                
                hasChanges = true;
                
                // Update last position and velocity
                if (lastPos) {
                    lastPos.x = node.x;
                    lastPos.y = node.y;
                    lastPos.z = node.z;
                    lastPos.vx = node.vx;
                    lastPos.vy = node.vy;
                    lastPos.vz = node.vz;
                }

                // Position (vec3<f32>)
                dataView[offset] = node.x;
                dataView[offset + 1] = node.y;
                dataView[offset + 2] = node.z;

                // Velocity (vec3<f32>)
                dataView[offset + 3] = node.vx || 0;
                dataView[offset + 4] = node.vy || 0;
                dataView[offset + 5] = node.vz || 0;
            }
        });

        if (hasChanges || isInitialLayout) {
            // Log the buffer size before sending
            console.log(`Sending position update buffer of size: ${buffer.byteLength} bytes for ${nodes.length} nodes (isInitialLayout: ${isInitialLayout})`);
            
            // Dispatch binary data event
            window.dispatchEvent(new CustomEvent('positionUpdate', {
                detail: buffer
            }));
        }
    }

    startContinuousSimulation(graphData) {
        if (this.isSimulating) return;
        
        console.log('Starting continuous simulation');
        this.isSimulating = true;
        const animate = () => {
            if (!this.isSimulating) return;
            
            // Send position updates at regular intervals
            this.performLayout(graphData);
            this.animationFrameId = requestAnimationFrame(animate);
        };
        
        animate();
    }

    stopSimulation() {
        console.log('Stopping simulation');
        this.isSimulating = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }
}
