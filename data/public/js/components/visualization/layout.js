export class LayoutManager {
    constructor(settings = {}) {
        // Configuration
        this.initialIterations = settings.forceDirectedIterations || 250;
        this.updateIterations = 1;       // Single iteration for smooth continuous updates
        this.targetRadius = 200;
        this.naturalLength = 100;
        this.attraction = settings.forceDirectedAttraction || 0.01;
        this.repulsion = settings.forceDirectedRepulsion || 1000;
        
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
    }

    initializePositions(nodes) {
        console.log('Initializing positions for nodes:', nodes);
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
        console.log('Position initialization complete');
    }

    applyForceDirectedLayout(nodes, edges) {
        if (!this.isInitialized) {
            console.warn('Layout manager not initialized');
            return;
        }

        console.log('Applying force-directed layout');
        const damping = 0.9;
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
                const force = (distance - this.naturalLength) * this.attraction * (edge.weight || 1);
                
                const fx = (dx / distance) * force;
                const fy = (dy / distance) * force;
                const fz = (dz / distance) * force;

                // Apply forces to both nodes
                sourceNode.vx += fx;
                sourceNode.vy += fy;
                sourceNode.vz += fz;
                targetNode.vx -= fx;
                targetNode.vy -= fy;
                targetNode.vz -= fz;
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
            node.vx *= damping;
            node.vy *= damping;
            node.vz *= damping;

            // Bound checking
            const bound = 500;
            if (Math.abs(node.x) > bound) node.vx *= -0.5;
            if (Math.abs(node.y) > bound) node.vy *= -0.5;
            if (Math.abs(node.z) > bound) node.vz *= -0.5;
        });
    }

    updateFeature(control, value) {
        console.log(`Updating layout feature: ${control} = ${value}`);
        switch(control) {
            case 'forceDirectedIterations':
                this.initialIterations = value;
                break;
            case 'forceDirectedRepulsion':
                this.repulsion = value;
                break;
            case 'forceDirectedAttraction':
                this.attraction = value;
                break;
        }
    }

    performLayout(graphData) {
        if (!this.isInitialized || !graphData) {
            console.warn('Cannot perform layout: not initialized or no graph data');
            return;
        }

        const now = Date.now();
        if (now - this.lastUpdateTime >= this.updateInterval) {
            // Apply force-directed layout
            this.applyForceDirectedLayout(graphData.nodes, graphData.edges);
            
            // Send position updates
            this.sendPositionUpdates(graphData.nodes);
            this.lastUpdateTime = now;
        }
    }

    sendPositionUpdates(nodes) {
        if (!this.lastPositions) return;

        // Create binary buffer for all node positions and velocities (24 bytes per node)
        const buffer = new ArrayBuffer(nodes.length * 24);
        const dataView = new DataView(buffer);
        let hasChanges = false;

        nodes.forEach((node, index) => {
            const offset = index * 24;
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
                dataView.setFloat32(offset, node.x, true);
                dataView.setFloat32(offset + 4, node.y, true);
                dataView.setFloat32(offset + 8, node.z, true);

                // Velocity (vec3<f32>)
                dataView.setFloat32(offset + 12, node.vx || 0, true);
                dataView.setFloat32(offset + 16, node.vy || 0, true);
                dataView.setFloat32(offset + 20, node.vz || 0, true);
            }
        });

        if (hasChanges) {
            // Dispatch binary data event
            window.dispatchEvent(new CustomEvent('positionUpdate', {
                detail: buffer
            }));
        }
    }

    applyPositionUpdates(positions) {
        if (!this.lastPositions) return;

        // Handle binary data format (24 bytes per node)
        if (positions instanceof ArrayBuffer) {
            const dataView = new DataView(positions);
            for (let i = 0; i < this.lastPositions.length; i++) {
                const offset = i * 24;
                this.lastPositions[i] = {
                    x: dataView.getFloat32(offset, true),
                    y: dataView.getFloat32(offset + 4, true),
                    z: dataView.getFloat32(offset + 8, true),
                    vx: dataView.getFloat32(offset + 12, true),
                    vy: dataView.getFloat32(offset + 16, true),
                    vz: dataView.getFloat32(offset + 20, true)
                };
            }
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
