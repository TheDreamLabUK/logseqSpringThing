export class LayoutManager {
    constructor() {
        // Configuration
        this.initialIterations = 250;    // High iteration count for initial layout
        this.updateIterations = 1;       // Single iteration for smooth continuous updates
        this.forceDirectedSpring = 2.0;
        this.forceDirectedDamping = 0.8;
        this.targetRadius = 200;
        this.naturalLength = 100;
        
        // State
        this.isInitialized = false;
        this.isSimulating = false;
        this.animationFrameId = null;
        this.lastPositions = null;       // Store previous positions for change detection
        this.updateThreshold = 0.001;    // Minimum position change to trigger update
        this.lastUpdateTime = 0;         // Last time positions were sent to server
        this.updateInterval = 50;        // Minimum ms between position updates
    }

    initializePositions(nodes) {
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

        // Initialize last positions
        this.lastPositions = nodes.map(node => ({
            x: node.x,
            y: node.y,
            z: node.z
        }));
    }

    calculateSpringForce(pos1, pos2, mass1, mass2, isConnected) {
        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const dz = pos2.z - pos1.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;

        let forceMagnitude;
        if (isConnected) {
            forceMagnitude = this.forceDirectedSpring * (distance - this.naturalLength);
        } else {
            forceMagnitude = -this.forceDirectedSpring * mass1 * mass2 / (distance * distance + 0.1);
        }

        const maxForce = 50.0;
        forceMagnitude = Math.min(Math.max(forceMagnitude, -maxForce), maxForce);

        return {
            fx: (dx / distance) * forceMagnitude,
            fy: (dy / distance) * forceMagnitude,
            fz: (dz / distance) * forceMagnitude
        };
    }

    applyForceDirectedLayout(graphData, onComplete) {
        const nodes = graphData.nodes;
        
        // Initialize positions if needed
        if (!this.isInitialized) {
            this.initializePositions(nodes);
            this.isInitialized = true;
            // Start continuous refinement
            this.startContinuousSimulation(graphData);
        }

        // Perform a single iteration of force-directed layout
        this.performLayout(graphData, this.updateIterations);
        
        if (onComplete) {
            onComplete();
        }
    }

    performLayout(graphData, iterations) {
        const nodes = graphData.nodes;
        const edges = graphData.edges;

        for (let iteration = 0; iteration < iterations; iteration++) {
            // Calculate forces between all nodes
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const force = this.calculateSpringForce(
                        nodes[i], nodes[j],
                        nodes[i].mass || 1, nodes[j].mass || 1,
                        false
                    );

                    nodes[i].vx -= force.fx;
                    nodes[i].vy -= force.fy;
                    nodes[i].vz -= force.fz;
                    nodes[j].vx += force.fx;
                    nodes[j].vy += force.fy;
                    nodes[j].vz += force.fz;
                }
            }

            // Calculate forces along edges
            edges.forEach(edge => {
                const source = nodes.find(node => node.id === edge.source);
                const target = nodes.find(node => node.id === edge.target_node);
                if (source && target) {
                    const force = this.calculateSpringForce(
                        source, target,
                        source.mass || 1, target.mass || 1,
                        true
                    );

                    source.vx += force.fx;
                    source.vy += force.fy;
                    source.vz += force.fz;
                    target.vx -= force.fx;
                    target.vy -= force.fy;
                    target.vz -= force.fz;
                }
            });

            // Apply centering force and update positions
            const maxVelocity = 10.0;
            nodes.forEach(node => {
                // Centering force
                const distance = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
                if (distance > this.targetRadius) {
                    const centerForce = 0.05 * (distance - this.targetRadius); // Reduced strength
                    node.vx -= (node.x / distance) * centerForce;
                    node.vy -= (node.y / distance) * centerForce;
                    node.vz -= (node.z / distance) * centerForce;
                }

                // Apply damping
                node.vx *= this.forceDirectedDamping;
                node.vy *= this.forceDirectedDamping;
                node.vz *= this.forceDirectedDamping;

                // Limit velocity
                const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy + node.vz * node.vz);
                if (speed > maxVelocity) {
                    const scale = maxVelocity / speed;
                    node.vx *= scale;
                    node.vy *= scale;
                    node.vz *= scale;
                }

                // Update positions with smaller step size for smoother motion
                const stepSize = 0.5;
                node.x += node.vx * stepSize;
                node.y += node.vy * stepSize;
                node.z += node.vz * stepSize;

                // Ensure positions stay within bounds
                const maxCoord = 100.0;
                node.x = Math.max(Math.min(node.x, maxCoord), -maxCoord);
                node.y = Math.max(Math.min(node.y, maxCoord), -maxCoord);
                node.z = Math.max(Math.min(node.z, maxCoord), -maxCoord);
            });
        }

        // Check if we should send position updates
        const now = Date.now();
        if (now - this.lastUpdateTime >= this.updateInterval) {
            this.sendPositionUpdates(nodes);
            this.lastUpdateTime = now;
        }
    }

    sendPositionUpdates(nodes) {
        if (!this.lastPositions) return;

        // Convert to array format for binary transmission
        const positions = nodes.map((node, index) => {
            const lastPos = this.lastPositions[index];
            if (!lastPos) return [node.x, node.y, node.z];

            // Check if position has changed significantly
            if (Math.abs(node.x - lastPos.x) > this.updateThreshold ||
                Math.abs(node.y - lastPos.y) > this.updateThreshold ||
                Math.abs(node.z - lastPos.z) > this.updateThreshold) {
                
                // Update last position
                lastPos.x = node.x;
                lastPos.y = node.y;
                lastPos.z = node.z;
                
                return [node.x, node.y, node.z];
            }
            return [lastPos.x, lastPos.y, lastPos.z];
        });

        // Send position update event
        window.dispatchEvent(new CustomEvent('positionUpdate', {
            detail: {
                type: 'PositionUpdate',
                positions: positions
            }
        }));
    }

    startContinuousSimulation(graphData) {
        if (this.isSimulating) return;
        
        this.isSimulating = true;
        const animate = () => {
            if (!this.isSimulating) return;
            
            // Perform continuous gentle refinement
            this.performLayout(graphData, 1);
            this.animationFrameId = requestAnimationFrame(animate);
        };
        
        animate();
    }

    stopSimulation() {
        this.isSimulating = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    updateFeature(control, value) {
        switch (control) {
            case 'force_directed_spring':
                this.forceDirectedSpring = value;
                break;
            case 'force_directed_damping':
                this.forceDirectedDamping = value;
                break;
        }
    }

    // Handle incoming position updates from server
    applyPositionUpdates(positions) {
        if (!this.lastPositions) return;

        // Handle array-based format (new binary format)
        if (Array.isArray(positions)) {
            positions.forEach((position, index) => {
                if (this.lastPositions[index]) {
                    const [x, y, z] = position;
                    this.lastPositions[index] = { x, y, z };
                }
            });
        }
        // Handle legacy object-based format
        else if (typeof positions === 'object') {
            Object.entries(positions).forEach(([index, position]) => {
                const idx = parseInt(index);
                if (this.lastPositions[idx]) {
                    this.lastPositions[idx] = position;
                }
            });
        }
    }
}
