export class LayoutManager {
    constructor() {
        // Configuration
        this.initialIterations = 250;    // High iteration count for initial layout
        this.updateIterations = 5;       // Low iteration count for interactive updates
        this.forceDirectedSpring = 2.0;
        this.forceDirectedDamping = 0.8;
        this.targetRadius = 200;
        this.naturalLength = 100;
        
        // State
        this.isInitialized = false;
        this.isSimulating = false;
        this.animationFrameId = null;
        this.useServerPositions = true;  // New flag to respect server positions
    }

    initializePositions(nodes) {
        nodes.forEach(node => {
            // Only initialize if positions are undefined, null, or NaN
            if (isNaN(node.x) || isNaN(node.y) || isNaN(node.z)) {
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = this.targetRadius * Math.cbrt(Math.random());
                
                node.x = r * Math.sin(phi) * Math.cos(theta);
                node.y = r * Math.sin(phi) * Math.sin(theta);
                node.z = r * Math.cos(phi);
            }
            // Always initialize velocities
            node.vx = 0;
            node.vy = 0;
            node.vz = 0;
        });
    }

    centerAndScaleGraph(nodes) {
        // Only center and scale if not using server positions
        if (this.useServerPositions) return;

        let centerX = 0, centerY = 0, centerZ = 0;
        nodes.forEach(node => {
            centerX += node.x;
            centerY += node.y;
            centerZ += node.z;
        });
        centerX /= nodes.length;
        centerY /= nodes.length;
        centerZ /= nodes.length;

        nodes.forEach(node => {
            node.x -= centerX;
            node.y -= centerY;
            node.z -= centerZ;
        });

        let maxRadius = 0;
        nodes.forEach(node => {
            const r = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
            maxRadius = Math.max(maxRadius, r);
        });

        if (maxRadius > 0) {
            const scale = this.targetRadius / maxRadius;
            nodes.forEach(node => {
                node.x *= scale;
                node.y *= scale;
                node.z *= scale;
            });
        }
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
        
        // Check if we have valid server positions
        const hasValidPositions = nodes.every(node => 
            !isNaN(node.x) && !isNaN(node.y) && !isNaN(node.z)
        );

        if (hasValidPositions) {
            this.useServerPositions = true;
            this.isInitialized = true;
            // Don't start continuous simulation when using server positions
            this.stopSimulation();
        } else {
            this.useServerPositions = false;
            if (!this.isInitialized) {
                // Initial layout with high iteration count
                this.performLayout(graphData, this.initialIterations);
                this.isInitialized = true;
            } else {
                // Interactive update with low iteration count
                this.performLayout(graphData, this.updateIterations);
            }
        }
        
        if (onComplete) {
            onComplete();
        }
    }

    performLayout(graphData, iterations) {
        // Skip layout if using server positions
        if (this.useServerPositions) return;

        const nodes = graphData.nodes;
        const edges = graphData.edges;

        if (!this.isInitialized) {
            this.initializePositions(nodes);
        }

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
                    const centerForce = 0.1 * (distance - this.targetRadius);
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

                // Update positions
                node.x += node.vx;
                node.y += node.vy;
                node.z += node.vz;
            });

            if (iteration % 10 === 0) {
                this.centerAndScaleGraph(nodes);
            }
        }

        this.centerAndScaleGraph(nodes);
    }

    startContinuousSimulation(graphData) {
        // Don't start simulation if using server positions
        if (this.useServerPositions || this.isSimulating) return;
        
        this.isSimulating = true;
        const animate = () => {
            if (!this.isSimulating) return;
            
            this.performLayout(graphData, 1);  // Single iteration per frame
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
}
