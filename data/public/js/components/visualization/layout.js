export class LayoutManager {
    constructor() {
        this.forceDirectedIterations = 250;  // Matches .env
        this.forceDirectedSpring = 2.0;      // Matches .env
        this.forceDirectedDamping = 0.8;     // Matches .env
        this.targetRadius = 200;             // Matches CENTER_RADIUS in WGSL
        this.naturalLength = 100;            // Matches NATURAL_LENGTH in WGSL
    }

    initializePositions(nodes) {
        nodes.forEach(node => {
            if (!node.x && !node.y && !node.z) {
                // Initialize new nodes in a sphere
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = this.targetRadius * Math.cbrt(Math.random()); // Cube root for uniform distribution
                
                node.x = r * Math.sin(phi) * Math.cos(theta);
                node.y = r * Math.sin(phi) * Math.sin(theta);
                node.z = r * Math.cos(phi);
            }
            // Initialize velocities
            node.vx = 0;
            node.vy = 0;
            node.vz = 0;
        });
    }

    centerAndScaleGraph(nodes) {
        // Calculate center of mass
        let centerX = 0, centerY = 0, centerZ = 0;
        nodes.forEach(node => {
            centerX += node.x;
            centerY += node.y;
            centerZ += node.z;
        });
        centerX /= nodes.length;
        centerY /= nodes.length;
        centerZ /= nodes.length;

        // Center nodes
        nodes.forEach(node => {
            node.x -= centerX;
            node.y -= centerY;
            node.z -= centerZ;
        });

        // Calculate current radius
        let maxRadius = 0;
        nodes.forEach(node => {
            const r = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
            maxRadius = Math.max(maxRadius, r);
        });

        // Scale to target radius
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
            // Connected nodes: Hooke's law with natural length
            forceMagnitude = this.forceDirectedSpring * (distance - this.naturalLength);
        } else {
            // Unconnected nodes: Inverse square repulsion
            forceMagnitude = -this.forceDirectedSpring * mass1 * mass2 / (distance * distance + 0.1);
        }

        // Limit maximum force
        const maxForce = 50.0;  // Matches MAX_FORCE in WGSL
        forceMagnitude = Math.min(Math.max(forceMagnitude, -maxForce), maxForce);

        return {
            fx: (dx / distance) * forceMagnitude,
            fy: (dy / distance) * forceMagnitude,
            fz: (dz / distance) * forceMagnitude
        };
    }

    applyForceDirectedLayout(graphData) {
        console.log('Applying force-directed layout');
        const nodes = graphData.nodes;
        const edges = graphData.edges;

        // Initialize positions for new nodes
        this.initializePositions(nodes);

        for (let iteration = 0; iteration < this.forceDirectedIterations; iteration++) {
            // Calculate forces between all nodes
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const force = this.calculateSpringForce(
                        nodes[i], nodes[j],
                        nodes[i].mass || 1, nodes[j].mass || 1,
                        false  // Not connected by default
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
                        true  // Connected by edge
                    );

                    source.vx += force.fx;
                    source.vy += force.fy;
                    source.vz += force.fz;
                    target.vx -= force.fx;
                    target.vy -= force.fy;
                    target.vz -= force.fz;
                }
            });

            // Apply centering force
            nodes.forEach(node => {
                const distance = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
                if (distance > this.targetRadius) {
                    const centerForce = 0.1 * (distance - this.targetRadius);  // Matches CENTER_FORCE_STRENGTH in WGSL
                    node.vx -= (node.x / distance) * centerForce;
                    node.vy -= (node.y / distance) * centerForce;
                    node.vz -= (node.z / distance) * centerForce;
                }
            });

            // Update positions with velocity damping
            const maxVelocity = 10.0;  // Matches MAX_VELOCITY in WGSL
            nodes.forEach(node => {
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

            // Periodically center and scale the graph
            if (iteration % 10 === 0) {
                this.centerAndScaleGraph(nodes);
            }
        }

        // Final centering and scaling
        this.centerAndScaleGraph(nodes);

        console.log('Force-directed layout applied');
    }

    updateFeature(control, value) {
        switch (control) {
            case 'force_directed_spring':
                this.forceDirectedSpring = value;
                break;
            case 'force_directed_damping':
                this.forceDirectedDamping = value;
                break;
            case 'forceDirectedIterations':
                this.forceDirectedIterations = value;
                break;
        }
    }
}
