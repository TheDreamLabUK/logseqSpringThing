export class LayoutManager {
    constructor() {
        this.forceDirectedIterations = 100;
        this.forceDirectedRepulsion = 100.0;  // Increased from 1.0
        this.forceDirectedAttraction = 0.01;
        this.forceDirectedDamping = 0.9;
        this.targetRadius = 100;  // Target radius for the graph
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

    applyForceDirectedLayout(graphData) {
        console.log('Applying force-directed layout');
        const nodes = graphData.nodes;
        const edges = graphData.edges;

        // Initialize positions for new nodes
        this.initializePositions(nodes);

        // Calculate average edge length for force scaling
        let avgEdgeLength = 0;
        edges.forEach(edge => {
            const source = nodes.find(node => node.id === edge.source);
            const target = nodes.find(node => node.id === edge.target_node);
            if (source && target) {
                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const dz = target.z - source.z;
                avgEdgeLength += Math.sqrt(dx * dx + dy * dy + dz * dz);
            }
        });
        avgEdgeLength = avgEdgeLength / edges.length || this.targetRadius / 2;

        // Scale forces based on graph size
        const repulsionScale = avgEdgeLength * avgEdgeLength;
        const attractionScale = 1 / avgEdgeLength;

        for (let iteration = 0; iteration < this.forceDirectedIterations; iteration++) {
            // Calculate repulsion between all nodes
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[j].x - nodes[i].x;
                    const dy = nodes[j].y - nodes[i].y;
                    const dz = nodes[j].z - nodes[i].z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                    const force = (this.forceDirectedRepulsion * repulsionScale) / (distance * distance);

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

            // Calculate attraction along edges
            edges.forEach(edge => {
                const source = nodes.find(node => node.id === edge.source);
                const target = nodes.find(node => node.id === edge.target_node);
                if (source && target) {
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dz = target.z - source.z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                    const force = this.forceDirectedAttraction * attractionScale * distance * edge.weight;

                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    const fz = (dz / distance) * force;

                    source.vx += fx;
                    source.vy += fy;
                    source.vz += fz;
                    target.vx -= fx;
                    target.vy -= fy;
                    target.vz -= fz;
                }
            });

            // Update positions with velocity damping
            nodes.forEach(node => {
                node.x += node.vx * this.forceDirectedDamping;
                node.y += node.vy * this.forceDirectedDamping;
                node.z += node.vz * this.forceDirectedDamping;
                node.vx *= this.forceDirectedDamping;
                node.vy *= this.forceDirectedDamping;
                node.vz *= this.forceDirectedDamping;
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
            case 'forceDirectedIterations':
                this.forceDirectedIterations = value;
                break;
            case 'forceDirectedRepulsion':
                this.forceDirectedRepulsion = value;
                break;
            case 'forceDirectedAttraction':
                this.forceDirectedAttraction = value;
                break;
            case 'forceDirectedDamping':
                this.forceDirectedDamping = value;
                break;
        }
    }
}
