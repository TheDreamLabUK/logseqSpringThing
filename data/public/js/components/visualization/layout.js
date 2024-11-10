export class LayoutManager {
    constructor() {
        // Configuration
        this.initialIterations = 250;    // High iteration count for initial layout
        this.updateIterations = 1;       // Single iteration for smooth continuous updates
        this.targetRadius = 200;
        this.naturalLength = 100;
        
        // State
        this.isInitialized = false;
        this.isSimulating = false;
        this.animationFrameId = null;
        this.lastPositions = null;       // Store previous positions for change detection
        this.updateThreshold = 0.001;    // Minimum position change to trigger update
        this.lastUpdateTime = 0;         // Last time positions were sent to server
        this.updateInterval = 16.67;     // Exactly 60fps
        this.positionBuffer = null;
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

    initializePositionBuffer(nodeCount) {
        // Pre-allocate buffer for position updates
        this.positionBuffer = new ArrayBuffer(nodeCount * 12);
        this.positionView = new Float32Array(this.positionBuffer);
    }

    // Update position of a node (e.g., from VR interaction)
    updateNodePosition(nodeId, position) {
        const offset = nodeId * 3;
        this.positionView[offset] = position.x;
        this.positionView[offset + 1] = position.y;
        this.positionView[offset + 2] = position.z;
        
        // Mark for update
        this.needsUpdate = true;
    }

    performLayout(graphData) {
        const now = Date.now();
        if (now - this.lastUpdateTime >= this.updateInterval) {
            this.sendPositionUpdates(graphData.nodes);
            this.lastUpdateTime = now;
        }
    }

    sendPositionUpdates(nodes) {
        if (!this.lastPositions) return;

        // Create binary buffer for all node positions (24 bytes per node)
        const buffer = new ArrayBuffer(nodes.length * 24);
        const dataView = new DataView(buffer);
        let hasChanges = false;

        nodes.forEach((node, index) => {
            const offset = index * 24;
            const lastPos = this.lastPositions[index];

            if (!lastPos || 
                Math.abs(node.x - lastPos.x) > this.updateThreshold ||
                Math.abs(node.y - lastPos.y) > this.updateThreshold ||
                Math.abs(node.z - lastPos.z) > this.updateThreshold) {
                
                hasChanges = true;
                
                // Update last position
                if (lastPos) {
                    lastPos.x = node.x;
                    lastPos.y = node.y;
                    lastPos.z = node.z;
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

    // Handle incoming position updates from server
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
                    z: dataView.getFloat32(offset + 8, true)
                };
            }
        }
    }

    startContinuousSimulation(graphData) {
        if (this.isSimulating) return;
        
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
        this.isSimulating = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    update() {
        if (!this.needsUpdate) return;
        
        const now = Date.now();
        if (now - this.lastUpdateTime >= this.updateInterval) {
            this.lastUpdateTime = now;
            
            // Send position updates
            window.dispatchEvent(new CustomEvent('positionUpdate', {
                detail: this.positionBuffer
            }));
            
            this.needsUpdate = false;
        }
    }
}
