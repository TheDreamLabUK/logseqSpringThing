import * as THREE from 'three';

// Constants
export const BLOOM_LAYER = 1;
export const NORMAL_LAYER = 0;

export const NODE_COLORS = {
    NEW: new THREE.Color(0x00ff88),      // Bright green for very recent files (< 3 days)
    RECENT: new THREE.Color(0x4444ff),    // Blue for recent files (< 7 days)
    MEDIUM: new THREE.Color(0xffaa00),    // Orange for medium-age files (< 30 days)
    OLD: new THREE.Color(0xff4444)        // Red for old files (>= 30 days)
};

export const NODE_SHAPES = {
    FEW_LINKS: 'sphere',      // 0-5 links
    MEDIUM_LINKS: 'box',      // 6-15 links
    MANY_LINKS: 'octahedron'  // 16+ links
};

export class NodeManager {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.nodeMeshes = new Map();
        this.nodeLabels = new Map();
        this.edgeMeshes = new Map();
        
        // Node settings
        this.minNodeSize = 0.1;  // Minimum node size in visualization
        this.maxNodeSize = 5;    // Maximum node size in visualization
        this.nodeSizeScalingFactor = 1;  // Global scaling factor
        this.labelFontSize = 18;
        this.nodeColor = new THREE.Color(0x4444ff);  // Initialize as THREE.Color

        // Edge settings
        this.edgeColor = new THREE.Color(0x4444ff);  // Initialize as THREE.Color
        this.edgeOpacity = 0.6;

        // Server-side node size range (must match constants in file_service.rs)
        this.serverMinNodeSize = 5.0;
        this.serverMaxNodeSize = 50.0;
    }

    getNodeSize(metadata) {
        // Use the node_size from metadata if available
        if (metadata.node_size) {
            // Convert from server's range (5.0-50.0) to visualization range (0.1-5.0)
            const serverSize = parseFloat(metadata.node_size);
            const normalizedSize = (serverSize - this.serverMinNodeSize) / (this.serverMaxNodeSize - this.serverMinNodeSize);
            return this.minNodeSize + (this.maxNodeSize - this.minNodeSize) * normalizedSize * this.nodeSizeScalingFactor;
        }
        
        // Fallback to a default size if node_size is not available
        return this.minNodeSize;
    }

    calculateNodeColor(lastModified) {
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const dayInMs = 24 * 60 * 60 * 1000;
        
        if (age < 3 * dayInMs) return NODE_COLORS.NEW;        // Less than 3 days old
        if (age < 7 * dayInMs) return NODE_COLORS.RECENT;     // Less than 7 days old
        if (age < 30 * dayInMs) return NODE_COLORS.MEDIUM;    // Less than 30 days old
        return NODE_COLORS.OLD;                               // 30 days or older
    }

    createNodeGeometry(size, hyperlinkCount) {
        if (hyperlinkCount < 6) {
            return new THREE.SphereGeometry(size, 32, 32);
        } else if (hyperlinkCount < 16) {
            return new THREE.BoxGeometry(size, size, size);
        } else {
            return new THREE.OctahedronGeometry(size);
        }
    }

    createNodeLabel(text, fileSize, lastModified, hyperlinkCount) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = `${this.labelFontSize}px Arial`;
        
        const nameMetrics = context.measureText(text);
        const infoText = `${this.formatFileSize(fileSize)} | ${this.formatAge(lastModified)} | ${hyperlinkCount} links`;
        const infoMetrics = context.measureText(infoText);
        
        const textWidth = Math.max(nameMetrics.width, infoMetrics.width);
        canvas.width = textWidth + 20;
        canvas.height = this.labelFontSize * 2 + 30;

        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 10, this.labelFontSize);
        
        context.font = `${this.labelFontSize / 2}px Arial`;
        context.fillStyle = 'lightgray';
        context.fillText(infoText, 10, this.labelFontSize + 20);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);
        sprite.layers.set(NORMAL_LAYER);

        return sprite;
    }

    formatFileSize(size) {
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let i = 0;
        while (size >= 1024 && i < units.length - 1) {
            size /= 1024;
            i++;
        }
        return `${size.toFixed(2)} ${units[i]}`;
    }

    formatAge(lastModified) {
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const days = Math.floor(age / (24 * 60 * 60 * 1000));
        
        if (days < 1) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days}d ago`;
        if (days < 30) return `${Math.floor(days / 7)}w ago`;
        if (days < 365) return `${Math.floor(days / 30)}m ago`;
        return `${Math.floor(days / 365)}y ago`;
    }

    centerNodes(nodes) {
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

        // Subtract center from all positions to center around origin
        nodes.forEach(node => {
            node.x -= centerX;
            node.y -= centerY;
            node.z -= centerZ;
        });

        // Scale positions to reasonable range
        const maxDist = nodes.reduce((max, node) => {
            const dist = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
            return Math.max(max, dist);
        }, 0);

        if (maxDist > 0) {
            const scale = 100 / maxDist; // Scale to fit in 100 unit radius
            nodes.forEach(node => {
                node.x *= scale;
                node.y *= scale;
                node.z *= scale;
            });
        }
    }

    updateNodes(nodes) {
        console.log(`Updating nodes: ${nodes.length}`);
        
        // Center and scale nodes
        this.centerNodes(nodes);
        
        const existingNodeIds = new Set(nodes.map(node => node.id));

        // Remove non-existent nodes
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                this.scene.remove(mesh);
                this.nodeMeshes.delete(nodeId);
                const label = this.nodeLabels.get(nodeId);
                if (label) {
                    this.scene.remove(label);
                    this.nodeLabels.delete(nodeId);
                }
            }
        });

        // Update or create nodes
        nodes.forEach(node => {
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node);
                return;
            }

            const metadata = node.metadata || {};
            const fileSize = parseInt(metadata.file_size) || 1;
            const lastModified = metadata.last_modified || new Date().toISOString();
            const hyperlinkCount = parseInt(metadata.hyperlink_count) || 0;

            const size = this.getNodeSize(metadata);
            const color = this.calculateNodeColor(lastModified);

            let mesh = this.nodeMeshes.get(node.id);

            if (!mesh) {
                const geometry = this.createNodeGeometry(size, hyperlinkCount);
                const material = new THREE.MeshStandardMaterial({
                    color: color,
                    metalness: 0.2,  // Reduced for more glow
                    roughness: 0.2,  // Reduced for more glow
                    emissive: color,
                    emissiveIntensity: 1.0  // Increased from 0.5 to 1.0 for stronger glow
                });

                mesh = new THREE.Mesh(geometry, material);
                mesh.layers.enable(BLOOM_LAYER);
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                const label = this.createNodeLabel(node.label || node.id, fileSize, lastModified, hyperlinkCount);
                this.scene.add(label);
                this.nodeLabels.set(node.id, label);
            } else {
                mesh.geometry.dispose();
                mesh.geometry = this.createNodeGeometry(size, hyperlinkCount);
                mesh.material.color.copy(color);
                mesh.material.emissive.copy(color);
                mesh.material.emissiveIntensity = 1.0; // Ensure updated nodes also have strong glow
            }

            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id);
            if (label) {
                label.position.set(node.x, node.y + size + 2, node.z);
            }
        });
    }

    updateEdges(edges) {
        console.log(`Updating edges: ${edges.length}`);
        const existingEdgeKeys = new Set(edges.map(edge => `${edge.source}-${edge.target_node}`));

        // Remove non-existent edges
        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!existingEdgeKeys.has(edgeKey)) {
                this.scene.remove(line);
                this.edgeMeshes.delete(edgeKey);
            }
        });

        // Update or create edges
        edges.forEach(edge => {
            if (!edge.source || !edge.target_node) {
                console.warn('Invalid edge data:', edge);
                return;
            }

            const edgeKey = `${edge.source}-${edge.target_node}`;
            let line = this.edgeMeshes.get(edgeKey);
            const sourceMesh = this.nodeMeshes.get(edge.source);
            const targetMesh = this.nodeMeshes.get(edge.target_node);

            if (!line && sourceMesh && targetMesh) {
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(6);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                const material = new THREE.LineBasicMaterial({
                    color: this.edgeColor,
                    transparent: true,
                    opacity: this.edgeOpacity,
                    linewidth: 2
                });

                line = new THREE.Line(geometry, material);
                line.layers.set(NORMAL_LAYER);
                this.scene.add(line);
                this.edgeMeshes.set(edgeKey, line);
            }

            if (line && sourceMesh && targetMesh) {
                const positions = line.geometry.attributes.position.array;
                positions[0] = sourceMesh.position.x;
                positions[1] = sourceMesh.position.y;
                positions[2] = sourceMesh.position.z;
                positions[3] = targetMesh.position.x;
                positions[4] = targetMesh.position.y;
                positions[5] = targetMesh.position.z;
                line.geometry.attributes.position.needsUpdate = true;
            }
        });
    }

    updateLabelOrientations(camera) {
        this.nodeLabels.forEach(label => {
            label.lookAt(camera.position);
        });
    }

    updateFeature(control, value) {
        console.log(`Updating feature: ${control} = ${value}`);
        switch (control) {
            // Node features
            case 'nodeColor':
                if (typeof value === 'number' || typeof value === 'string') {
                    this.nodeColor = new THREE.Color(value);
                    this.nodeMeshes.forEach(mesh => {
                        if (mesh.material) {
                            mesh.material.color.copy(this.nodeColor);
                            mesh.material.emissive.copy(this.nodeColor);
                        }
                    });
                }
                break;
            case 'nodeSizeScalingFactor':
                this.nodeSizeScalingFactor = value;
                break;
            case 'labelFontSize':
                this.labelFontSize = value;
                break;

            // Edge features
            case 'edgeColor':
                if (typeof value === 'number' || typeof value === 'string') {
                    this.edgeColor = new THREE.Color(value);
                    this.edgeMeshes.forEach(line => {
                        if (line.material) {
                            line.material.color.copy(this.edgeColor);
                        }
                    });
                }
                break;
            case 'edgeOpacity':
                this.edgeOpacity = value;
                this.edgeMeshes.forEach(line => {
                    if (line.material) {
                        line.material.opacity = value;
                    }
                });
                break;
        }
    }

    dispose() {
        // Dispose node resources
        this.nodeMeshes.forEach(mesh => {
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
        });
        this.nodeLabels.forEach(label => {
            if (label.material.map) label.material.map.dispose();
            if (label.material) label.material.dispose();
        });

        // Dispose edge resources
        this.edgeMeshes.forEach(line => {
            if (line.geometry) line.geometry.dispose();
            if (line.material) line.material.dispose();
        });
    }
}
