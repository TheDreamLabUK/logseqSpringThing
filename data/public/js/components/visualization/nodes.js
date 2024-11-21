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

// Removed NODE_SHAPES since we're using a single smooth geometry type

export class NodeManager {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.nodeMeshes = new Map();
        this.nodeLabels = new Map();
        this.edgeMeshes = new Map();
        this.nodeData = new Map(); // Store node data for click handling
        
        // Initialize raycaster for click detection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Node settings
        this.minNodeSize = 0.01;  // Reduced from 0.1
        this.maxNodeSize = 0.5;   // Reduced from 5
        this.nodeSizeScalingFactor = 1;  // Reduced scaling factor
        this.labelFontSize = 18;
        this.nodeColor = new THREE.Color(0x4444ff);  // Initialize as THREE.Color

        // Edge settings
        this.edgeColor = new THREE.Color(0x4444ff);  // Initialize as THREE.Color
        this.edgeOpacity = 0.6;

        // Server-side node size range (must match constants in file_service.rs)
        this.serverMinNodeSize = 0.5;  // Reduced from 5.0
        this.serverMaxNodeSize = 5.0;  // Reduced from 50.0

        // Bind click handler
        this.handleClick = this.handleClick.bind(this);
    }

    getNodeSize(metadata) {
        // Use the node_size from metadata if available
        if (metadata.node_size) {
            // Convert from server's range to visualization range
            const serverSize = parseFloat(metadata.node_size);
            const normalizedSize = (serverSize - this.serverMinNodeSize) / 
                                 (this.serverMaxNodeSize - this.serverMinNodeSize);
            return this.minNodeSize + 
                   (this.maxNodeSize - this.minNodeSize) * 
                   normalizedSize * 
                   this.nodeSizeScalingFactor;
        }
        
        // Fallback to minimum size if node_size is not available
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
        // Create a smooth sphere with more segments for better quality
        // Scale detail level based on hyperlink count for performance
        const segments = Math.min(32, Math.max(16, Math.floor(hyperlinkCount / 2) + 16));
        return new THREE.SphereGeometry(size, segments, segments);
    }

    createNodeMaterial(color, age) {
        // Calculate emissive intensity based on age
        // Newer nodes glow more brightly
        const now = Date.now();
        const ageInDays = (now - new Date(age).getTime()) / (24 * 60 * 60 * 1000);
        const maxAge = 30; // days
        const minIntensity = 0.3;
        const maxIntensity = 1.0;
        
        // Normalize age to 0-1 range and invert (newer = brighter)
        const normalizedAge = Math.min(ageInDays / maxAge, 1);
        const emissiveIntensity = maxIntensity - (normalizedAge * (maxIntensity - minIntensity));

        return new THREE.MeshPhysicalMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: emissiveIntensity,
            metalness: 0.2,
            roughness: 0.2,
            transparent: true,
            opacity: 0.9,
            envMapIntensity: 1.0,
            clearcoat: 0.3,
            clearcoatRoughness: 0.2
        });
    }

    createNodeLabel(text, fileSize, lastModified, hyperlinkCount) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = `${this.labelFontSize}px Arial`;
        
        // Measure text dimensions
        const nameMetrics = context.measureText(text);
        const infoText = `${this.formatFileSize(fileSize)} | ${this.formatAge(lastModified)} | ${hyperlinkCount} links`;
        const infoMetrics = context.measureText(infoText);
        
        // Set canvas size to fit text with padding
        const textWidth = Math.max(nameMetrics.width, infoMetrics.width);
        canvas.width = textWidth + 20;  // Add padding
        canvas.height = this.labelFontSize * 2 + 30;  // Height for two lines plus padding

        // Draw semi-transparent background
        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);

        // Draw node name
        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 10, this.labelFontSize);
        
        // Draw info text in smaller size
        context.font = `${this.labelFontSize / 2}px Arial`;
        context.fillStyle = 'lightgray';
        context.fillText(infoText, 10, this.labelFontSize + 20);

        // Create sprite from canvas
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        
        // Scale sprite to maintain readable text size
        sprite.scale.set(canvas.width / 20, canvas.height / 20, 1);
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

    formatNodeNameToUrl(nodeName) {
        // Convert node name to lowercase and replace spaces with %20
        const formattedName = nodeName.toLowerCase().replace(/ /g, '%20');
        return `https://narrativegoldmine.com/#/page/${formattedName}`;
    }

    handleClick(event) {
        // Calculate mouse position in normalized device coordinates (-1 to +1)
        const rect = event.target.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Update the picking ray with the camera and mouse position
        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Calculate objects intersecting the picking ray
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodeMeshes.values()));

        if (intersects.length > 0) {
            // Find the clicked node
            const clickedMesh = intersects[0].object;
            const nodeId = Array.from(this.nodeMeshes.entries())
                .find(([_, mesh]) => mesh === clickedMesh)?.[0];

            if (nodeId) {
                const nodeData = this.nodeData.get(nodeId);
                if (nodeData) {
                    // Open URL in new tab
                    const url = this.formatNodeNameToUrl(nodeData.label || nodeId);
                    window.open(url, '_blank');
                }
            }
        }
    }

    initClickHandling(renderer) {
        // Add click event listener to renderer's DOM element
        renderer.domElement.addEventListener('click', this.handleClick);
    }

    removeClickHandling(renderer) {
        // Remove click event listener
        renderer.domElement.removeEventListener('click', this.handleClick);
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
                this.nodeData.delete(nodeId);
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

            // Store node data for click handling
            this.nodeData.set(node.id, node);

            const metadata = node.metadata || {};
            const fileSize = parseInt(metadata.file_size) || 1;
            const lastModified = metadata.last_modified || new Date().toISOString();
            const hyperlinkCount = parseInt(metadata.hyperlink_count) || 0;

            const size = this.getNodeSize(metadata);
            const color = this.calculateNodeColor(lastModified);

            let mesh = this.nodeMeshes.get(node.id);

            if (!mesh) {
                const geometry = this.createNodeGeometry(size, hyperlinkCount);
                const material = this.createNodeMaterial(color, lastModified);

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
                mesh.material.dispose();
                mesh.material = this.createNodeMaterial(color, lastModified);
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
        
        // Create a map of edges with their weights from topic counts
        const edgeWeights = new Map();
        edges.forEach(edge => {
            if (!edge.source || !edge.target_node) {
                console.warn('Invalid edge data:', edge);
                return;
            }

            const edgeKey = `${edge.source}-${edge.target_node}`;
            const weight = edge.weight || 1; // Use provided weight or default to 1
            edgeWeights.set(edgeKey, weight);
        });

        // Remove non-existent edges
        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!edgeWeights.has(edgeKey)) {
                this.scene.remove(line);
                this.edgeMeshes.delete(edgeKey);
            }
        });

        // Update or create edges
        edgeWeights.forEach((weight, edgeKey) => {
            const [source, target] = edgeKey.split('-');
            let line = this.edgeMeshes.get(edgeKey);
            const sourceMesh = this.nodeMeshes.get(source);
            const targetMesh = this.nodeMeshes.get(target);

            if (!line && sourceMesh && targetMesh) {
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(6);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                // Scale edge opacity based on weight
                const normalizedWeight = Math.min(weight / 10, 1); // Normalize weight, cap at 1
                const material = new THREE.LineBasicMaterial({
                    color: this.edgeColor,
                    transparent: true,
                    opacity: this.edgeOpacity * normalizedWeight,
                    linewidth: Math.max(1, Math.min(weight, 5)) // Scale line width with weight, between 1-5
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

                // Update edge appearance based on weight
                const normalizedWeight = Math.min(weight / 10, 1);
                line.material.opacity = this.edgeOpacity * normalizedWeight;
                line.material.linewidth = Math.max(1, Math.min(weight, 5));
            }
        });
    }

    updateLabelOrientations(camera) {
        this.nodeLabels.forEach((label, nodeId) => {
            const mesh = this.nodeMeshes.get(nodeId);
            if (mesh) {
                // Position label closer to node due to smaller size
                const size = mesh.geometry.parameters.radius || 
                           mesh.geometry.parameters.width || 
                           this.minNodeSize;
                label.position.set(
                    mesh.position.x,
                    mesh.position.y + size + 0.2, // Reduced offset
                    mesh.position.z
                );
                label.lookAt(camera.position);
            }
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

    updateNodePositions(positions) {
        if (!positions) {
            console.warn('Received null or undefined positions');
            return;
        }

        // Ensure positions is an array
        const positionsArray = Array.isArray(positions) ? positions : Array.from(positions);
        console.log('Updating positions for', positionsArray.length, 'nodes');

        // Get array of node IDs
        const nodeIds = Array.from(this.nodeMeshes.keys());

        // Update each node position
        positionsArray.forEach((position, index) => {
            const nodeId = nodeIds[index];
            if (!nodeId) {
                console.warn(`No node found for index ${index}`);
                return;
            }

            const mesh = this.nodeMeshes.get(nodeId);
            const label = this.nodeLabels.get(nodeId);
            
            if (!mesh) {
                console.warn(`No mesh found for node ${nodeId}`);
                return;
            }

            // Extract position values, handling both array and object formats
            let x, y, z;
            if (Array.isArray(position)) {
                [x, y, z] = position;
            } else if (position && typeof position === 'object') {
                ({ x, y, z } = position);
            } else {
                console.warn(`Invalid position format for node ${nodeId}:`, position);
                return;
            }

            // Validate position values
            if (typeof x === 'number' && typeof y === 'number' && typeof z === 'number' &&
                !isNaN(x) && !isNaN(y) && !isNaN(z)) {
                
                // Update mesh position
                mesh.position.set(x, y, z);
                
                // Update label position if it exists
                if (label) {
                    const size = mesh.geometry.parameters.radius || 
                               mesh.geometry.parameters.width || 
                               1; // fallback size
                    label.position.set(x, y + size + 2, z);
                }

                // Update connected edges
                this.updateEdgesForNode(nodeId);
            } else {
                console.warn(`Invalid position values for node ${nodeId}: x=${x}, y=${y}, z=${z}`);
            }
        });
    }

    updateEdgesForNode(nodeId) {
        this.edgeMeshes.forEach((line, edgeKey) => {
            const [source, target] = edgeKey.split('-');
            if (source === nodeId || target === nodeId) {
                const positions = line.geometry.attributes.position.array;
                const sourceMesh = this.nodeMeshes.get(source);
                const targetMesh = this.nodeMeshes.get(target);

                if (sourceMesh && targetMesh) {
                    positions[0] = sourceMesh.position.x;
                    positions[1] = sourceMesh.position.y;
                    positions[2] = sourceMesh.position.z;
                    positions[3] = targetMesh.position.x;
                    positions[4] = targetMesh.position.y;
                    positions[5] = targetMesh.position.z;
                    line.geometry.attributes.position.needsUpdate = true;
                }
            }
        });
    }

    getNodePositions() {
        return Array.from(this.nodeMeshes.values()).map(mesh => [
            mesh.position.x,
            mesh.position.y,
            mesh.position.z
        ]);
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

        // Clear data maps
        this.nodeMeshes.clear();
        this.nodeLabels.clear();
        this.edgeMeshes.clear();
        this.nodeData.clear();
    }
}
