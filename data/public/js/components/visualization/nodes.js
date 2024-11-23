import * as THREE from 'three';
import { visualizationSettings } from '../../services/visualizationSettings.js';
import { LAYERS, LAYER_GROUPS, LayerManager } from './layerManager.js';

export class NodeManager {
    constructor(scene, camera, settings = {}) {
        this.scene = scene;
        this.camera = camera;
        this.nodeMeshes = new Map();
        this.nodeLabels = new Map();
        this.edgeMeshes = new Map();
        this.nodeData = new Map();
        
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Get settings from visualization settings service
        const nodeSettings = visualizationSettings.getNodeSettings();
        
        // Physical dimensions in meters
        this.minNodeSize = settings.minNodeSize || nodeSettings.minNodeSize;
        this.maxNodeSize = settings.maxNodeSize || nodeSettings.maxNodeSize;
        
        // Visual settings
        this.labelFontSize = settings.labelFontSize || nodeSettings.labelFontSize;
        this.nodeColor = new THREE.Color(settings.nodeColor || nodeSettings.color);
        this.materialSettings = nodeSettings.material;
        this.ageColors = {
            NEW: new THREE.Color(nodeSettings.colorNew),
            RECENT: new THREE.Color(nodeSettings.colorRecent),
            MEDIUM: new THREE.Color(nodeSettings.colorMedium),
            OLD: new THREE.Color(nodeSettings.colorOld)
        };
        this.maxAge = nodeSettings.ageMaxDays;

        // Edge settings
        const edgeSettings = visualizationSettings.getEdgeSettings();
        this.edgeColor = new THREE.Color(settings.edgeColor || edgeSettings.color);
        this.edgeOpacity = settings.edgeOpacity || edgeSettings.opacity;

        this.handleClick = this.handleClick.bind(this);
        this.xrEnabled = false;
        this.xrLabelManager = null;
    }

    centerNodes(nodes) {
        if (!Array.isArray(nodes) || nodes.length === 0) {
            return nodes;
        }

        // Calculate center of mass
        let centerX = 0, centerY = 0, centerZ = 0;
        nodes.forEach(node => {
            centerX += node.x || 0;
            centerY += node.y || 0;
            centerZ += node.z || 0;
        });
        centerX /= nodes.length;
        centerY /= nodes.length;
        centerZ /= nodes.length;

        // Center nodes around origin
        return nodes.map(node => ({
            ...node,
            x: (node.x || 0) - centerX,
            y: (node.y || 0) - centerY,
            z: (node.z || 0) - centerZ
        }));
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
        // Get base URL from environment or default to logseq
        const baseUrl = window.location.origin;
        // Convert node name to lowercase and replace spaces with dashes
        const formattedName = nodeName.toLowerCase().replace(/ /g, '-');
        return `${baseUrl}/#/page/${formattedName}`;
    }

    getNodeSize(metadata) {
        // Calculate node size in meters based on metadata
        if (metadata.node_size) {
            const size = parseFloat(metadata.node_size);
            // Normalize size between minNodeSize (0.1m) and maxNodeSize (0.3m)
            return this.minNodeSize + (size * (this.maxNodeSize - this.minNodeSize));
        }
        return this.minNodeSize; // Default to minimum size (10cm)
    }

    calculateNodeColor(metadata) {
        // Use github_last_modified if available, otherwise fall back to last_modified
        const lastModified = metadata.github_last_modified || metadata.last_modified || new Date().toISOString();
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const dayInMs = 24 * 60 * 60 * 1000;
        
        if (age < 3 * dayInMs) return this.ageColors.NEW;        // Less than 3 days old
        if (age < 7 * dayInMs) return this.ageColors.RECENT;     // Less than 7 days old
        if (age < 30 * dayInMs) return this.ageColors.MEDIUM;    // Less than 30 days old
        return this.ageColors.OLD;                               // 30 days or older
    }

    createNodeGeometry(size, hyperlinkCount) {
        // Create a sphere with radius in meters
        // Scale segments based on hyperlink count for performance vs. quality
        const minSegments = visualizationSettings.getNodeSettings().geometryMinSegments;
        const maxSegments = visualizationSettings.getNodeSettings().geometryMaxSegments;
        const segmentPerLink = visualizationSettings.getNodeSettings().geometrySegmentPerHyperlink;
        
        const segments = Math.min(
            maxSegments,
            Math.max(minSegments, Math.floor(hyperlinkCount * segmentPerLink) + minSegments)
        );
        
        return new THREE.SphereGeometry(size, segments, segments);
    }

    createNodeMaterial(color, metadata) {
        const lastModified = metadata.github_last_modified || metadata.last_modified || new Date().toISOString();
        const now = Date.now();
        const ageInDays = (now - new Date(lastModified).getTime()) / (24 * 60 * 60 * 1000);
        
        const normalizedAge = Math.min(ageInDays / this.maxAge, 1);
        const emissiveIntensity = this.materialSettings.emissiveMaxIntensity - 
            (normalizedAge * (this.materialSettings.emissiveMaxIntensity - this.materialSettings.emissiveMinIntensity));

        return new THREE.MeshPhysicalMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: emissiveIntensity,
            metalness: this.materialSettings.metalness,
            roughness: this.materialSettings.roughness,
            transparent: true,
            opacity: this.materialSettings.opacity,
            envMapIntensity: 1.0,
            clearcoat: this.materialSettings.clearcoat,
            clearcoatRoughness: this.materialSettings.clearcoatRoughness,
            toneMapped: false
        });
    }

    createNodeLabel(text, metadata) {
        // Dispose existing texture if any
        const existingLabel = this.nodeLabels.get(text);
        if (existingLabel && existingLabel.material.map) {
            existingLabel.material.map.dispose();
            existingLabel.material.dispose();
        }

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d', {
            alpha: true,
            desynchronized: true // Optimize for performance
        });
        context.font = `${this.labelFontSize}px Arial`;
        
        // Get metadata values
        const fileSize = parseInt(metadata.file_size) || 1;
        const lastModified = metadata.github_last_modified || metadata.last_modified || new Date().toISOString();
        const hyperlinkCount = parseInt(metadata.hyperlink_count) || 0;
        const githubInfo = metadata.github_info || {};
        
        // Measure and create text
        const nameMetrics = context.measureText(text);
        let infoText = `${this.formatFileSize(fileSize)} | ${this.formatAge(lastModified)} | ${hyperlinkCount} links`;
        if (githubInfo.author) {
            infoText += ` | ${githubInfo.author}`;
        }
        if (githubInfo.commit_message) {
            const shortMessage = githubInfo.commit_message.split('\n')[0].slice(0, 30);
            infoText += ` | ${shortMessage}${githubInfo.commit_message.length > 30 ? '...' : ''}`;
        }
        
        const infoMetrics = context.measureText(infoText);
        const textWidth = Math.max(nameMetrics.width, infoMetrics.width);
        
        // Set canvas size to power of 2 for better texture performance
        const canvasWidth = Math.pow(2, Math.ceil(Math.log2(textWidth + 20)));
        const canvasHeight = Math.pow(2, Math.ceil(Math.log2(this.labelFontSize * 2 + 30)));
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;

        // Draw background and text
        context.fillStyle = visualizationSettings.getLabelSettings().backgroundColor;
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.font = `${this.labelFontSize}px ${visualizationSettings.getLabelSettings().fontFamily}`;
        context.fillStyle = visualizationSettings.getLabelSettings().textColor;
        context.fillText(text, 10, this.labelFontSize);
        
        context.font = `${this.labelFontSize / 2}px ${visualizationSettings.getLabelSettings().fontFamily}`;
        context.fillStyle = visualizationSettings.getLabelSettings().infoTextColor;
        context.fillText(infoText, 10, this.labelFontSize + 20);

        // Create sprite with optimized texture settings
        const texture = new THREE.CanvasTexture(canvas);
        texture.generateMipmaps = false;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.format = THREE.RGBAFormat;
        
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            sizeAttenuation: true,
            toneMapped: false // Important for proper visibility
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        
        // Scale sprite to maintain readable text size in meters
        const labelScale = visualizationSettings.getLabelSettings().verticalOffset;
        sprite.scale.set(
            (canvas.width / this.labelFontSize) * labelScale * 1.5,
            (canvas.height / this.labelFontSize) * labelScale * 1.5,
            1
        );
        
        // Set label to be visible in all necessary layers
        LayerManager.setLayerGroup(sprite, 'LABEL');

        return sprite;
    }

    handleClick(event, isXR = false, intersectedObject = null) {
        let clickedMesh;

        if (isXR && intersectedObject) {
            // In XR mode, use the passed intersected object directly
            clickedMesh = intersectedObject;
        } else if (!isXR && event) {
            // Regular mouse click handling
            const rect = event.target.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);
            const intersects = this.raycaster.intersectObjects(Array.from(this.nodeMeshes.values()));
            
            if (intersects.length > 0) {
                clickedMesh = intersects[0].object;
            }
        }

        if (clickedMesh) {
            // Find the clicked node
            const nodeId = Array.from(this.nodeMeshes.entries())
                .find(([_, mesh]) => mesh === clickedMesh)?.[0];

            if (nodeId) {
                const nodeData = this.nodeData.get(nodeId);
                if (nodeData) {
                    // Open URL in new tab
                    const url = this.formatNodeNameToUrl(nodeData.label || nodeId);
                    window.open(url, '_blank');

                    // Visual feedback
                    const originalEmissive = clickedMesh.material.emissiveIntensity;
                    clickedMesh.material.emissiveIntensity = 2.0;
                    setTimeout(() => {
                        clickedMesh.material.emissiveIntensity = originalEmissive;
                    }, 200);

                    // Show XR label if in XR mode
                    if (isXR && this.xrLabelManager) {
                        this.xrLabelManager.showLabel(
                            nodeData.label || nodeId,
                            clickedMesh.position,
                            {
                                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                color: '#ffffff',
                                font: '24px Arial'
                            }
                        );
                    }

                    // Trigger haptic feedback in XR mode
                    if (isXR && window.xrSession) {
                        const inputSource = Array.from(window.xrSession.inputSources).find(source => 
                            source.handedness === 'right' || source.handedness === 'left'
                        );
                        if (inputSource?.gamepad?.hapticActuators?.length > 0) {
                            inputSource.gamepad.hapticActuators[0].pulse(0.5, 100);
                        }
                    }
                }
            }
        }
    }

    initClickHandling(renderer) {
        renderer.domElement.addEventListener('click', this.handleClick);
    }

    removeClickHandling(renderer) {
        renderer.domElement.removeEventListener('click', this.handleClick);
    }

    updateNodes(nodes) {
        if (!Array.isArray(nodes)) {
            console.error('updateNodes received invalid nodes:', nodes);
            return;
        }

        console.log(`Updating nodes: ${nodes.length}`);
        
        // Center and scale nodes
        const centeredNodes = this.centerNodes(nodes);
        if (!centeredNodes) return;
        
        const existingNodeIds = new Set(centeredNodes.map(node => node.id));

        // Remove non-existent nodes and properly dispose resources
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                if (mesh.geometry) {
                    mesh.geometry.dispose();
                }
                if (mesh.material) {
                    if (Array.isArray(mesh.material)) {
                        mesh.material.forEach(mat => mat.dispose());
                    } else {
                        mesh.material.dispose();
                    }
                }
                this.scene.remove(mesh);
                this.nodeMeshes.delete(nodeId);
                this.nodeData.delete(nodeId);

                const label = this.nodeLabels.get(nodeId);
                if (label) {
                    if (label.material.map) {
                        label.material.map.dispose();
                    }
                    label.material.dispose();
                    this.scene.remove(label);
                    this.nodeLabels.delete(nodeId);
                }
            }
        });

        // Update or create nodes
        centeredNodes.forEach(node => {
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node);
                return;
            }

            // Store node data for click handling
            this.nodeData.set(node.id, node);

            const metadata = node.metadata || {};
            const size = this.getNodeSize(metadata);
            const color = this.calculateNodeColor(metadata);

            let mesh = this.nodeMeshes.get(node.id);

            if (!mesh) {
                const geometry = this.createNodeGeometry(size, metadata.hyperlink_count || 0);
                const material = this.createNodeMaterial(color, metadata);

                mesh = new THREE.Mesh(geometry, material);
                LayerManager.setLayerGroup(mesh, 'BLOOM');
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                const label = this.createNodeLabel(node.label || node.id, metadata);
                this.scene.add(label);
                this.nodeLabels.set(node.id, label);
            } else {
                // Update existing mesh
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
                
                mesh.geometry = this.createNodeGeometry(size, metadata.hyperlink_count || 0);
                mesh.material = this.createNodeMaterial(color, metadata);
                LayerManager.setLayerGroup(mesh, 'BLOOM');
            }

            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id);
            if (label) {
                const labelOffset = size * 1.5;
                label.position.set(node.x, node.y + labelOffset, node.z);
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
                if (line.geometry) {
                    line.geometry.dispose();
                }
                if (line.material) {
                    line.material.dispose();
                }
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

                const normalizedWeight = Math.min(weight / 10, 1);
                const material = new THREE.LineBasicMaterial({
                    color: this.edgeColor,
                    transparent: true,
                    opacity: this.edgeOpacity * normalizedWeight,
                    linewidth: Math.max(1, Math.min(weight, 5)),
                    toneMapped: false
                });

                line = new THREE.Line(geometry, material);
                LayerManager.setLayerGroup(line, 'EDGE');
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
                const size = mesh.geometry.parameters.radius || 
                           mesh.geometry.parameters.width || 
                           this.minNodeSize;
                const labelOffset = size * 1.5; // Increased offset
                label.position.set(
                    mesh.position.x,
                    mesh.position.y + labelOffset,
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
            case 'minNodeSize':
                this.minNodeSize = value; // Value in meters
                break;
            case 'maxNodeSize':
                this.maxNodeSize = value; // Value in meters
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

    updateMaterial(settings) {
        console.log('Updating node material settings:', settings);
        
        // Update material settings
        this.materialSettings = {
            ...this.materialSettings,
            metalness: settings.metalness ?? this.materialSettings.metalness,
            roughness: settings.roughness ?? this.materialSettings.roughness,
            clearcoat: settings.clearcoat ?? this.materialSettings.clearcoat,
            clearcoatRoughness: settings.clearcoatRoughness ?? this.materialSettings.clearcoatRoughness,
            opacity: settings.opacity ?? this.materialSettings.opacity,
            emissiveMinIntensity: settings.emissiveMinIntensity ?? this.materialSettings.emissiveMinIntensity,
            emissiveMaxIntensity: settings.emissiveMaxIntensity ?? this.materialSettings.emissiveMaxIntensity
        };

        // Update all existing node materials
        this.nodeMeshes.forEach((mesh, nodeId) => {
            const nodeData = this.nodeData.get(nodeId);
            if (nodeData && mesh.material) {
                // Create new material with updated settings
                mesh.material.dispose(); // Dispose old material
                mesh.material = this.createNodeMaterial(mesh.material.color, nodeData.metadata || {});
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
            if (mesh.geometry) {
                mesh.geometry.dispose();
            }
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => {
                        if (mat.map) mat.map.dispose();
                        mat.dispose();
                    });
                } else {
                    if (mesh.material.map) mesh.material.map.dispose();
                    mesh.material.dispose();
                }
            }
            if (mesh.parent) {
                mesh.parent.remove(mesh);
            }
        });

        // Dispose label resources
        this.nodeLabels.forEach(label => {
            if (label.material) {
                if (label.material.map) {
                    label.material.map.dispose();
                }
                label.material.dispose();
            }
            if (label.parent) {
                label.parent.remove(label);
            }
        });

        // Dispose edge resources
        this.edgeMeshes.forEach(line => {
            if (line.geometry) {
                line.geometry.dispose();
            }
            if (line.material) {
                if (line.material.map) line.material.map.dispose();
                line.material.dispose();
            }
            if (line.parent) {
                line.parent.remove(line);
            }
        });

        // Clear data maps
        this.nodeMeshes.clear();
        this.nodeLabels.clear();
        this.edgeMeshes.clear();
        this.nodeData.clear();

        // Clean up event listeners
        if (this.renderer) {
            this.removeClickHandling(this.renderer);
        }
    }
}
