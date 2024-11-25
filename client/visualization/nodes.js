import * as THREE from 'three';
import { visualizationSettings } from '../../services/visualizationSettings.js';
import { LAYERS, LAYER_GROUPS, LayerManager } from './layerManager.js';

export class NodeManager {
    constructor(scene, camera, settings = {}) {
        // Initialize required properties first
        this.scene = scene;
        this.camera = camera;
        this.nodeData = new Map();
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Wait for settings before initializing
        this.initialized = false;
        this.pendingInitialization = true;

        // Store references that will be initialized once settings are received
        this.nodeInstancedMeshes = null;
        this.edgeInstancedMesh = null;
        this.instancedContainer = null;
        this.labelPool = new Map();

        // Initialize settings with defaults - will be updated from server
        const nodeSettings = visualizationSettings.getNodeSettings();
        const xrSettings = visualizationSettings.getXRSettings();
        
        if (!nodeSettings) {
            console.warn('Node settings not yet received from server');
            return;
        }

        // Initialize with server settings
        this.minNodeSize = nodeSettings.minNodeSize;
        this.maxNodeSize = nodeSettings.maxNodeSize;
        this.labelFontSize = nodeSettings.labelFontSize;
        this.nodeColor = new THREE.Color(nodeSettings.color);
        this.materialSettings = nodeSettings.material;
        this.ageColors = {
            NEW: new THREE.Color(nodeSettings.colorNew),
            RECENT: new THREE.Color(nodeSettings.colorRecent),
            MEDIUM: new THREE.Color(nodeSettings.colorMedium),
            OLD: new THREE.Color(nodeSettings.colorOld)
        };
        this.maxAge = nodeSettings.ageMaxDays;

        const edgeSettings = visualizationSettings.getEdgeSettings();
        if (!edgeSettings) {
            console.warn('Edge settings not yet received from server');
            return;
        }

        this.edgeColor = new THREE.Color(edgeSettings.color);
        this.edgeOpacity = edgeSettings.opacity;

        // XR properties
        this.xrEnabled = false;
        this.xrController = null;
        this.xrHitTestSource = null;
        this.xrInteractionRadius = xrSettings?.interactionRadius || 0.2;
        this.xrHapticStrength = xrSettings?.hapticStrength || 0.5;
        this.xrHapticDuration = xrSettings?.hapticDuration || 50;
        this.xrMinDistance = xrSettings?.minInteractionDistance || 0.1;
        this.xrMaxDistance = xrSettings?.maxInteractionDistance || 5.0;
        this.xrScale = xrSettings?.nodeScale || 0.1;

        // Create container for instanced meshes
        this.instancedContainer = new THREE.Group();
        this.instancedContainer.name = 'instancedContainer';
        this.scene.add(this.instancedContainer);

        // Initialize instanced rendering
        this.initInstancedMeshes();
        
        // Label pooling
        this.labelCanvas = document.createElement('canvas');
        this.labelContext = this.labelCanvas.getContext('2d', {
            alpha: true,
            desynchronized: true,
            willReadFrequently: false
        });

        // Matrix and vector reuse
        this.matrix = new THREE.Matrix4();
        this.quaternion = new THREE.Quaternion();
        this.position = new THREE.Vector3();
        this.scale = new THREE.Vector3();
        this.color = new THREE.Color();

        // XR interaction helpers
        this.xrRaycaster = new THREE.Raycaster();
        this.tempMatrix = new THREE.Matrix4();
        this.instancePositions = new Float32Array(30000); // For XR hit testing
        this.instanceSizes = new Float32Array(10000);
        this.instanceIds = new Map(); // Map positions to node IDs

        // Initialize projection matrix and frustum for label updates
        this.projScreenMatrix = new THREE.Matrix4();
        this.frustum = new THREE.Frustum();

        // Bind methods
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        this.handleXRStateChange = this.handleXRStateChange.bind(this);
        this.handleXRSelect = this.handleXRSelect.bind(this);
        this.updateLabelOrientations = this.updateLabelOrientations.bind(this);
        this.updateNodes = this.updateNodes.bind(this);
        this.updateEdges = this.updateEdges.bind(this);
        
        // Add event listeners
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        window.addEventListener('xrsessionstart', () => this.handleXRStateChange(true));
        window.addEventListener('xrsessionend', () => this.handleXRStateChange(false));

        console.log('NodeManager initialized with scene:', !!scene, 'camera:', !!camera);
        this.initialized = true;
    }

    handleSettingsUpdate(event) {
        console.log('Received settings update:', event.detail);
        this.updateFromSettings(event.detail);
    }

    updateFromSettings(settings) {
        if (!settings?.visualization) return;

        const vis = settings.visualization;
        
        // Update colors
        this.nodeColor = new THREE.Color(vis.node_color);
        this.ageColors = {
            NEW: new THREE.Color(vis.node_color_new),
            RECENT: new THREE.Color(vis.node_color_recent),
            MEDIUM: new THREE.Color(vis.node_color_medium),
            OLD: new THREE.Color(vis.node_color_old)
        };
        this.edgeColor = new THREE.Color(vis.edge_color);

        // Update sizes
        this.minNodeSize = vis.min_node_size;
        this.maxNodeSize = vis.max_node_size;
        this.edgeOpacity = vis.edge_opacity;

        // Update material settings
        this.materialSettings = {
            metalness: vis.node_material_metalness,
            roughness: vis.node_material_roughness,
            clearcoat: vis.node_material_clearcoat,
            clearcoatRoughness: vis.node_material_clearcoat_roughness,
            opacity: vis.node_material_opacity,
            emissiveMinIntensity: vis.node_emissive_min_intensity,
            emissiveMaxIntensity: vis.node_emissive_max_intensity
        };

        // Update materials
        if (this.nodeInstancedMeshes) {
            Object.values(this.nodeInstancedMeshes).forEach(mesh => {
                if (mesh.material) {
                    mesh.material.color.copy(this.nodeColor);
                    mesh.material.emissive.copy(this.nodeColor);
                    mesh.material.metalness = this.materialSettings.metalness;
                    mesh.material.roughness = this.materialSettings.roughness;
                    mesh.material.opacity = this.materialSettings.opacity;
                    mesh.material.emissiveIntensity = this.materialSettings.emissiveMinIntensity;
                    mesh.material.needsUpdate = true;
                }
            });
        }

        if (this.edgeInstancedMesh?.material) {
            this.edgeInstancedMesh.material.color.copy(this.edgeColor);
            this.edgeInstancedMesh.material.opacity = this.edgeOpacity;
            this.edgeInstancedMesh.material.needsUpdate = true;
        }

        // Update existing nodes and edges
        if (this.nodeData.size > 0) {
            const nodes = Array.from(this.nodeData.values());
            this.updateNodes(nodes);
        }
    }

    initInstancedMeshes() {
        try {
            console.log('Initializing instanced meshes');
            
            // Create geometries for different detail levels
            const highDetailGeometry = new THREE.SphereGeometry(1, 32, 32);
            const mediumDetailGeometry = new THREE.SphereGeometry(1, 16, 16);
            const lowDetailGeometry = new THREE.SphereGeometry(1, 8, 8);

            // Create material with instance color support and increased emission
            const nodeMaterial = new THREE.MeshStandardMaterial({
                metalness: this.materialSettings.metalness,
                roughness: this.materialSettings.roughness,
                transparent: false,
                opacity: this.materialSettings.opacity,
                emissive: this.nodeColor,
                emissiveIntensity: this.materialSettings.emissiveMinIntensity
            });

            // Initialize instance attributes
            const maxInstances = 10000; // Adjust based on expected graph size
            
            // Create instanced meshes for different detail levels
            this.nodeInstancedMeshes = {
                high: new THREE.InstancedMesh(highDetailGeometry, nodeMaterial.clone(), maxInstances),
                medium: new THREE.InstancedMesh(mediumDetailGeometry, nodeMaterial.clone(), maxInstances),
                low: new THREE.InstancedMesh(lowDetailGeometry, nodeMaterial.clone(), maxInstances)
            };

            // Add meshes to container and enable all layers for XR compatibility
            Object.values(this.nodeInstancedMeshes).forEach(mesh => {
                mesh.count = 0;
                mesh.layers.enableAll();
                mesh.frustumCulled = true;
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                this.instancedContainer.add(mesh);
            });

            // Create edge instanced mesh with thicker edges
            const edgeGeometry = new THREE.CylinderGeometry(0.15, 0.15, 1, 8);
            edgeGeometry.rotateX(Math.PI / 2);
            const edgeMaterial = new THREE.MeshBasicMaterial({
                color: this.edgeColor,
                transparent: true,
                opacity: this.edgeOpacity,
                depthWrite: false
            });

            this.edgeInstancedMesh = new THREE.InstancedMesh(
                edgeGeometry,
                edgeMaterial,
                maxInstances * 2
            );
            this.edgeInstancedMesh.count = 0;
            this.edgeInstancedMesh.layers.enableAll();
            this.edgeInstancedMesh.frustumCulled = true;
            this.instancedContainer.add(this.edgeInstancedMesh);

            // Initialize dummy matrix and color for instances
            const dummyMatrix = new THREE.Matrix4();
            const dummyColor = new THREE.Color();

            // Pre-initialize instance attributes for better performance
            for (let i = 0; i < maxInstances; i++) {
                Object.values(this.nodeInstancedMeshes).forEach(mesh => {
                    mesh.setMatrixAt(i, dummyMatrix);
                    mesh.setColorAt(i, dummyColor);
                });
                this.edgeInstancedMesh.setMatrixAt(i, dummyMatrix);
            }

            // Update instance buffers
            Object.values(this.nodeInstancedMeshes).forEach(mesh => {
                mesh.instanceMatrix.needsUpdate = true;
                mesh.instanceColor.needsUpdate = true;
            });
            this.edgeInstancedMesh.instanceMatrix.needsUpdate = true;

            console.log('Instanced meshes initialized successfully');
        } catch (error) {
            console.error('Error initializing instanced meshes:', error);
            throw error;
        }
    }

    handleXRStateChange(isXR) {
        console.log('XR state changed:', isXR);
        this.xrEnabled = isXR;
        
        // Update container scale for XR
        const scale = isXR ? this.xrScale : 1;
        this.instancedContainer.scale.setScalar(scale);

        if (isXR) {
            // Set up XR controller if needed
            if (!this.xrController && this.scene.parent?.renderer?.xr) {
                const controller = this.scene.parent.renderer.xr.getController(0);
                controller.addEventListener('select', this.handleXRSelect);
                this.scene.add(controller);
                this.xrController = controller;
            }
        } else {
            // Clean up XR controller
            if (this.xrController) {
                this.xrController.removeEventListener('select', this.handleXRSelect);
                this.scene.remove(this.xrController);
                this.xrController = null;
            }
        }

        // Update all instances with new scale
        if (this.nodeData.size > 0) {
            const nodes = Array.from(this.nodeData.values());
            this.updateNodes(nodes);
        }
    }

    handleXRSelect(event) {
        if (!this.xrEnabled || !this.xrController) return;

        // Get controller position and direction
        this.tempMatrix.identity().extractRotation(this.xrController.matrixWorld);
        this.xrRaycaster.ray.origin.setFromMatrixPosition(this.xrController.matrixWorld);
        this.xrRaycaster.ray.direction.set(0, 0, -1).applyMatrix4(this.tempMatrix);

        // Check intersection with node instances
        const intersects = [];
        let index = 0;
        for (let i = 0; i < this.instancePositions.length; i += 3) {
            const position = new THREE.Vector3(
                this.instancePositions[i],
                this.instancePositions[i + 1],
                this.instancePositions[i + 2]
            );

            // Apply XR scale to position check
            position.multiplyScalar(this.xrScale);
            
            const size = this.instanceSizes[index] * this.xrScale;
            const distance = this.xrRaycaster.ray.origin.distanceTo(position);
            
            // Only check nodes within XR interaction distance range
            if (distance >= this.xrMinDistance && distance <= this.xrMaxDistance) {
                // Create bounding sphere for hit testing
                const sphere = new THREE.Sphere(position, size + this.xrInteractionRadius);
                if (this.xrRaycaster.ray.intersectsSphere(sphere)) {
                    intersects.push({ distance, index, position });
                }
            }
            index++;
        }

        // Handle closest intersection
        if (intersects.length > 0) {
            intersects.sort((a, b) => a.distance - b.distance);
            const closest = intersects[0];
            const nodeId = this.instanceIds.get(`${closest.position.toArray()}`);
            
            if (nodeId) {
                const node = this.nodeData.get(nodeId);
                if (node) {
                    // Provide haptic feedback
                    if (this.xrController.gamepad?.hapticActuators?.length > 0) {
                        this.xrController.gamepad.hapticActuators[0].pulse(
                            this.xrHapticStrength,
                            this.xrHapticDuration
                        );
                    }

                    // Open URL
                    const url = this.formatNodeNameToUrl(node.label || node.id);
                    window.open(url, '_blank');
                }
            }
        }
    }

    updateNodes(nodes) {
        if (!Array.isArray(nodes)) {
            console.error('updateNodes received invalid nodes:', nodes);
            return;
        }

        // Update node data map
        this.nodeData.clear();
        this.instanceIds.clear();
        let positionIndex = 0;
        let sizeIndex = 0;

        // Reset instance counts
        Object.values(this.nodeInstancedMeshes).forEach(mesh => mesh.count = 0);
        
        // Update instances
        nodes.forEach((node, index) => {
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node);
                return;
            }

            this.nodeData.set(node.id, node);

            const metadata = node.metadata || {};
            const size = this.getNodeSize(metadata);
            const color = this.calculateNodeColor(metadata);

            // Store position and size for XR hit testing
            this.instancePositions[positionIndex] = node.x;
            this.instancePositions[positionIndex + 1] = node.y;
            this.instancePositions[positionIndex + 2] = node.z;
            this.instanceSizes[sizeIndex] = size;
            this.instanceIds.set(`${[node.x, node.y, node.z]}`, node.id);

            // Set position and scale
            this.position.set(node.x, node.y, node.z);
            this.scale.set(size, size, size);
            this.matrix.compose(this.position, this.quaternion, this.scale);

            // Determine detail level based on size/distance
            const distance = this.camera.position.distanceTo(this.position);
            let targetMesh;
            if (distance < 50) targetMesh = this.nodeInstancedMeshes.high;
            else if (distance < 100) targetMesh = this.nodeInstancedMeshes.medium;
            else targetMesh = this.nodeInstancedMeshes.low;

            // Update instance
            targetMesh.setMatrixAt(targetMesh.count, this.matrix);
            targetMesh.setColorAt(targetMesh.count, color);
            targetMesh.count++;

            // Update indices
            positionIndex += 3;
            sizeIndex++;

            // Update label if needed and not in XR mode
            if (!this.xrEnabled) {
                this.updateNodeLabel(node, size);
            }
        });

        // Update instance buffers
        Object.values(this.nodeInstancedMeshes).forEach(mesh => {
            mesh.instanceMatrix.needsUpdate = true;
            mesh.instanceColor.needsUpdate = true;
        });
    }

    updateEdges(edges) {
        if (!Array.isArray(edges)) return;

        this.edgeInstancedMesh.count = 0;
        let edgeIndex = 0;

        edges.forEach(edge => {
            if (!edge.source || !edge.target_node) return;

            const sourceNode = this.nodeData.get(edge.source);
            const targetNode = this.nodeData.get(edge.target_node);
            if (!sourceNode || !targetNode) return;

            // Calculate edge position and direction
            const start = new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z);
            const end = new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z);
            const direction = end.clone().sub(start);
            const length = direction.length();

            // Skip if edge length is zero
            if (length === 0) return;

            // Calculate edge center and orientation
            const center = start.clone().add(end).multiplyScalar(0.5);
            this.position.copy(center);

            // Create rotation quaternion from direction
            direction.normalize();
            this.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

            // Set scale with thicker edges
            const edgeWidth = Math.max(0.15, Math.min(edge.weight || 1, 0.4));
            this.scale.set(edgeWidth, length, edgeWidth);

            // Update instance transform
            this.matrix.compose(this.position, this.quaternion, this.scale);
            this.edgeInstancedMesh.setMatrixAt(edgeIndex, this.matrix);

            edgeIndex++;
        });

        // Update instance count and buffer
        this.edgeInstancedMesh.count = edgeIndex;
        this.edgeInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    updateNodeLabel(node, size) {
        const existingLabel = this.labelPool.get(node.id);
        if (existingLabel) {
            existingLabel.position.set(node.x, node.y + size * 1.5, node.z);
            existingLabel.visible = true;
            return;
        }

        // Create new label only if within view distance
        const distance = this.camera.position.distanceTo(new THREE.Vector3(node.x, node.y, node.z));
        if (distance > 100) return;

        const label = this.createNodeLabel(node.label || node.id, node.metadata || {});
        if (label) {
            label.position.set(node.x, node.y + size * 1.5, node.z);
            label.layers.set(LAYERS.NORMAL_LAYER);
            this.scene.add(label);
            this.labelPool.set(node.id, label);
        }
    }

    createNodeLabel(text, metadata) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d', {
            alpha: true,
            desynchronized: true,
            willReadFrequently: false
        });

        if (!context) {
            console.error('Failed to get 2D context for label');
            return null;
        }

        const labelSettings = visualizationSettings.getLabelSettings();
        if (!labelSettings) return null;

        const fontSize = this.xrEnabled ? labelSettings.xrFontSize : labelSettings.fontSize;
        context.font = `${fontSize}px ${labelSettings.fontFamily}`;
        
        const fileSize = parseInt(metadata.file_size) || 1;
        const lastModified = metadata.github_last_modified || metadata.last_modified || new Date().toISOString();
        const hyperlinkCount = parseInt(metadata.hyperlink_count) || 0;
        
        const nameMetrics = context.measureText(text);
        let infoText = `${this.formatFileSize(fileSize)} | ${this.formatAge(lastModified)} | ${hyperlinkCount} links`;
        
        const infoMetrics = context.measureText(infoText);
        const textWidth = Math.max(nameMetrics.width, infoMetrics.width);
        
        const padding = labelSettings.padding;
        const canvasWidth = Math.pow(2, Math.ceil(Math.log2(textWidth + padding * 2)));
        const canvasHeight = Math.pow(2, Math.ceil(Math.log2(fontSize * 3)));

        canvas.width = canvasWidth;
        canvas.height = canvasHeight;

        context.clearRect(0, 0, canvas.width, canvas.height);

        context.fillStyle = labelSettings.backgroundColor;
        const cornerRadius = 8;
        this.roundRect(context, 0, 0, canvas.width, canvas.height, cornerRadius);

        context.font = `${fontSize}px ${labelSettings.fontFamily}`;
        context.textBaseline = 'middle';
        context.textAlign = 'left';
        context.fillStyle = labelSettings.textColor;
        context.fillText(text, padding, canvas.height * 0.35);
        
        context.font = `${fontSize * 0.6}px ${labelSettings.fontFamily}`;
        context.fillStyle = labelSettings.infoTextColor;
        context.fillText(infoText, padding, canvas.height * 0.7);

        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            opacity: 0.9,
            depthWrite: true,
            depthTest: true,
            sizeAttenuation: true
        });

        const sprite = new THREE.Sprite(spriteMaterial);
        
        const labelScale = this.xrEnabled ? 0.4 : 0.8;
        sprite.scale.set(
            (canvas.width / fontSize) * labelScale,
            (canvas.height / fontSize) * labelScale,
            1
        );

        canvas.width = 1;
        canvas.height = 1;

        return sprite;
    }

    roundRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fill();
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
        const baseUrl = 'https://www.narrativegoldmine.com';
        const formattedName = encodeURIComponent(nodeName.toLowerCase());
        return `${baseUrl}/#/page/${formattedName}`;
    }

    getNodeSize(metadata) {
        if (metadata.node_size) {
            const size = parseFloat(metadata.node_size);
            return this.minNodeSize + (Math.pow(size, 1.5) * (this.maxNodeSize - this.minNodeSize));
        }
        return this.minNodeSize;
    }

    calculateNodeColor(metadata) {
        const lastModified = metadata.github_last_modified || metadata.last_modified || new Date().toISOString();
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const dayInMs = 24 * 60 * 60 * 1000;
        
        if (age < 3 * dayInMs) return this.ageColors.NEW;
        if (age < 7 * dayInMs) return this.ageColors.RECENT;
        if (age < 30 * dayInMs) return this.ageColors.MEDIUM;
        return this.ageColors.OLD;
    }

    updateLabelOrientations(camera) {
        this.projScreenMatrix.multiplyMatrices(
            camera.projectionMatrix,
            camera.matrixWorldInverse
        );
        this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

        this.labelPool.forEach((label, nodeId) => {
            const node = this.nodeData.get(nodeId);
            if (node) {
                const position = new THREE.Vector3(node.x, node.y, node.z);
                const distance = camera.position.distanceTo(position);
                const inView = this.frustum.containsPoint(position);
                const notTooFar = distance < 100;

                if (inView && notTooFar) {
                    label.visible = true;
                    label.lookAt(camera.position);
                    
                    const baseScale = this.xrEnabled ? 0.4 : 0.8;
                    const distanceScale = Math.max(0.8, Math.min(2.0, distance * 0.04));
                    const finalScale = baseScale * distanceScale;
                    label.scale.setScalar(finalScale);

                    if (label.material) {
                        const opacity = Math.max(0.2, 1 - (distance / 100));
                        label.material.opacity = opacity;
                    }
                } else {
                    label.visible = false;
                }
            }
        });
    }

    dispose() {
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        window.removeEventListener('xrsessionstart', () => this.handleXRStateChange(true));
        window.removeEventListener('xrsessionend', () => this.handleXRStateChange(false));
        
        if (this.xrController) {
            this.xrController.removeEventListener('select', this.handleXRSelect);
            this.scene.remove(this.xrController);
        }

        Object.values(this.nodeInstancedMeshes).forEach(mesh => {
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
            this.instancedContainer.remove(mesh);
        });

        if (this.edgeInstancedMesh) {
            if (this.edgeInstancedMesh.geometry) this.edgeInstancedMesh.geometry.dispose();
            if (this.edgeInstancedMesh.material) this.edgeInstancedMesh.material.dispose();
            this.instancedContainer.remove(this.edgeInstancedMesh);
        }

        this.labelPool.forEach(label => {
            if (label.material) {
                if (label.material.map) label.material.map.dispose();
                label.material.dispose();
            }
            this.scene.remove(label);
        });

        this.scene.remove(this.instancedContainer);
        
        this.nodeData.clear();
        this.labelPool.clear();
        this.instanceIds.clear();
    }
}
