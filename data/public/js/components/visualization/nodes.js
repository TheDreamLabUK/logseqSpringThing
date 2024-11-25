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
        
        const nodeSettings = visualizationSettings.getNodeSettings();
        
        this.minNodeSize = settings.minNodeSize || nodeSettings.minNodeSize;
        this.maxNodeSize = settings.maxNodeSize || nodeSettings.maxNodeSize;
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

        const edgeSettings = visualizationSettings.getEdgeSettings();
        this.edgeColor = new THREE.Color(settings.edgeColor || edgeSettings.color);
        this.edgeOpacity = settings.edgeOpacity || edgeSettings.opacity;

        this.handleClick = this.handleClick.bind(this);
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

        this.xrEnabled = false;
        this.xrLabelManager = null;
        this.frustum = new THREE.Frustum();
        this.projScreenMatrix = new THREE.Matrix4();
    }

    handleSettingsUpdate(event) {
        const settings = event.detail;
        if (!settings) return;

        if (settings.visual) {
            if (settings.visual.nodeColor !== undefined) {
                this.updateFeature('nodeColor', settings.visual.nodeColor);
            }
            if (settings.visual.minNodeSize !== undefined) {
                this.updateFeature('minNodeSize', settings.visual.minNodeSize);
            }
            if (settings.visual.maxNodeSize !== undefined) {
                this.updateFeature('maxNodeSize', settings.visual.maxNodeSize);
            }
            if (settings.visual.labelFontSize !== undefined) {
                this.updateFeature('labelFontSize', settings.visual.labelFontSize);
            }
            if (settings.visual.edgeColor !== undefined) {
                this.updateFeature('edgeColor', settings.visual.edgeColor);
            }
            if (settings.visual.edgeOpacity !== undefined) {
                this.updateFeature('edgeOpacity', settings.visual.edgeOpacity);
            }
        }

        if (settings.material) {
            this.updateMaterial(settings.material);
        }

        if (settings.ageColors) {
            if (settings.ageColors.new) this.ageColors.NEW.set(settings.ageColors.new);
            if (settings.ageColors.recent) this.ageColors.RECENT.set(settings.ageColors.recent);
            if (settings.ageColors.medium) this.ageColors.MEDIUM.set(settings.ageColors.medium);
            if (settings.ageColors.old) this.ageColors.OLD.set(settings.ageColors.old);
            
            this.nodeMeshes.forEach((mesh, nodeId) => {
                const nodeData = this.nodeData.get(nodeId);
                if (nodeData) {
                    const color = this.calculateNodeColor(nodeData.metadata || {});
                    if (mesh.material) {
                        mesh.material.color.copy(color);
                        mesh.material.emissive.copy(color);
                    }
                }
            });
        }
    }

    centerNodes(nodes) {
        if (!Array.isArray(nodes) || nodes.length === 0) {
            return nodes;
        }

        let centerX = 0, centerY = 0, centerZ = 0;
        nodes.forEach(node => {
            centerX += node.x || 0;
            centerY += node.y || 0;
            centerZ += node.z || 0;
        });
        centerX /= nodes.length;
        centerY /= nodes.length;
        centerZ /= nodes.length;

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
        const baseUrl = 'https://www.narrativegoldmine.com';
        const formattedName = encodeURIComponent(nodeName.toLowerCase());
        return `${baseUrl}/#/page/${formattedName}`;
    }

    getNodeSize(metadata) {
        if (metadata.node_size) {
            const size = parseFloat(metadata.node_size);
            return this.minNodeSize + (size * (this.maxNodeSize - this.minNodeSize));
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

    createNodeGeometry(size, hyperlinkCount) {
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
        const material = new THREE.MeshStandardMaterial({
            color: color,
            metalness: 0.1,
            roughness: 0.7,
            transparent: false,
            opacity: 1.0,
            emissive: color,
            emissiveIntensity: 0.5,
            side: THREE.FrontSide,
            depthWrite: true,
            depthTest: true
        });

        console.log('Created node material:', {
            color: color.getHexString(),
            metalness: material.metalness,
            roughness: material.roughness,
            emissiveIntensity: material.emissiveIntensity
        });

        return material;
    }

    createNodeLabel(text, metadata) {
        const existingLabel = this.nodeLabels.get(text);
        if (existingLabel && existingLabel.material.map) {
            existingLabel.material.map.dispose();
            existingLabel.material.dispose();
        }

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

        context.font = `${this.labelFontSize}px Arial`;
        
        const fileSize = parseInt(metadata.file_size) || 1;
        const lastModified = metadata.github_last_modified || metadata.last_modified || new Date().toISOString();
        const hyperlinkCount = parseInt(metadata.hyperlink_count) || 0;
        const githubInfo = metadata.github_info || {};
        
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
        
        const padding = 20;
        const canvasWidth = Math.pow(2, Math.ceil(Math.log2(textWidth + padding * 2)));
        const canvasHeight = Math.pow(2, Math.ceil(Math.log2(this.labelFontSize * 3)));

        canvas.width = canvasWidth;
        canvas.height = canvasHeight;

        context.clearRect(0, 0, canvas.width, canvas.height);

        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        const cornerRadius = 8;
        this.roundRect(context, 0, 0, canvas.width, canvas.height, cornerRadius);

        context.font = `${this.labelFontSize}px Arial`;
        context.textBaseline = 'middle';
        context.textAlign = 'left';
        context.fillStyle = '#ffffff';
        context.fillText(text, padding, canvas.height * 0.35);
        
        context.font = `${this.labelFontSize * 0.6}px Arial`;
        context.fillStyle = '#cccccc';
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
        
        const labelScale = 0.8;
        sprite.scale.set(
            (canvas.width / this.labelFontSize) * labelScale,
            (canvas.height / this.labelFontSize) * labelScale,
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

    handleClick(event, isXR = false, intersectedObject = null) {
        let clickedMesh;

        if (isXR && intersectedObject) {
            clickedMesh = intersectedObject;
        } else if (!isXR && event) {
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
            const nodeId = Array.from(this.nodeMeshes.entries())
                .find(([_, mesh]) => mesh === clickedMesh)?.[0];

            if (nodeId) {
                const nodeData = this.nodeData.get(nodeId);
                if (nodeData) {
                    const url = this.formatNodeNameToUrl(nodeData.label || nodeId);
                    window.open(url, '_blank');

                    const originalEmissive = clickedMesh.material.emissiveIntensity;
                    clickedMesh.material.emissiveIntensity = 2.0;
                    setTimeout(() => {
                        clickedMesh.material.emissiveIntensity = originalEmissive;
                    }, 200);

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
        
        const centeredNodes = this.centerNodes(nodes);
        if (!centeredNodes) return;
        
        const existingNodeIds = new Set(centeredNodes.map(node => node.id));

        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                this.scene.remove(mesh);
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
                this.nodeMeshes.delete(nodeId);
                this.nodeData.delete(nodeId);

                const label = this.nodeLabels.get(nodeId);
                if (label) {
                    this.scene.remove(label);
                    if (label.material) {
                        if (label.material.map) label.material.map.dispose();
                        label.material.dispose();
                    }
                    this.nodeLabels.delete(nodeId);
                }
            }
        });

        centeredNodes.forEach(node => {
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node);
                return;
            }

            this.nodeData.set(node.id, node);

            const metadata = node.metadata || {};
            const size = this.getNodeSize(metadata);
            const color = this.calculateNodeColor(metadata);

            let mesh = this.nodeMeshes.get(node.id);

            if (!mesh) {
                const geometry = this.createNodeGeometry(size, metadata.hyperlink_count || 0);
                const material = this.createNodeMaterial(color, metadata);

                mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(node.x, node.y, node.z);
                mesh.layers.set(LAYERS.NORMAL_LAYER);
                
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                console.log(`Created node ${node.id} at position:`, {
                    x: node.x.toFixed(2),
                    y: node.y.toFixed(2),
                    z: node.z.toFixed(2),
                    size: size.toFixed(2)
                });

                setTimeout(() => {
                    const label = this.createNodeLabel(node.label || node.id, metadata);
                    if (label) {
                        label.position.set(node.x, node.y + size * 1.5, node.z);
                        label.layers.set(LAYERS.NORMAL_LAYER);
                        this.scene.add(label);
                        this.nodeLabels.set(node.id, label);
                    }
                }, 1000);
            } else {
                mesh.position.set(node.x, node.y, node.z);

                if (!mesh.material.color.equals(color)) {
                    mesh.material.color.copy(color);
                    mesh.material.emissive.copy(color);
                }

                const label = this.nodeLabels.get(node.id);
                if (label) {
                    label.position.set(node.x, node.y + size * 1.5, node.z);
                }
            }
        });

        console.log(`Updated ${centeredNodes.length} nodes`);
    }

    updateEdges(edges) {
        console.log(`Updating edges: ${edges.length}`);
        
        const edgeWeights = new Map();
        edges.forEach(edge => {
            if (!edge.source || !edge.target_node) {
                console.warn('Invalid edge data:', edge);
                return;
            }
            const edgeKey = `${edge.source}-${edge.target_node}`;
            const weight = edge.weight || 1;
            edgeWeights.set(edgeKey, weight);
        });

        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!edgeWeights.has(edgeKey)) {
                this.scene.remove(line);
                if (line.geometry) line.geometry.dispose();
                if (line.material) line.material.dispose();
                this.edgeMeshes.delete(edgeKey);
            }
        });

        edgeWeights.forEach((weight, edgeKey) => {
            const [source, target] = edgeKey.split('-');
            const sourceMesh = this.nodeMeshes.get(source);
            const targetMesh = this.nodeMeshes.get(target);

            if (sourceMesh && targetMesh) {
                let line = this.edgeMeshes.get(edgeKey);

                if (!line) {
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array(6);
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    const material = new THREE.LineBasicMaterial({
                        color: this.edgeColor,
                        transparent: true,
                        opacity: Math.min(1.0, this.edgeOpacity * 2),
                        linewidth: Math.max(1, Math.min(weight, 5))
                    });

                    line = new THREE.Line(geometry, material);
                    line.layers.set(LAYERS.NORMAL_LAYER);
                    this.scene.add(line);
                    this.edgeMeshes.set(edgeKey, line);
                }

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
        this.projScreenMatrix.multiplyMatrices(
            camera.projectionMatrix,
            camera.matrixWorldInverse
        );
        this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

        this.nodeLabels.forEach((label, nodeId) => {
            const mesh = this.nodeMeshes.get(nodeId);
            if (mesh) {
                const size = mesh.geometry.parameters.radius || 
                           mesh.geometry.parameters.width || 
                           this.minNodeSize;

                const labelSettings = visualizationSettings.getLabelSettings();
                const verticalOffset = labelSettings.verticalOffset || 0.8;
                const labelOffset = size * 3.0 * verticalOffset;
                
                label.position.set(
                    mesh.position.x,
                    mesh.position.y + labelOffset,
                    mesh.position.z
                );

                label.lookAt(camera.position);

                const distance = camera.position.distanceTo(mesh.position);
                const baseScale = 0.8;
                const distanceScale = Math.max(0.8, Math.min(2.0, distance * 0.04));
                const finalScale = baseScale * distanceScale;
                
                label.scale.set(finalScale, finalScale, 1);

                const inView = this.frustum.containsPoint(mesh.position);
                const notTooFar = distance < 100;
                label.visible = inView && notTooFar;

                if (label.material) {
                    const opacity = Math.max(0.2, 1 - (distance / 100));
                    label.material.opacity = opacity;
                }
            }
        });
    }

    updateFeature(control, value) {
        console.log(`Updating feature: ${control} = ${value}`);
        switch (control) {
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
                this.minNodeSize = value;
                break;
            case 'maxNodeSize':
                this.maxNodeSize = value;
                break;
            case 'labelFontSize':
                this.labelFontSize = value;
                break;
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
    }

    getNodePositions() {
        return Array.from(this.nodeMeshes.values()).map(mesh => [
            mesh.position.x,
            mesh.position.y,
            mesh.position.z
        ]);
    }

    dispose() {
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        
        if (this.renderer) {
            this.removeClickHandling(this.renderer);
        }

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

        this.nodeLabels.forEach(label => {
            if (label.material) {
                if (label.material.map) label.material.map.dispose();
                label.material.dispose();
            }
            if (label.parent) {
                label.parent.remove(label);
            }
        });

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

        this.nodeMeshes.clear();
        this.nodeLabels.clear();
        this.edgeMeshes.clear();
        this.nodeData.clear();
    }

    initInstancedMeshes() {
        console.log('Initializing instanced meshes');
        try {
            // Existing initialization code...
            
            console.log('Instanced meshes created:', {
                nodeCount: this.nodeInstanceCount,
                linkCount: this.linkInstanceCount,
                highDetailMesh: !!this.nodeInstancedMeshes.high,
                mediumDetailMesh: !!this.nodeInstancedMeshes.medium,
                lowDetailMesh: !!this.nodeInstancedMeshes.low,
                linkMesh: !!this.linkInstancedMesh
            });
        } catch (error) {
            console.error('Error initializing instanced meshes:', error);
            throw error;
        }
    }
}
