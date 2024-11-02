// Keep all existing imports...
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01;
const ROTATION_SPEED = 0.01;

/**
 * Class representing a WebXR visualization environment.
 */
export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        // Initialize the scene, camera, and renderer
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000); // Ensure black background
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.position.set(0, 0, 500);
        // Enable layer 1 for node labels (excluded from bloom)
        this.camera.layers.enable(1);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.toneMappingExposure = 1;
        this.renderer.setClearColor(0x000000); // Ensure black background

        // Rest of the constructor remains exactly the same...
        this.controls = null;
        this.composer = null;

        // Maps to hold node and edge meshes and their labels
        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();
        this.nodeLabels = new Map();

        this.hologramGroup = new THREE.Group();
        this.animationFrameId = null;

        this.selectedNode = null;

        // Add force-directed layout parameters
        this.forceDirectedIterations = 100;
        this.forceDirectedRepulsion = 1.0;
        this.forceDirectedAttraction = 0.01;

        // Call method to initialize settings
        this.initializeSettings();

        // Add event listener for graph data updates
        window.addEventListener('graphDataUpdated', (event) => {
            console.log('Received graphDataUpdated event:', event.detail);
            this.updateVisualization();
        });

        console.log('WebXRVisualization constructor completed');
    }

    initializeSettings() {
        console.log('Initializing settings');
        this.nodeColor = 0x1A0B31;
        this.edgeColor = 0xff0000;
        this.hologramColor = 0xFFD700;
        this.nodeSizeScalingFactor = 1;
        this.hologramScale = 1;
        this.hologramOpacity = 0.1;
        this.edgeOpacity = 0.3;
        this.labelFontSize = 48;
        this.fogDensity = 0.002;
        this.minNodeSize = 1;
        this.maxNodeSize = 5;
        this.bloomStrength = 1.5;
        this.bloomRadius = 0.4;
        this.bloomThreshold = 0.2;
        console.log('Settings initialized');
    }

    initThreeJS() {
        console.log('Initializing Three.js');
        const container = document.getElementById('scene-container');
        if (container) {
            container.appendChild(this.renderer.domElement);
        } else {
            console.error("Could not find 'scene-container' element");
            return;
        }

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        this.addLights();
        this.initPostProcessing();
        this.createHologramStructure();

        window.addEventListener('resize', this.onWindowResize.bind(this), false);

        this.animate();
        console.log('Three.js initialization completed');
    }

    initPostProcessing() {
        console.log('Initializing post-processing');
        this.composer = new EffectComposer(this.renderer);
        
        // Main render pass for everything
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        // Add bloom pass
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            this.bloomStrength,
            this.bloomRadius,
            this.bloomThreshold
        );
        this.composer.addPass(bloomPass);

        console.log('Post-processing initialized');
    }

    addLights() {
        console.log('Adding lights to the scene');
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);
        console.log('Lights added to the scene');
    }

    createHologramStructure() {
        console.log('Creating hologram structure');
        this.hologramGroup.clear();

        const buckyGeometry = new THREE.IcosahedronGeometry(40 * this.hologramScale, 1);
        const buckyMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const buckySphere = new THREE.Mesh(buckyGeometry, buckyMaterial);
        buckySphere.userData.rotationSpeed = 0.0001;
        buckySphere.layers.set(0);
        this.hologramGroup.add(buckySphere);

        const geodesicGeometry = new THREE.IcosahedronGeometry(10 * this.hologramScale, 1);
        const geodesicMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const geodesicDome = new THREE.Mesh(geodesicGeometry, geodesicMaterial);
        geodesicDome.userData.rotationSpeed = 0.0002;
        geodesicDome.layers.set(0);
        this.hologramGroup.add(geodesicDome);

        const triangleGeometry = new THREE.SphereGeometry(100 * this.hologramScale, 32, 32);
        const triangleMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const triangleSphere = new THREE.Mesh(triangleGeometry, triangleMaterial);
        triangleSphere.userData.rotationSpeed = 0.0003;
        triangleSphere.layers.set(0);
        this.hologramGroup.add(triangleSphere);

        this.scene.add(this.hologramGroup);
        console.log('Hologram structure created');
    }

    updateVisualization() {
        console.log('Updating visualization');
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData) {
            console.warn('No graph data available for visualization update');
            return;
        }
        console.log('Graph data received:', graphData);
        
        this.applyForceDirectedLayout(graphData);
        
        this.updateNodes(graphData.nodes);
        this.updateEdges(graphData.edges);
        console.log('Visualization update completed');
    }

    applyForceDirectedLayout(graphData) {
        console.log('Applying force-directed layout');
        const nodes = graphData.nodes;
        const edges = graphData.edges;

        for (let iteration = 0; iteration < this.forceDirectedIterations; iteration++) {
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[j].x - nodes[i].x;
                    const dy = nodes[j].y - nodes[i].y;
                    const dz = nodes[j].z - nodes[i].z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                    const force = this.forceDirectedRepulsion / (distance * distance);

                    nodes[i].x -= dx * force / distance;
                    nodes[i].y -= dy * force / distance;
                    nodes[i].z -= dz * force / distance;
                    nodes[j].x += dx * force / distance;
                    nodes[j].y += dy * force / distance;
                    nodes[j].z += dz * force / distance;
                }
            }

            for (const edge of edges) {
                const source = nodes.find(node => node.id === edge.source);
                const target = nodes.find(node => node.id === edge.target_node);
                if (source && target) {
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dz = target.z - source.z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                    const force = this.forceDirectedAttraction * distance;

                    source.x += dx * force / distance;
                    source.y += dy * force / distance;
                    source.z += dz * force / distance;
                    target.x -= dx * force / distance;
                    target.y -= dy * force / distance;
                    target.z -= dz * force / distance;
                }
            }
        }

        console.log('Force-directed layout applied');
    }

    updateVisualFeatures(changes) {
        console.log('Updating visual features:', changes);
        let needsUpdate = false;
        let layoutChanged = false;

        for (const [name, value] of Object.entries(changes)) {
            if (this.hasOwnProperty(name)) {
                console.log(`Setting property ${name} to`, value);
                this[name] = value;
                needsUpdate = true;

                if (name.includes('forceDirected')) {
                    layoutChanged = true;
                }
            } else {
                console.warn(`Property ${name} does not exist on WebXRVisualization`);
            }
        }

        if (needsUpdate) {
            if (layoutChanged) {
                this.updateVisualization();
            } else {
                this.updateNodes(this.graphDataManager.getGraphData().nodes);
                this.updateEdges(this.graphDataManager.getGraphData().edges);
            }
            
            if (changes.hologramScale !== undefined) {
                this.hologramGroup.scale.set(this.hologramScale, this.hologramScale, this.hologramScale);
            }
        }

        this.composer.render();
        console.log('Visual features update completed');
    }

    updateNodes(nodes) {
        console.log(`Updating nodes: ${nodes.length}`);
        const existingNodeIds = new Set(nodes.map(node => node.id));

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

        nodes.forEach(node => {
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node);
                return;
            }
            let mesh = this.nodeMeshes.get(node.id);
            const fileSize = node.metadata && node.metadata.file_size ? parseInt(node.metadata.file_size) : 1;
            if (isNaN(fileSize) || fileSize <= 0) {
                console.warn(`Invalid file_size for node ${node.id}:`, node.metadata.file_size);
                return;
            }
            const size = this.calculateNodeSize(fileSize);
            const color = this.calculateNodeColor(fileSize);

            console.log(`Node ${node.id}: fileSize = ${fileSize}, calculated size = ${size}`);

            if (!mesh) {
                const geometry = this.createNodeGeometry(size, fileSize);
                const material = new THREE.MeshStandardMaterial({ color: color });
                mesh = new THREE.Mesh(geometry, material);
                mesh.layers.set(0); // Set nodes to layer 0 for bloom
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                const label = this.createNodeLabel(node.label || node.id, fileSize);
                this.scene.add(label);
                this.nodeLabels.set(node.id, label);
            } else {
                this.updateNodeGeometry(mesh, size, fileSize);
                mesh.material.color.setHex(color);
            }

            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id);
            if (label) {
                label.position.set(node.x, node.y + size + 2, node.z);
                this.updateNodeLabel(label, node.label || node.id, fileSize);
            }
        });
    }

    calculateNodeSize(fileSize) {
        const logSize = Math.log(fileSize + 1) / Math.log(10);
        return Math.max(this.minNodeSize, Math.min(this.maxNodeSize, logSize * this.nodeSizeScalingFactor));
    }

    calculateNodeColor(lastModified) {
        const now = Date.now();
        const timeDifference = now - new Date(lastModified).getTime();
        const maxAge = 1000 * 60 * 60 * 24 * 30;
        const t = Math.min(timeDifference / maxAge, 1);
        const r = Math.floor(255 * (1 - t));
        const g = Math.floor(255 * t);
        const b = 100;
        return (r << 16) | (g << 8) | b;
    }

    createNodeGeometry(size, fileSize) {
        if (fileSize < 1000) {
            return new THREE.SphereGeometry(size, 16, 16);
        } else if (fileSize < 1000000) {
            return new THREE.BoxGeometry(size, size, size);
        } else {
            return new THREE.OctahedronGeometry(size);
        }
    }

    updateNodeGeometry(mesh, size, fileSize) {
        const newGeometry = this.createNodeGeometry(size, fileSize);
        mesh.geometry.dispose();
        mesh.geometry = newGeometry;
    }

    createNodeLabel(text, fileSize) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = `${this.labelFontSize}px Arial`;
        const metrics = context.measureText(text);
        const textWidth = metrics.width;

        canvas.width = textWidth + 20;
        canvas.height = this.labelFontSize + 30;

        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 10, this.labelFontSize);
        
        context.font = `${this.labelFontSize / 2}px Arial`;
        context.fillStyle = 'lightgray';
        context.fillText(this.formatFileSize(fileSize), 10, this.labelFontSize + 20);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);
        sprite.layers.set(1);

        spriteMaterial.depthWrite = false;
        spriteMaterial.transparent = true;

        return sprite;
    }

    updateNodeLabel(label, text, fileSize) {
        const canvas = label.material.map.image;
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 10, this.labelFontSize);
        
        context.font = `${this.labelFontSize / 2}px Arial`;
        context.fillStyle = 'lightgray';
        context.fillText(this.formatFileSize(fileSize), 10, this.labelFontSize + 20);

        label.material.map.needsUpdate = true;
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

    updateEdges(edges) {
        console.log(`Updating edges: ${edges.length}`);
        const existingEdgeKeys = new Set(edges.map(edge => `${edge.source}-${edge.target_node}`));

        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!existingEdgeKeys.has(edgeKey)) {
                this.scene.remove(line);
                this.edgeMeshes.delete(edgeKey);
            }
        });

        edges.forEach(edge => {
            if (!edge.source || !edge.target_node) {
                console.warn('Invalid edge data:', edge);
                return;
            }
            const edgeKey = `${edge.source}-${edge.target_node}`;
            let line = this.edgeMeshes.get(edgeKey);
            const sourceMesh = this.nodeMeshes.get(edge.source);
            const targetMesh = this.nodeMeshes.get(edge.target_node);
            
            if (!line) {
                if (sourceMesh && targetMesh) {
                    const geometry = new THREE.BufferGeometry().setFromPoints([
                        sourceMesh.position,
                        targetMesh.position
                    ]);
                    const material = new THREE.LineBasicMaterial({
                        color: this.edgeColor,
                        transparent: true,
                        opacity: this.edgeOpacity
                    });
                    line = new THREE.Line(geometry, material);
                    line.layers.set(0); // Set edges to layer 0 for bloom
                    this.scene.add(line);
                    this.edgeMeshes.set(edgeKey, line);
                } else {
                    console.warn(`Unable to create edge: ${edgeKey}. Source or target node not found.`);
                }
            } else if (sourceMesh && targetMesh) {
                const positions = line.geometry.attributes.position.array;
                positions[0] = sourceMesh.position.x;
                positions[1] = sourceMesh.position.y;
                positions[2] = sourceMesh.position.z;
                positions[3] = targetMesh.position.x;
                positions[4] = targetMesh.position.y;
                positions[5] = targetMesh.position.z;
                line.geometry.attributes.position.needsUpdate = true;
                line.material.color.setHex(this.edgeColor);
                line.material.opacity = this.edgeOpacity;
                line.material.needsUpdate = true;
            } else {
                console.warn(`Unable to update edge: ${edgeKey}. Source or target node not found.`);
            }
        });
    }

    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        this.controls.update();

        this.hologramGroup.children.forEach(child => {
            child.rotation.x += child.userData.rotationSpeed;
            child.rotation.y += child.userData.rotationSpeed;
        });

        this.nodeLabels.forEach(label => {
            label.lookAt(this.camera.position);
        });

        this.composer.render();
    }

    onWindowResize() {
        console.log('Window resized');
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.composer) {
            this.composer.setSize(window.innerWidth, window.innerHeight);
        }
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        this.scene.traverse(object => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        this.renderer.dispose();
        if (this.controls) {
            this.controls.dispose();
        }
        console.log('WebXRVisualization disposed');
    }

    updateNodeLabels() {
        console.log('Updating node labels');
        this.nodeLabels.forEach((label, nodeId) => {
            const node = this.nodeMeshes.get(nodeId);
            if (node) {
                this.scene.remove(label);
                const newLabel = this.createNodeLabel(label.userData.text);
                newLabel.position.copy(label.position);
                this.scene.add(newLabel);
                this.nodeLabels.set(nodeId, newLabel);
            }
        });
        console.log('Node labels update completed');
    }

    handleSpacemouseInput(x, y, z, rx, ry, rz) {
        if (!this.camera || !this.controls) {
            console.warn('Camera or controls not initialized for Spacemouse input');
            return;
        }

        this.camera.position.x += x * TRANSLATION_SPEED;
        this.camera.position.y += y * TRANSLATION_SPEED;
        this.camera.position.z += z * TRANSLATION_SPEED;
        
        this.camera.rotation.x += rx * ROTATION_SPEED;
        this.camera.rotation.y += ry * ROTATION_SPEED;
        this.camera.rotation.z += rz * ROTATION_SPEED;

        this.controls.update();
    }

    debugLabels() {
        console.log('Debugging labels');
        console.log('Total labels:', this.nodeLabels.size);

        this.camera.updateMatrixWorld();
        this.camera.updateProjectionMatrix();

        const frustum = new THREE.Frustum();
        const cameraViewProjectionMatrix = new THREE.Matrix4();
        cameraViewProjectionMatrix.multiplyMatrices(
            this.camera.projectionMatrix,
            this.camera.matrixWorldInverse
        );
        frustum.setFromProjectionMatrix(cameraViewProjectionMatrix);

        this.nodeLabels.forEach((label, nodeId) => {
            console.log(`Label for node ${nodeId}:`, {
                position: label.position.toArray(),
                visible: label.visible,
                inFrustum: frustum.containsPoint(label.position),
                material: {
                    color: label.material.color.getHex(),
                    opacity: label.material.opacity,
                    transparent: label.material.transparent,
                    visible: label.material.visible
                },
                geometry: {
                    type: label.geometry.type,
                    parameters: label.geometry.parameters
                }
            });
        });
    }
}
