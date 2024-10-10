import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import toml from 'toml';
import net from 'net'; // Node.js module for network communication

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01;
import WebSocketService from '../services/websocketService.js'; // Updated import to use default import

// Function to load and parse settings.toml
async function loadSettings() {
    const response = await fetch('/settings.toml');
    const text = await response.text();
    return toml.parse(text);
}

/**
 * Connects to the Spacemouse via spacenavd and listens for movement events
 * @param {THREE.Camera} camera - The Three.js camera to control
 * @param {THREE.OrbitControls} controls - The OrbitControls instance to update
 */
function connectSpacemouse(camera, controls) {
    const socketPath = '/var/run/spnav.sock'; // Path to spacenavd socket

    // Create a connection to spacenavd
    const client = net.createConnection(socketPath, () => {
        console.log('Connected to Spacenavd');
    });

    client.on('data', (data) => {
        // Spacenavd sends movement data in 12-byte packets
        if (data.length !== 12) {
            console.warn('Unexpected data length from Spacenavd');
            return;
        }

        // Read movement data: translation (6 bytes) + rotation (6 bytes)
        const tx = data.readInt16LE(0) * TRANSLATION_SPEED;
        const ty = data.readInt16LE(2) * TRANSLATION_SPEED;
        const tz = data.readInt16LE(4) * TRANSLATION_SPEED;
        const rx = data.readInt16LE(6) * ROTATION_SPEED;
        const ry = data.readInt16LE(8) * ROTATION_SPEED;
        const rz = data.readInt16LE(10) * ROTATION_SPEED;

        // Update camera position based on translation data
        camera.position.x += tx;
        camera.position.y -= ty; // Inverted to match expected behavior
        camera.position.z -= tz;

        // Update camera rotation based on rotation data
        camera.rotation.x += rx;
        camera.rotation.y += ry;
        camera.rotation.z += rz;

        // Update controls target for proper interaction
        controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion));

        // Update controls
        controls.update();
    });

    client.on('error', (err) => {
        console.error('Error connecting to Spacenavd:', err);
    });

    client.on('close', () => {
        console.log('Disconnected from Spacenavd');
    });
}

export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.position.set(0, 0, 500);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.toneMappingExposure = 1;
        
        const container = document.getElementById('scene-container');
        if (container) {
            container.appendChild(this.renderer.domElement);
        } else {
            console.error("Could not find 'scene-container' element");
        }

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        this.composer = null;

        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();
        this.nodeLabels = new Map();

        this.hologramGroup = new THREE.Group();
        this.animationFrameId = null;

        this.selectedNode = null;

        this.initialize();
    }

    async initialize() {
        try {
            const settings = await loadSettings();
            this.applySettings(settings);
        } catch (error) {
            console.error('Error loading settings:', error);
            this.applyDefaultSettings();
        }
        this.initThreeJS();
        this.createHologramStructure();
        window.addEventListener('resize', this.onWindowResize.bind(this), false);
        this.animate();
        this.updateVisualization();
    }

    applySettings(settings) {
        this.nodeColor = parseInt(settings.visualization.node_color, 16) || 0x1A0B31;
        this.edgeColor = parseInt(settings.visualization.edge_color, 16) || 0xff0000;
        this.hologramColor = parseInt(settings.visualization.hologram_color, 16) || 0xFFD700;
        this.nodeSizeScalingFactor = settings.visualization.node_size_scaling_factor || 1000;
        this.hologramScale = settings.visualization.hologram_scale || 1;
        this.hologramOpacity = (settings.visualization.hologram_opacity || 0.1) * 1;
        this.edgeOpacity = settings.visualization.edge_opacity || 0.3;
        this.labelFontSize = settings.visualization.label_font_size || 48;
        this.fogDensity = settings.visualization.fog_density || 0.002;
        this.minNodeSize = settings.visualization.min_node_size || 1;
        this.maxNodeSize = settings.visualization.max_node_size || 10;
    }

    applyDefaultSettings() {
        this.nodeColor = 0x1A0B31;
        this.edgeColor = 0xff0000;
        this.hologramColor = 0xFFD700;
        this.nodeSizeScalingFactor = 1000;
        this.hologramScale = 1;
        this.hologramOpacity = 0.1;
        this.edgeOpacity = 0.3;
        this.labelFontSize = 48;
        this.fogDensity = 0.002;
        this.minNodeSize = 1;
        this.maxNodeSize = 10;
    }

    initThreeJS() {
        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        this.addLights();
        this.initPostProcessing();
    }

    initPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            2.0,  // Strength (increased for heavier bloom)
            0.5,  // Radius
            0.1   // Threshold (lowered to make bloom more visible)
        );
        bloomPass.threshold = 0;
        bloomPass.strength = 3.0;  // Increased bloom strength
        bloomPass.radius = 1;
        this.composer.addPass(bloomPass);
    }

    addLights() {
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);
    }

    createHologramStructure() {
        // Create a Buckminster Fullerene
        const buckyGeometry = new THREE.IcosahedronGeometry(40 * this.hologramScale, 1);
        const buckyMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const buckySphere = new THREE.Mesh(buckyGeometry, buckyMaterial);
        buckySphere.userData.rotationSpeed = 0.0001;
        buckySphere.layers.enable(1);  // Enable bloom for this sphere
        this.hologramGroup.add(buckySphere);

        // Create a Geodesic Dome
        const geodesicGeometry = new THREE.IcosahedronGeometry(30 * this.hologramScale, 1);
        const geodesicMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const geodesicDome = new THREE.Mesh(geodesicGeometry, geodesicMaterial);
        geodesicDome.userData.rotationSpeed = 0.0002;
        geodesicDome.layers.enable(1);  // Enable bloom for this sphere
        this.hologramGroup.add(geodesicDome);

        // Create a Normal Triangle Sphere
        const triangleGeometry = new THREE.SphereGeometry(20 * this.hologramScale, 32, 32);
        const triangleMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const triangleSphere = new THREE.Mesh(triangleGeometry, triangleMaterial);
        triangleSphere.userData.rotationSpeed = 0.0003;
        triangleSphere.layers.enable(1);  // Enable bloom for this sphere
        this.hologramGroup.add(triangleSphere);

        this.scene.add(this.hologramGroup);
    }

    updateVisualization() {
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData) {
            console.warn('No graph data available for visualization update');
            return;
        }
        this.updateNodes(graphData.nodes);
        this.updateEdges(graphData.edges);
    }

    updateNodes(nodes) {
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
            const size = Math.max(this.minNodeSize, Math.min(this.maxNodeSize, Math.sqrt(fileSize) / this.nodeSizeScalingFactor));

            if (!mesh) {
                const geometry = new THREE.SphereGeometry(size, 32, 32);
                const material = new THREE.MeshStandardMaterial({ color: this.nodeColor });
                mesh = new THREE.Mesh(geometry, material);
                mesh.layers.enable(1); // Enable bloom for nodes
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                const label = this.createNodeLabel(node.label || node.id);
                this.scene.add(label);
                this.nodeLabels.set(node.id, label);
            } else {
                mesh.scale.setScalar(size);
            }

            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id);
            label.position.set(node.x, node.y + size + 2, node.z);
        });
    }

    updateEdges(edges) {
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
            if (!line) {
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target_node);
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
                    line.layers.enable(1); // Enable bloom for edges
                    this.scene.add(line);
                    this.edgeMeshes.set(edgeKey, line);
                } else {
                    console.warn(`Unable to create edge: ${edgeKey}. Source or target node not found.`);
                }
            } else {
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target_node);
                if (sourceMesh && targetMesh) {
                    const positions = line.geometry.attributes.position.array;
                    positions[0] = sourceMesh.position.x;
                    positions[1] = sourceMesh.position.y;
                    positions[2] = sourceMesh.position.z;
                    positions[3] = targetMesh.position.x;
                    positions[4] = targetMesh.position.y;
                    positions[5] = targetMesh.position.z;
                    line.geometry.attributes.position.needsUpdate = true;
                } else {
                    console.warn(`Unable to update edge: ${edgeKey}. Source or target node not found.`);
                }
            }
        });
    }

    createNodeLabel(text) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = `${this.labelFontSize}px Arial`;
        const metrics = context.measureText(text);
        const textWidth = metrics.width;

        canvas.width = textWidth + 10;
        canvas.height = this.labelFontSize + 10;

        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 5, this.labelFontSize);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);
        sprite.layers.set(1); // Enable bloom for labels

        return sprite;
    }

    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        this.controls.update();

        this.hologramGroup.children.forEach(child => {
            child.rotation.x += child.userData.rotationSpeed;
            child.rotation.y += child.userData.rotationSpeed;
        });

        this.composer.render();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.composer) {
            this.composer.setSize(window.innerWidth, window.innerHeight);
        }
    }

    dispose() {
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
        this.composer.dispose();
    }
}