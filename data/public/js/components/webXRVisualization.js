import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import toml from 'toml';

// Function to load and parse settings.toml
async function loadSettings() {
    const response = await fetch('/settings.toml');
    const text = await response.text();
    return toml.parse(text);
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

        const fileSizes = nodes.map(node => parseInt(node.metadata.file_size) || 0);
        const minFileSize = Math.min(...fileSizes);
        const maxFileSize = Math.max(...fileSizes);

        nodes.forEach(node => {
            let mesh = this.nodeMeshes.get(node.id);
            const fileSize = parseInt(node.metadata.file_size) || 0;
            
            // Calculate normalized size using an exponential function with reduced scaling
            const normalizedSize = (Math.exp(fileSize / maxFileSize) - 1) / (Math.E - 1);
            const size = this.minNodeSize + (normalizedSize * (this.maxNodeSize - this.minNodeSize)) / 3;

            if (!mesh) {
                const geometry = new THREE.SphereGeometry(size, 32, 32);
                const material = new THREE.MeshStandardMaterial({ color: this.nodeColor });
                mesh = new THREE.Mesh(geometry, material);
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                const label = this.createNodeLabel(node.label);
                this.scene.add(label);
                this.nodeLabels.set(node.id, label);
            } else {
                // Update existing mesh size
                mesh.scale.setScalar(size / this.minNodeSize);
            }

            // Update position and label
            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id);
            label.position.set(node.x, node.y + size + 2, node.z); // Adjust label position based on node size
        });
    }

    createNodeLabel(text) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = `${this.labelFontSize}px Arial`;
        const metrics = context.measureText(text);
        const textWidth = metrics.width;

        canvas.width = textWidth + 10;
        canvas.height = 60; // Adjusted height for larger text

        context.fillStyle = 'rgba(0, 0, 0, 0.5)'; // Inverted color
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'white'; // Inverted text color
        context.textBaseline = 'middle';
        context.fillText(text, 5, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(canvas.width / 50, canvas.height / 50, 1);

        return sprite;
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
            const edgeKey = `${edge.source}-${edge.target_node}`;
            let line = this.edgeMeshes.get(edgeKey);
            if (!line) {
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target_node);
                if (sourceMesh && targetMesh) {
                    const positions = new Float32Array([
                        sourceMesh.position.x, sourceMesh.position.y, sourceMesh.position.z,
                        targetMesh.position.x, targetMesh.position.y, targetMesh.position.z
                    ]);
                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    const material = new THREE.LineBasicMaterial({ color: this.edgeColor, opacity: this.edgeOpacity, transparent: true });
                    line = new THREE.Line(geometry, material);
                    this.scene.add(line);
                    this.edgeMeshes.set(edgeKey, line);
                } else {
                    console.warn(`Unable to create edge ${edgeKey}: Source or target node not found`);
                }
            } else {
                const positions = line.geometry.attributes.position.array;
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target_node);
                if (sourceMesh && targetMesh) {
                    positions[0] = sourceMesh.position.x;
                    positions[1] = sourceMesh.position.y;
                    positions[2] = sourceMesh.position.z;
                    positions[3] = targetMesh.position.x;
                    positions[4] = targetMesh.position.y;
                    positions[5] = targetMesh.position.z;
                    line.geometry.attributes.position.needsUpdate = true;
                } else {
                    console.warn(`Unable to update edge ${edgeKey}: Source or target node not found`);
                }
            }
        });
    }

    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        
        // Rotate hologram spheres
        this.hologramGroup.children.forEach(child => {
            child.rotation.x += child.userData.rotationSpeed;
            child.rotation.y += child.userData.rotationSpeed;
        });

        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }

    stopAnimation() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.composer) {
            this.composer.setSize(window.innerWidth, window.innerHeight);
        }
    }
}
