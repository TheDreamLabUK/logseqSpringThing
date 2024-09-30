// public/js/components/webXRVisualization.js

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { initXRSession, handleXRSession } from '../xr/xrSetup.js';
import { initXRInteraction } from '../xr/xrInteraction.js';
import { isGPUAvailable, initGPU, computeOnGPU } from '../gpuUtils.js';

// Import post-processing modules
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

/**
 * WebXRVisualization class manages the 3D graph visualization with WebXR support.
 */
export class WebXRVisualization {
    /**
     * Creates a new WebXRVisualization instance.
     * @param {GraphDataManager} graphDataManager - The GraphDataManager instance.
     */
    constructor(graphDataManager) {
        this.graphDataManager = graphDataManager;

        // Three.js essentials
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null; // For post-processing

        // GPU acceleration
        this.gpu = null;
        this.isGPUEnabled = false;

        // Mesh pools for nodes and edges
        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();

        // Holographic elements
        this.hologramGroup = new THREE.Group(); // Group for holographic shells and effects
        this.particleSystem = null;
    }

    /**
     * Initializes the Three.js scene, camera, renderer, controls, and XR sessions.
     */
    initialize() {
        this.initThreeJS();
        this.setupGPU();
        initXRSession(this.renderer, this.scene, this.camera);
        initXRInteraction(this.scene, this.camera, this.renderer, this.onSelect.bind(this));
        window.addEventListener('resize', this.onWindowResize.bind(this), false);
        this.animate();
    }

    /**
     * Sets up GPU acceleration if available.
     */
    setupGPU() {
        if (isGPUAvailable()) {
            this.gpu = initGPU();
            this.isGPUEnabled = true;
            console.log('GPU acceleration enabled.');
        } else {
            console.warn('GPU acceleration not available. Falling back to CPU rendering.');
            this.isGPUEnabled = false;
        }
    }

    /**
     * Initializes the Three.js scene, camera, renderer, controls, and holographic elements.
     */
    initThreeJS() {
        // Create the scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000000, 0.002);

        // Create the camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        this.camera.position.set(0, 0, 100);

        // Create the renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.xr.enabled = true; // Enable WebXR
        document.getElementById('scene-container').appendChild(this.renderer.domElement);

        // Add orbit controls for camera manipulation
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Set up post-processing
        this.initPostProcessing();

        // Add lights to the scene
        this.addLights();

        // Create holographic elements
        this.createHologramStructure();

        // Add hologram group to the scene
        this.scene.add(this.hologramGroup);
    }

    /**
     * Initializes post-processing effects like bloom.
     */
    initPostProcessing() {
        // Create composer for post-processing
        this.composer = new EffectComposer(this.renderer);

        // Add render pass
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        // Add bloom pass
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5, // strength
            0.4, // radius
            0.85 // threshold
        );
        this.composer.addPass(bloomPass);
    }

    /**
     * Adds ambient and directional lights to the scene.
     */
    addLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
        this.scene.add(ambientLight);

        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);
    }

    /**
     * Creates the holographic spherical structure with layers, wireframes, and particle systems.
     */
    createHologramStructure() {
        const numShells = 5; // Number of concentric shells
        const shellSpacing = 20; // Spacing between shells
        const shellMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide,
        });

        // Create concentric holographic shells
        for (let i = 0; i < numShells; i++) {
            const radius = 50 + i * shellSpacing;
            const geometry = new THREE.SphereGeometry(radius, 64, 64);
            const shell = new THREE.Mesh(geometry, shellMaterial.clone());
            shell.material.opacity = 0.2 - i * 0.03; // Decrease opacity for inner shells
            shell.rotationSpeed = 0.001 + i * 0.0005; // Vary rotation speed
            this.hologramGroup.add(shell);
        }

        // Add wireframes to shells
        this.hologramGroup.children.forEach((shell) => {
            const wireframeGeometry = new THREE.EdgesGeometry(shell.geometry);
            const wireframeMaterial = new THREE.LineBasicMaterial({
                color: 0x00ffff,
                transparent: true,
                opacity: 0.2,
            });
            const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
            shell.add(wireframe);
        });

        // Create dynamic particle system
        this.createParticleSystem();
    }

    /**
     * Creates a particle system to simulate energy flow within the hologram.
     */
    createParticleSystem() {
        const particleCount = 1000;
        const particles = new THREE.BufferGeometry();
        const positions = [];
        const velocities = [];

        // Initialize particle positions and velocities
        for (let i = 0; i < particleCount; i++) {
            const radius = Math.random() * 100;
            const theta = Math.random() * 2 * Math.PI;
            const phi = Math.random() * Math.PI;

            const x = radius * Math.sin(phi) * Math.cos(theta);
            const y = radius * Math.sin(phi) * Math.sin(theta);
            const z = radius * Math.cos(phi);

            positions.push(x, y, z);
            velocities.push(Math.random() * 0.02 - 0.01, Math.random() * 0.02 - 0.01, Math.random() * 0.02 - 0.01);
        }

        particles.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        particles.setAttribute('velocity', new THREE.Float32BufferAttribute(velocities, 3));

        // Particle material
        const particleMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 1,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.7,
            depthWrite: false,
        });

        // Create particle system
        this.particleSystem = new THREE.Points(particles, particleMaterial);
        this.hologramGroup.add(this.particleSystem);
    }

    /**
     * Handles node selection via controllers.
     * @param {THREE.Object3D} selectedObject - The selected Three.js object.
     */
    onSelect(selectedObject) {
        if (selectedObject && selectedObject.userData && selectedObject.userData.id) {
            console.log(`Selected Node: ${selectedObject.userData.name}`);
            // Display node information
            const nodeInfo = {
                id: selectedObject.userData.id,
                name: selectedObject.userData.name,
                // Add more properties as needed
            };
            // Emit a custom event to update the UI
            const event = new CustomEvent('nodeSelected', { detail: nodeInfo });
            window.dispatchEvent(event);
        }
    }

    /**
     * Updates the visualization with new graph data.
     */
    updateVisualization() {
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData) return;

        // Update nodes and edges
        this.updateNodes(graphData.nodes);
        this.updateEdges(graphData.edges);
    }

    /**
     * Updates nodes in the scene based on graph data.
     * @param {Array} nodes - Array of node objects.
     */
    updateNodes(nodes) {
        const existingNodeIds = new Set(nodes.map((node) => node.id));

        // Remove nodes that no longer exist
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                this.scene.remove(mesh);
                this.nodeMeshes.delete(nodeId);
            }
        });

        // Add or update nodes
        nodes.forEach((node) => {
            if (this.nodeMeshes.has(node.id)) {
                const mesh = this.nodeMeshes.get(node.id);
                mesh.position.set(node.x, node.y, node.z);
            } else {
                // Create a new node mesh
                const geometry = new THREE.SphereGeometry(2, 16, 16);
                const material = new THREE.MeshStandardMaterial({ color: this.getNodeColor(node) });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(node.x, node.y, node.z);
                mesh.userData = { id: node.id, name: node.label };

                // Add interaction event
                mesh.callback = () => this.onSelect(mesh);

                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);
            }
        });
    }

    /**
     * Updates edges in the scene based on graph data.
     * @param {Array} edges - Array of edge objects.
     */
    updateEdges(edges) {
        const existingEdgeKeys = new Set(edges.map((edge) => `${edge.source}-${edge.target}`));

        // Remove edges that no longer exist
        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!existingEdgeKeys.has(edgeKey)) {
                this.scene.remove(line);
                this.edgeMeshes.delete(edgeKey);
            }
        });

        // Add or update edges
        edges.forEach((edge) => {
            const edgeKey = `${edge.source}-${edge.target}`;
            if (this.edgeMeshes.has(edgeKey)) {
                const line = this.edgeMeshes.get(edgeKey);
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target);
                if (sourceMesh && targetMesh) {
                    const positions = line.geometry.attributes.position.array;
                    positions[0] = sourceMesh.position.x;
                    positions[1] = sourceMesh.position.y;
                    positions[2] = sourceMesh.position.z;
                    positions[3] = targetMesh.position.x;
                    positions[4] = targetMesh.position.y;
                    positions[5] = targetMesh.position.z;
                    line.geometry.attributes.position.needsUpdate = true;
                }
            } else {
                // Create a new edge line
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target);
                if (sourceMesh && targetMesh) {
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array(6); // 2 points * 3 coordinates
                    positions[0] = sourceMesh.position.x;
                    positions[1] = sourceMesh.position.y;
                    positions[2] = sourceMesh.position.z;
                    positions[3] = targetMesh.position.x;
                    positions[4] = targetMesh.position.y;
                    positions[5] = targetMesh.position.z;
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    const material = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.5, transparent: true });
                    const line = new THREE.Line(geometry, material);
                    this.scene.add(line);
                    this.edgeMeshes.set(edgeKey, line);
                }
            }
        });
    }

    /**
     * Determines the color of a node based on its properties.
     * @param {object} node - The node object.
     * @returns {THREE.Color} - The color of the node.
     */
    getNodeColor(node) {
        // Example: Color nodes based on a 'type' property
        if (node.type === 'core') {
            return new THREE.Color(0xffa500); // Orange for core nodes
        } else if (node.type === 'secondary') {
            return new THREE.Color(0x00ffff); // Cyan for secondary nodes
        } else {
            return new THREE.Color(0x00ff00); // Green for default nodes
        }
    }

    /**
     * Continuously renders the scene and updates label orientations.
     */
    animate() {
        const animateLoop = () => {
            requestAnimationFrame(animateLoop);

            // Update controls
            this.controls.update();

            // Rotate hologram shells
            this.rotateHologram();

            // Update particle system
            this.updateParticles();

            // GPU acceleration (if enabled)
            if (this.isGPUEnabled && this.gpu) {
                computeOnGPU(this.gpu, { /* Pass necessary data */ });
            }

            // Render the scene with post-processing
            this.composer.render();
        };

        animateLoop();
    }

    /**
     * Rotates the hologram shells at different speeds.
     */
    rotateHologram() {
        this.hologramGroup.children.forEach((shell) => {
            shell.rotation.y += shell.rotationSpeed;
        });
    }

    /**
     * Updates the particle system positions to simulate energy flow.
     */
    updateParticles() {
        const positions = this.particleSystem.geometry.attributes.position.array;
        const velocities = this.particleSystem.geometry.attributes.velocity.array;

        for (let i = 0; i < positions.length; i += 3) {
            positions[i] += velocities[i] * 0.1;
            positions[i + 1] += velocities[i + 1] * 0.1;
            positions[i + 2] += velocities[i + 2] * 0.1;

            // Reset particle if it goes beyond a certain radius
            const distance = Math.sqrt(
                positions[i] * positions[i] +
                positions[i + 1] * positions[i + 1] +
                positions[i + 2] * positions[i + 2]
            );
            if (distance > 150) {
                positions[i] = positions[i] * -1;
                positions[i + 1] = positions[i + 1] * -1;
                positions[i + 2] = positions[i + 2] * -1;
            }
        }

        this.particleSystem.geometry.attributes.position.needsUpdate = true;
    }

    /**
     * Handles window resize events by updating camera and renderer dimensions.
     */
    onWindowResize() {
        // Update camera
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        // Update renderer
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        // Update composer
        this.composer.setSize(window.innerWidth, window.innerHeight);
    }
}
