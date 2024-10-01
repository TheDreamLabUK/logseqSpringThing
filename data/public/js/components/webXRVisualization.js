import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
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
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        // Three.js essentials
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null; // For post-processing

        // GPU acceleration (optional)
        this.gpu = null;
        this.isGPUEnabled = false;

        // Mesh pools for nodes, edges, and node labels
        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();
        this.nodeLabels = new Map(); // Billboard name tags for nodes

        // Holographic elements
        this.hologramGroup = new THREE.Group();
        this.particleSystem = null;
    }

    /**
     * Initializes the Three.js scene, camera, renderer, controls, GPU setup, and post-processing.
     */
    initialize() {
        console.log('WebXRVisualization initialize method called');
        try {
            this.initThreeJS();
            this.createHologramStructure();
            window.addEventListener('resize', this.onWindowResize.bind(this), false);
            this.animate();
            this.updateVisualization(); // Call updateVisualization after initialization
        } catch (error) {
            console.error('Error in WebXRVisualization initialize method:', error);
        }
    }

    /**
     * Initializes the Three.js scene, camera, renderer, controls, and post-processing.
     */
    initThreeJS() {
        console.log('Initializing Three.js components');
        // Create the scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000000, 0.002); // Add fog for depth effect

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
        document.getElementById('scene-container').appendChild(this.renderer.domElement);

        // Add orbit controls for camera manipulation
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add lighting to the scene
        this.addLights();

        // Initialize post-processing
        this.initPostProcessing();

        console.log('Three.js components initialized');
    }

    /**
     * Initializes post-processing effects.
     */
    initPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5, // strength
            0.4, // radius
            0.85 // threshold
        );
        this.composer.addPass(bloomPass);
    }

    /**
     * Adds lights to the scene.
     */
    addLights() {
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5); // Soft light
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(50, 50, 50); // Directional light
        this.scene.add(directionalLight);
    }

    /**
     * Updates the visualization with new graph data.
     */
    updateVisualization() {
        console.log('Updating visualization');
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData) {
            console.warn('No graph data available for visualization update');
            return;
        }

        this.updateNodes(graphData.nodes);
        this.updateEdges(graphData.edges);
    }

    /**
     * Updates nodes in the scene and adds billboard name tags.
     * @param {Array} nodes - Array of node objects.
     */
    updateNodes(nodes) {
        console.log('Updating nodes:', nodes.length);
        const existingNodeIds = new Set(nodes.map((node) => node.id));

        // Remove nodes that no longer exist
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                this.scene.remove(mesh);
                this.nodeMeshes.delete(nodeId);

                // Remove node label
                const label = this.nodeLabels.get(nodeId);
                if (label) {
                    this.scene.remove(label);
                    this.nodeLabels.delete(nodeId);
                }
            }
        });

        // Add or update nodes and their labels
        nodes.forEach((node) => {
            if (this.nodeMeshes.has(node.id)) {
                const mesh = this.nodeMeshes.get(node.id);
                mesh.position.set(node.x, node.y, node.z);

                // Update label position
                const label = this.nodeLabels.get(node.id);
                if (label) {
                    label.position.set(node.x, node.y + 5, node.z); // Slightly above the node
                }
            } else {
                // Create a new node mesh
                const geometry = new THREE.SphereGeometry(2, 16, 16);
                const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(node.x, node.y, node.z);
                mesh.userData = { id: node.id, name: node.label };

                // Create a name label (billboard text)
                const label = this.createNodeLabel(node.label);
                label.position.set(node.x, node.y + 5, node.z); // Slightly above the node

                this.scene.add(mesh);
                this.scene.add(label);

                this.nodeMeshes.set(node.id, mesh);
                this.nodeLabels.set(node.id, label);
            }
        });
    }

    /**
     * Creates a billboard label for a node.
     * @param {string} text - The text to display.
     * @returns {THREE.Sprite} - The label sprite.
     */
    createNodeLabel(text) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = '24px Arial';
        const metrics = context.measureText(text);
        const textWidth = metrics.width;

        canvas.width = textWidth + 10; // Add padding
        canvas.height = 40; // Text height

        context.fillStyle = 'rgba(0, 0, 0, 0.5)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'white';
        context.textBaseline = 'middle';
        context.fillText(text, 5, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(canvas.width / 50, canvas.height / 50, 1); // Scale appropriately

        return sprite;
    }

    /**
     * Updates edges in the scene.
     * @param {Array} edges - Array of edge objects.
     */
    updateEdges(edges) {
        console.log('Updating edges:', edges.length);
        const existingEdgeKeys = new Set(edges.map((edge) => `${edge.source}-${edge.target_node}`));

        // Remove edges that no longer exist
        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!existingEdgeKeys.has(edgeKey)) {
                this.scene.remove(line);
                this.edgeMeshes.delete(edgeKey);
            }
        });

        // Add or update edges
        edges.forEach((edge) => {
            const edgeKey = `${edge.source}-${edge.target_node}`;
            if (this.edgeMeshes.has(edgeKey)) {
                const line = this.edgeMeshes.get(edgeKey);
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
                }
            } else {
                // Create a new edge line
                const sourceMesh = this.nodeMeshes.get(edge.source);
                const targetMesh = this.nodeMeshes.get(edge.target_node);
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
     * Creates the holographic structure and particle system.
     */
    createHologramStructure() {
        console.log('Creating hologram structure');
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

        this.scene.add(this.hologramGroup);

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
     * Animation loop for continuous rendering.
     */
    animate() {
        const animateLoop = () => {
            requestAnimationFrame(animateLoop);
            this.controls.update();

            // Rotate holographic shells
            this.rotateHologram();

            // Update particle system
            this.updateParticles();

            // Use composer for post-processing if initialized
            if (this.composer) {
                this.composer.render();
            } else {
                this.renderer.render(this.scene, this.camera);
            }
        };

        animateLoop();
    }

    /**
     * Rotates the hologram.
     */
    rotateHologram() {
        this.hologramGroup.children.forEach((shell) => {
            shell.rotation.y += shell.rotationSpeed;
        });
    }

    /**
     * Updates particle positions.
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
     * Handles window resize events.
     */
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.composer) {
            this.composer.setSize(window.innerWidth, window.innerHeight);
        }
    }
}
