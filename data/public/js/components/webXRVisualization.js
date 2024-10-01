import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

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

        // Mesh pools for nodes and edges
        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();
    }

    /**
     * Initializes the Three.js scene, camera, renderer, controls.
     */
    initialize() {
        console.log('WebXRVisualization initialize method called');
        try {
            this.initThreeJS();
            window.addEventListener('resize', this.onWindowResize.bind(this), false);
            this.animate();
            this.updateVisualization(); // Call updateVisualization after initialization
        } catch (error) {
            console.error('Error in WebXRVisualization initialize method:', error);
        }
    }

    /**
     * Initializes the Three.js scene, camera, renderer, controls.
     */
    initThreeJS() {
        console.log('Initializing Three.js components');
        // Create the scene
        this.scene = new THREE.Scene();

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
        document.getElementById('scene-container').appendChild(this.renderer.domElement);

        // Add orbit controls for camera manipulation
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add basic lighting to the scene
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
        this.scene.add(ambientLight);
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
     * Updates nodes in the scene.
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
                const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(node.x, node.y, node.z);
                mesh.userData = { id: node.id, name: node.label };

                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);
            }
        });
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

                    const material = new THREE.LineBasicMaterial({ color: 0xffffff });
                    const line = new THREE.Line(geometry, material);
                    this.scene.add(line);
                    this.edgeMeshes.set(edgeKey, line);
                }
            }
        });
    }

    /**
     * Animation loop for continuous rendering.
     */
    animate() {
        const animateLoop = () => {
            requestAnimationFrame(animateLoop);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };

        animateLoop();
    }

    /**
     * Handles window resize events.
     */
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}