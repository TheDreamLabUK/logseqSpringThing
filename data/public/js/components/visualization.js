import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Visualization class handles the creation and rendering of the 3D graph using Three.js.
 */
export class Visualization {
  constructor(graphDataManager) {
    this.graphDataManager = graphDataManager;
    // Store references to node and edge meshes for easy updates
    this.nodeMeshes = new Map();
    this.edgeMeshes = new Map();
  }

  /**
   * Initializes the visualization.
   */
  initialize() {
    // Initialize Three.js components
    this.initThreeJS();
    
    // Set up animation loop
    this.animate();

    console.log("Visualization initialized");
  }

  /**
   * Initializes Three.js scene, camera, renderer, and controls.
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
      1000
    );
    this.camera.position.set(0, 0, 100);

    // Create the renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.xr.enabled = true; // Enable WebXR
    document.body.appendChild(this.renderer.domElement);

    // Add orbit controls for camera manipulation
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Add ambient light to the scene
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    // Add directional light to the scene
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(50, 50, 50);
    this.scene.add(directionalLight);

    // Handle window resize events
    window.addEventListener('resize', this.onWindowResize.bind(this), false);
  }

  /**
   * Creates the initial graph visualization based on fetched graph data.
   * @param {object} graphData - The graph data containing nodes and edges.
   */
  createGraph(graphData) {
    this.createNodeObjects(graphData.nodes);
    this.createEdgeObjects(graphData.edges);
  }

  /**
   * Creates Three.js Mesh objects for each node and adds them to the scene.
   * @param {Array} nodes - Array of node objects.
   */
  createNodeObjects(nodes) {
    // Define geometry and material for nodes
    const geometry = new THREE.SphereGeometry(1.5, 16, 16);
    const material = new THREE.MeshStandardMaterial({ color: 0xff0000 });

    nodes.forEach(node => {
      // Create a mesh for each node
      const mesh = new THREE.Mesh(geometry, material.clone());
      mesh.position.set(node.x, node.y, node.z);
      mesh.userData = { id: node.id, name: node.name };

      // Add the mesh to the scene
      this.scene.add(mesh);

      // Store the mesh in the nodeMeshes map for easy access
      this.nodeMeshes.set(node.id, mesh);
    });
  }

  /**
   * Creates Three.js Line objects for each edge and adds them to the scene.
   * @param {Array} edges - Array of edge objects.
   */
  createEdgeObjects(edges) {
    edges.forEach(edge => {
      const sourceNode = this.nodeMeshes.get(edge.source);
      const targetNode = this.nodeMeshes.get(edge.target_node);

      if (sourceNode && targetNode) {
        // Define geometry for the edge
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array([
          sourceNode.position.x,
          sourceNode.position.y,
          sourceNode.position.z,
          targetNode.position.x,
          targetNode.position.y,
          targetNode.position.z,
        ]);
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        // Define material for the edge
        const material = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.5, transparent: true });

        // Create the line object
        const line = new THREE.Line(geometry, material);

        // Add the line to the scene
        this.scene.add(line);

        // Store the line in the edgeMeshes map for easy access
        this.edgeMeshes.set(`${edge.source}-${edge.target_node}`, line);
      }
    });
  }

  /**
   * Updates the graph visualization based on new graph data received from the server.
   * @param {object} graphData - The updated graph data containing nodes and edges.
   */
  updateGraph(graphData) {
    // Update nodes
    graphData.nodes.forEach(node => {
      const mesh = this.nodeMeshes.get(node.id);
      if (mesh) {
        // Update node position
        mesh.position.set(node.x, node.y, node.z);

        // Optionally, update node properties like color or size
        // Example: mesh.material.color.set(node.color || 0xff0000);
      } else {
        // If the node doesn't exist, create it
        this.createNodeObjects([node]);
      }
    });

    // Remove nodes that no longer exist
    const existingNodeIds = new Set(graphData.nodes.map(node => node.id));
    this.nodeMeshes.forEach((mesh, nodeId) => {
      if (!existingNodeIds.has(nodeId)) {
        this.scene.remove(mesh);
        this.nodeMeshes.delete(nodeId);
      }
    });

    // Update edges
    graphData.edges.forEach(edge => {
      const edgeKey = `${edge.source}-${edge.target_node}`;
      const line = this.edgeMeshes.get(edgeKey);
      const sourceNode = this.nodeMeshes.get(edge.source);
      const targetNode = this.nodeMeshes.get(edge.target_node);

      if (line && sourceNode && targetNode) {
        // Update edge positions
        const positions = line.geometry.attributes.position.array;
        positions[0] = sourceNode.position.x;
        positions[1] = sourceNode.position.y;
        positions[2] = sourceNode.position.z;
        positions[3] = targetNode.position.x;
        positions[4] = targetNode.position.y;
        positions[5] = targetNode.position.z;
        line.geometry.attributes.position.needsUpdate = true;

        // Optionally, update edge properties like color or opacity
        // Example: line.material.color.set(edge.color || 0xffffff);
      } else if (sourceNode && targetNode) {
        // If the edge doesn't exist, create it
        this.createEdgeObjects([edge]);
      }
    });

    // Remove edges that no longer exist
    const existingEdgeKeys = new Set(graphData.edges.map(edge => `${edge.source}-${edge.target_node}`));
    this.edgeMeshes.forEach((line, edgeKey) => {
      if (!existingEdgeKeys.has(edgeKey)) {
        this.scene.remove(line);
        this.edgeMeshes.delete(edgeKey);
      }
    });
  }

  /**
   * Renders the scene using the Three.js renderer.
   */
  render() {
    // Update orbit controls for smooth camera movement
    this.controls.update();

    // Render the scene
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Animation loop for continuous rendering.
   */
  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.render();
  }

  /**
   * Handles window resize events by updating camera and renderer dimensions.
   */
  onWindowResize() {
    // Update camera aspect ratio
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();

    // Update renderer size
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
}
