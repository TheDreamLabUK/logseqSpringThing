import * as THREE from 'three'; // Importing the Three.js library for 3D graphics
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // Importing controls for orbiting the camera
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'; // For post-processing effects
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'; // For rendering the scene
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'; // For bloom effects in the scene

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01; // Speed at which the camera translates
const ROTATION_SPEED = 0.01; // Speed at which the camera rotates

/**
 * Class representing a WebXR visualization environment.
 */
export class WebXRVisualization {
    /**
     * Create a WebXR visualization.
     * @param {Object} graphDataManager - The manager for handling graph data.
     */
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called'); // Log when the constructor is called
        this.graphDataManager = graphDataManager; // Store the graph data manager

        // Initialize the scene, camera, and renderer
        this.scene = new THREE.Scene(); // Create a new Three.js scene
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000); // Create a camera with perspective
        this.camera.position.set(0, 0, 500); // Set camera position

        // Set up the renderer with antialiasing for smoother edges
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight); // Set the renderer size
        this.renderer.setPixelRatio(window.devicePixelRatio); // Adjust for device pixel ratio
        this.renderer.toneMapping = THREE.ReinhardToneMapping; // Set tone mapping
        this.renderer.toneMappingExposure = 1; // Set exposure level

        // Initialize variables for controls, composer, meshes, etc.
        this.controls = null; // For camera controls
        this.composer = null; // For post-processing composer

        // Maps to hold node and edge meshes and their labels
        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();
        this.nodeLabels = new Map();

        this.hologramGroup = new THREE.Group(); // Group for holographic elements
        this.animationFrameId = null; // For managing animation frames

        this.selectedNode = null; // Currently selected node

        // Initialize separate bloom passes for visual effects
        this.nodeBloomPass = null;
        this.edgeBloomPass = null;
        this.environmentBloomPass = null;

        // Call method to initialize settings
        this.initializeSettings();
        console.log('WebXRVisualization constructor completed'); // Log when constructor is done
    }

    /**
     * Initialize visualization settings.
     */
    initializeSettings() {
        console.log('Initializing settings');
        // Set up various color and visual settings
        this.nodeColor = 0x1A0B31; // Color for nodes
        this.edgeColor = 0xff0000; // Color for edges
        this.hologramColor = 0xFFD700; // Color for holograms
        this.nodeSizeScalingFactor = 1; // Scaling factor for node sizes
        this.hologramScale = 1; // Scale factor for holograms
        this.hologramOpacity = 0.1; // Opacity for holograms
        this.edgeOpacity = 0.3; // Opacity for edges
        this.labelFontSize = 48; // Font size for node labels
        this.fogDensity = 0.002; // Density for fog in the scene
        this.minNodeSize = 1; // Minimum size for nodes
        this.maxNodeSize = 5; // Maximum size for nodes
        this.nodeBloomStrength = 0.1; // Strength of node bloom effect
        this.nodeBloomRadius = 0.1; // Radius for node bloom effect
        this.nodeBloomThreshold = 0; // Threshold for node bloom effect
        this.edgeBloomStrength = 0.2; // Strength of edge bloom effect
        this.edgeBloomRadius = 0.3; // Radius for edge bloom effect
        this.edgeBloomThreshold = 0; // Threshold for edge bloom effect
        this.environmentBloomStrength = 1; // Strength of environmental bloom effect
        this.environmentBloomRadius = 1; // Radius for environmental bloom effect
        this.environmentBloomThreshold = 0; // Threshold for environmental bloom effect
        console.log('Settings initialized'); // Log completion of settings initialization
    }

    /**
     * Initialize Three.js components.
     */
    initThreeJS() {
        console.log('Initializing Three.js');
        const container = document.getElementById('scene-container'); // Get the container for the scene
        if (container) {
            container.appendChild(this.renderer.domElement); // Append the renderer's DOM element to the container
        } else {
            console.error("Could not find 'scene-container' element"); // Error if container not found
            return;
        }

        // Initialize OrbitControls for the camera
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true; // Enable damping for smoother control
        this.controls.dampingFactor = 0.05; // Set damping factor

        // Set up fog in the scene
        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        this.addLights(); // Add lighting to the scene
        this.initPostProcessing(); // Initialize post-processing effects
        this.createHologramStructure(); // Create the holographic structure

        // Add event listener for window resizing
        window.addEventListener('resize', this.onWindowResize.bind(this), false);

        this.animate(); // Start the animation loop
        console.log('Three.js initialization completed'); // Log completion of initialization
    }

    /**
     * Initialize post-processing effects.
     */
    initPostProcessing() {
        console.log('Initializing post-processing');
        this.composer = new EffectComposer(this.renderer); // Create a new EffectComposer for post-processing
        const renderPass = new RenderPass(this.scene, this.camera); // Create a render pass for the scene
        this.composer.addPass(renderPass); // Add the render pass to the composer

        // Environmental Bloom Pass (Layer 0)
        this.environmentBloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            this.environmentBloomStrength,
            this.environmentBloomRadius,
            this.environmentBloomThreshold
        );
        this.environmentBloomPass.renderToScreen = false; // Do not render this pass to the screen directly
        this.environmentBloomPass.clear = false; // Do not clear the pass
        this.composer.addPass(this.environmentBloomPass); // Add to composer

        // Nodes Bloom Pass (Layer 1)
        this.nodeBloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            this.nodeBloomStrength,
            this.nodeBloomRadius,
            this.nodeBloomThreshold
        );
        this.nodeBloomPass.renderToScreen = false; // Do not render this pass to the screen directly
        this.nodeBloomPass.clear = false; // Do not clear the pass
        this.composer.addPass(this.nodeBloomPass); // Add to composer

        // Edges Bloom Pass (Layer 2)
        this.edgeBloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            this.edgeBloomStrength,
            this.edgeBloomRadius,
            this.edgeBloomThreshold
        );
        this.edgeBloomPass.renderToScreen = true; // Render this pass to the screen
        this.edgeBloomPass.clear = true; // Clear this pass
        this.composer.addPass(this.edgeBloomPass); // Add to composer

        console.log('Post-processing initialized'); // Log completion of post-processing initialization
    }

    /**
     * Add lighting to the scene.
     */
    addLights() {
        console.log('Adding lights to the scene');
        // Ambient light to illuminate the scene evenly
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
        this.scene.add(ambientLight); // Add ambient light to the scene

        // Directional light to create shadows and highlights
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(50, 50, 50); // Set position of the light
        this.scene.add(directionalLight); // Add directional light to the scene
        console.log('Lights added to the scene'); // Log completion of lighting setup
    }

    /**
     * Create the hologram structure for the visualization.
     */
    createHologramStructure() {
        console.log('Creating hologram structure');
        this.hologramGroup.clear(); // Clear any existing hologram group contents

        // Create various geometries to represent the hologram
        const buckyGeometry = new THREE.IcosahedronGeometry(40 * this.hologramScale, 1); // Create an icosahedron geometry
        const buckyMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true, // Wireframe mode for visibility
            transparent: true,
            opacity: this.hologramOpacity // Set opacity
        });
        const buckySphere = new THREE.Mesh(buckyGeometry, buckyMaterial); // Create mesh for bucky sphere
        buckySphere.userData.rotationSpeed = 0.0001; // Rotation speed for animation
        buckySphere.layers.set(0); // Set layer for the environment
        this.hologramGroup.add(buckySphere); // Add to hologram group

        // Create a geodesic dome
        const geodesicGeometry = new THREE.IcosahedronGeometry(10 * this.hologramScale, 1);
        const geodesicMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const geodesicDome = new THREE.Mesh(geodesicGeometry, geodesicMaterial);
        geodesicDome.userData.rotationSpeed = 0.0002; // Set rotation speed
        geodesicDome.layers.set(0); // Set layer for the environment
        this.hologramGroup.add(geodesicDome); // Add to hologram group

        // Create a triangle geometry
        const triangleGeometry = new THREE.SphereGeometry(100 * this.hologramScale, 32, 32);
        const triangleMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const triangleSphere = new THREE.Mesh(triangleGeometry, triangleMaterial);
        triangleSphere.userData.rotationSpeed = 0.0003; // Set rotation speed
        triangleSphere.layers.set(0); // Set layer for the environment
        this.hologramGroup.add(triangleSphere); // Add to hologram group

        // Add the hologram group to the scene
        this.scene.add(this.hologramGroup);
        console.log('Hologram structure created'); // Log completion of hologram structure creation
    }

    /**
     * Update the visualization based on graph data.
     */
    updateVisualization() {
        console.log('Updating visualization');
        const graphData = this.graphDataManager.getGraphData(); // Retrieve graph data from the manager
        if (!graphData) {
            console.warn('No graph data available for visualization update'); // Warn if no data available
            return;
        }
        console.log('Graph data received:', graphData); // Log received graph data
        this.updateNodes(graphData.nodes); // Update nodes in the visualization
        this.updateEdges(graphData.edges); // Update edges in the visualization
        console.log('Visualization update completed'); // Log completion of update
    }

    /**
     * Update visual features based on provided changes.
     * @param {Object} changes - The changes to apply to visual features.
     */
    updateVisualFeatures(changes) {
        console.log('Updating visual features:', changes);
        let needsUpdate = false; // Flag to track if an update is needed
        let bloomChanged = false; // Flag to track if any bloom-related property has changed

        // Iterate over the changes and update properties
        for (const [name, value] of Object.entries(changes)) {
            if (this.hasOwnProperty(name)) {
                console.log(`Setting property ${name} to`, value); // Log property being updated
                this[name] = value; // Update the property
                needsUpdate = true; // Mark that an update is needed

                // Check if the changed property is bloom-related
                if (name.includes('Bloom')) {
                    bloomChanged = true; // Mark that bloom properties have changed
                }
            } else {
                console.warn(`Property ${name} does not exist on WebXRVisualization`); // Warn for unknown properties
            }
        }

        if (needsUpdate) {
            this.updateVisualization(); // Update the visualization if needed
            
            // Specifically handle hologram scale updates
            if (changes.hologramScale !== undefined) {
                this.hologramGroup.scale.set(this.hologramScale, this.hologramScale, this.hologramScale); // Scale hologram group
            }
        }

        // Always update bloom pass if any bloom-related property has changed
        if (bloomChanged) {
            this.updateBloomPass(); // Update bloom passes
        }

        this.composer.render(); // Render the scene with post-processing
        console.log('Visual features update completed'); // Log completion of visual feature update
    }

    /**
     * Update nodes based on new data.
     * @param {Array} nodes - The new nodes data.
     */
    updateNodes(nodes) {
        console.log(`Updating nodes: ${nodes.length}`); // Log number of nodes being updated
        const existingNodeIds = new Set(nodes.map(node => node.id)); // Create a set of existing node IDs

        // Remove any nodes that are no longer present
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                this.scene.remove(mesh); // Remove mesh from scene
                this.nodeMeshes.delete(nodeId); // Remove from node meshes map
                const label = this.nodeLabels.get(nodeId);
                if (label) {
                    this.scene.remove(label); // Remove label from scene
                    this.nodeLabels.delete(nodeId); // Remove from node labels map
                }
            }
        });

        // Iterate through the new nodes
        nodes.forEach(node => {
            // Validate node data
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node); // Warn for invalid node data
                return; // Skip invalid nodes
            }
            let mesh = this.nodeMeshes.get(node.id); // Check if mesh already exists
            const fileSize = node.metadata && node.metadata.file_size ? parseInt(node.metadata.file_size) : 1; // Get file size
            if (isNaN(fileSize) || fileSize <= 0) {
                console.warn(`Invalid file_size for node ${node.id}:`, node.metadata.file_size); // Warn for invalid file size
                return; // Skip if file size is invalid
            }
            const size = this.calculateNodeSize(fileSize); // Calculate node size based on file size
            const color = this.calculateNodeColor(fileSize); // Calculate node color based on file size

            console.log(`Node ${node.id}: fileSize = ${fileSize}, calculated size = ${size}`); // Log calculated size and file size

            // Create or update the mesh for the node
            if (!mesh) {
                // Create new mesh if it doesn't exist
                const geometry = this.createNodeGeometry(size, fileSize); // Create geometry based on size
                const material = new THREE.MeshStandardMaterial({ color: color }); // Create material for the mesh
                mesh = new THREE.Mesh(geometry, material); // Create mesh
                mesh.layers.enable(1); // Set layer for the node
                this.scene.add(mesh); // Add mesh to the scene
                this.nodeMeshes.set(node.id, mesh); // Store the mesh in the map

                // Create and add the node label
                const label = this.createNodeLabel(node.label || node.id, fileSize); // Create label
                this.scene.add(label); // Add label to the scene
                this.nodeLabels.set(node.id, label); // Store label in the map
            } else {
                // Update existing mesh if it already exists
                this.updateNodeGeometry(mesh, size, fileSize); // Update geometry
                mesh.material.color.setHex(color); // Update color
            }

            // Set position of the mesh
            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id); // Get associated label
            if (label) {
                // Set position of the label above the node
                label.position.set(node.x, node.y + size + 2, node.z);
                this.updateNodeLabel(label, node.label || node.id, fileSize); // Update the label text
            }
        });
    }

    /**
     * Calculate the size of a node based on its file size.
     * @param {number} fileSize - The file size of the node.
     * @returns {number} - The calculated size of the node.
     */
    calculateNodeSize(fileSize) {
        // Logarithmic scaling for better size distribution
        const logSize = Math.log(fileSize + 1) / Math.log(10); // Calculate log10 of the file size
        return Math.max(this.minNodeSize, Math.min(this.maxNodeSize, logSize * this.nodeSizeScalingFactor)); // Scale and clamp the size
    }

    /**
    * Calculate the color of a node based on its last modified date.
    * @param {Date} lastModified - The last modified date of the node.
    * @returns {number} - The calculated color of the node.
    */
    calculateNodeColor(lastModified) {
        // Normalize the last modified date to a color gradient
        const now = Date.now(); // Get the current time
        const timeDifference = now - new Date(lastModified).getTime(); // Calculate the time difference in milliseconds

        // Define thresholds for color scaling based on last modified time
        const maxAge = 1000 * 60 * 60 * 24 * 30; // 30 days in milliseconds
        const t = Math.min(timeDifference / maxAge, 1); // Normalize to [0, 1] based on the defined max age

        // Calculate RGB values based on normalized time
        const r = Math.floor(255 * (1 - t)); // Red component decreases with time
        const g = Math.floor(255 * t); // Green component increases with time
        const b = 100; // Keep the blue component constant for visibility

        return (r << 16) | (g << 8) | b; // Combine RGB into a single hexadecimal color value
    }


    /**
     * Create geometry for a node based on its size and file size.
     * @param {number} size - The size of the node.
     * @param {number} fileSize - The file size of the node.
     * @returns {THREE.Geometry} - The created geometry for the node.
     */
    createNodeGeometry(size, fileSize) {
        // Create different geometries based on file size ranges
        if (fileSize < 1000) { // < 1KB
            return new THREE.SphereGeometry(size, 16, 16); // Sphere for small sizes
        } else if (fileSize < 1000000) { // < 1MB
            return new THREE.BoxGeometry(size, size, size); // Box for medium sizes
        } else {
            return new THREE.OctahedronGeometry(size); // Octahedron for large sizes
        }
    }

    /**
     * Update the geometry of an existing node mesh.
     * @param {THREE.Mesh} mesh - The existing node mesh.
     * @param {number} size - The new size of the node.
     * @param {number} fileSize - The file size of the node.
     */
    updateNodeGeometry(mesh, size, fileSize) {
        const newGeometry = this.createNodeGeometry(size, fileSize); // Create new geometry based on updated size
        mesh.geometry.dispose(); // Dispose of old geometry
        mesh.geometry = newGeometry; // Assign new geometry
    }

    /**
     * Create a label for a node.
     * @param {string} text - The text for the label.
     * @param {number} fileSize - The file size associated with the label.
     * @returns {THREE.Sprite} - The created label as a sprite.
     */
    createNodeLabel(text, fileSize) {
        const canvas = document.createElement('canvas'); // Create an off-screen canvas for label
        const context = canvas.getContext('2d'); // Get the 2D drawing context
        context.font = `${this.labelFontSize}px Arial`; // Set font for the label
        const metrics = context.measureText(text); // Measure text width
        const textWidth = metrics.width; // Get text width

        // Set canvas size based on text dimensions
        canvas.width = textWidth + 20; // Increased padding
        canvas.height = this.labelFontSize + 30; // Increased height for file size info

        // Draw background and text on the canvas
        context.fillStyle = 'rgba(0, 0, 0, 0.8)'; // Background color with transparency
        context.fillRect(0, 0, canvas.width, canvas.height); // Draw background rectangle
        context.font = `${this.labelFontSize}px Arial`; // Set font for label text
        context.fillStyle = 'white'; // Set text color to white
        context.fillText(text, 10, this.labelFontSize); // Draw the main label text
        
        // Add file size information
        context.font = `${this.labelFontSize / 2}px Arial`; // Smaller font for file size
        context.fillStyle = 'lightgray'; // Set text color to light gray
        context.fillText(this.formatFileSize(fileSize), 10, this.labelFontSize + 20); // Draw file size text

        const texture = new THREE.CanvasTexture(canvas); // Create texture from canvas
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture }); // Create sprite material from texture
        const sprite = new THREE.Sprite(spriteMaterial); // Create sprite for the label
        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1); // Scale sprite
        sprite.layers.set(1); // Set layer for the label

        spriteMaterial.depthWrite = false; // Disable depth writing for transparency
        spriteMaterial.transparent = true; // Enable transparency for sprite material

        return sprite; // Return the created sprite
    }

    /**
     * Update an existing node label.
     * @param {THREE.Sprite} label - The label sprite to update.
     * @param {string} text - The new text for the label.
     * @param {number} fileSize - The file size associated with the label.
     */
    updateNodeLabel(label, text, fileSize) {
        const canvas = label.material.map.image; // Get the canvas from the label's material
        const context = canvas.getContext('2d'); // Get the 2D drawing context
        context.clearRect(0, 0, canvas.width, canvas.height); // Clear previous label content

        // Redraw the updated background and text on the canvas
        context.fillStyle = 'rgba(0, 0, 0, 0.8)'; // Background color with transparency
        context.fillRect(0, 0, canvas.width, canvas.height); // Draw background rectangle
        context.font = `${this.labelFontSize}px Arial`; // Set font for label text
        context.fillStyle = 'white'; // Set text color to white
        context.fillText(text, 10, this.labelFontSize); // Draw the main label text
        
        context.font = `${this.labelFontSize / 2}px Arial`; // Smaller font for file size
        context.fillStyle = 'lightgray'; // Set text color to light gray
        context.fillText(this.formatFileSize(fileSize), 10, this.labelFontSize + 20); // Draw file size text

        label.material.map.needsUpdate = true; // Indicate that the texture needs to be updated
    }

    /**
     * Format a file size into a human-readable string.
     * @param {number} size - The file size in bytes.
     * @returns {string} - The formatted file size string.
     */
    formatFileSize(size) {
        const units = ['B', 'KB', 'MB', 'GB', 'TB']; // Define size units
        let i = 0; // Unit index
        while (size >= 1024 && i < units.length - 1) {
            size /= 1024; // Convert size to next unit
            i++; // Increment unit index
        }
        return `${size.toFixed(2)} ${units[i]}`; // Return formatted size string
    }

    /**
     * Update edges based on new data.
     * @param {Array} edges - The new edges data.
     */
    updateEdges(edges) {
        console.log(`Updating edges: ${edges.length}`); // Log number of edges being updated
        const existingEdgeKeys = new Set(edges.map(edge => `${edge.source}-${edge.target_node}`)); // Create a set of existing edge keys

        // Remove any edges that are no longer present
        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!existingEdgeKeys.has(edgeKey)) {
                this.scene.remove(line); // Remove line from scene
                this.edgeMeshes.delete(edgeKey); // Remove from edge meshes map
            }
        });

        // Iterate through the new edges
        edges.forEach(edge => {
            // Validate edge data
            if (!edge.source || !edge.target_node) {
                console.warn('Invalid edge data:', edge); // Warn for invalid edge data
                return; // Skip invalid edges
            }
            const edgeKey = `${edge.source}-${edge.target_node}`; // Create a unique key for the edge
            let line = this.edgeMeshes.get(edgeKey); // Check if line already exists
            const sourceMesh = this.nodeMeshes.get(edge.source); // Get source mesh
            const targetMesh = this.nodeMeshes.get(edge.target_node); // Get target mesh
            
            if (!line) {
                // Create new edge line if it doesn't exist
                if (sourceMesh && targetMesh) {
                    const geometry = new THREE.BufferGeometry().setFromPoints([ // Create geometry from source and target positions
                        sourceMesh.position,
                        targetMesh.position
                    ]);
                    const material = new THREE.LineBasicMaterial({ // Create material for the line
                        color: this.edgeColor,
                        transparent: true,
                        opacity: this.edgeOpacity // Set opacity for transparency
                    });
                    line = new THREE.Line(geometry, material); // Create line from geometry and material
                    line.layers.enable(2); // Set layer for the edge
                    this.scene.add(line); // Add line to the scene
                    this.edgeMeshes.set(edgeKey, line); // Store the line in the map
                } else {
                    console.warn(`Unable to create edge: ${edgeKey}. Source or target node not found.`); // Warn if nodes not found
                }
            } else if (sourceMesh && targetMesh) {
                // Update existing edge if it already exists
                const positions = line.geometry.attributes.position.array; // Get position array from geometry
                // Update positions to match current node positions
                positions[0] = sourceMesh.position.x;
                positions[1] = sourceMesh.position.y;
                positions[2] = sourceMesh.position.z;
                positions[3] = targetMesh.position.x;
                positions[4] = targetMesh.position.y;
                positions[5] = targetMesh.position.z;
                line.geometry.attributes.position.needsUpdate = true; // Indicate that the geometry needs to be updated
                line.material.color.setHex(this.edgeColor); // Update color
                line.material.opacity = this.edgeOpacity; // Update opacity
                line.material.needsUpdate = true; // Indicate that the material needs to be updated
            } else {
                console.warn(`Unable to update edge: ${edgeKey}. Source or target node not found.`); // Warn if nodes not found
            }
        });
    }

    /**
     * Animate the scene.
     */
    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this)); // Request the next animation frame
        this.controls.update(); // Update controls for camera movement

        // Rotate hologram children for animation
        this.hologramGroup.children.forEach(child => {
            child.rotation.x += child.userData.rotationSpeed; // Rotate based on defined speed
            child.rotation.y += child.userData.rotationSpeed; // Rotate based on defined speed
        });

        // Update label orientations to face the camera
        this.nodeLabels.forEach(label => {
            label.lookAt(this.camera.position); // Make label face the camera
        });

        this.composer.render(); // Render the scene with post-processing effects
    }

    /**
     * Handle window resize events.
     */
    onWindowResize() {
        console.log('Window resized'); // Log when window is resized
        this.camera.aspect = window.innerWidth / window.innerHeight; // Update camera aspect ratio
        this.camera.updateProjectionMatrix(); // Update projection matrix for the camera
        this.renderer.setSize(window.innerWidth, window.innerHeight); // Update renderer size
        if (this.composer) {
            this.composer.setSize(window.innerWidth, window.innerHeight); // Update composer size
        }
    }

    /**
     * Dispose of the visualization and release resources.
     */
    dispose() {
        console.log('Disposing WebXRVisualization'); // Log when disposing
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId); // Cancel any running animation frames
        }
        this.scene.traverse(object => {
            // Traverse through the scene and dispose of geometries and materials
            if (object.geometry) {
                object.geometry.dispose(); // Dispose of geometry
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose()); // Dispose of each material if it's an array
                } else {
                    object.material.dispose(); // Dispose of the material
                }
            }
        });
        this.renderer.dispose(); // Dispose of the renderer
        if (this.controls) {
            this.controls.dispose(); // Dispose of the controls
        }
        console.log('WebXRVisualization disposed'); // Log completion of disposal
    }

    /**
     * Update node labels in the scene.
     */
    updateNodeLabels() {
        console.log('Updating node labels'); // Log when updating labels
        this.nodeLabels.forEach((label, nodeId) => {
            const node = this.nodeMeshes.get(nodeId); // Get the associated node mesh
            if (node) {
                this.scene.remove(label); // Remove the old label from the scene
                const newLabel = this.createNodeLabel(label.userData.text); // Create a new label
                newLabel.position.copy(label.position); // Set position to match old label
                this.scene.add(newLabel); // Add the new label to the scene
                this.nodeLabels.set(nodeId, newLabel); // Update the map with the new label
            }
        });
        console.log('Node labels update completed'); // Log completion of label update
    }

    /**
     * Update bloom passes based on current settings.
     */
    updateBloomPass() {
        console.log('Updating bloom passes'); // Log when updating bloom passes
        if (this.nodeBloomPass) {
            this.nodeBloomPass.strength = this.nodeBloomStrength; // Update strength for node bloom
            this.nodeBloomPass.radius = this.nodeBloomRadius; // Update radius for node bloom
            this.nodeBloomPass.threshold = this.nodeBloomThreshold; // Update threshold for node bloom
            console.log('Node bloom updated:', {
                strength: this.nodeBloomStrength,
                radius: this.nodeBloomRadius,
                threshold: this.nodeBloomThreshold
            });
        }
        if (this.edgeBloomPass) {
            this.edgeBloomPass.strength = this.edgeBloomStrength; // Update strength for edge bloom
            this.edgeBloomPass.radius = this.edgeBloomRadius; // Update radius for edge bloom
            this.edgeBloomPass.threshold = this.edgeBloomThreshold; // Update threshold for edge bloom
            console.log('Edge bloom updated:', {
                strength: this.edgeBloomStrength,
                radius: this.edgeBloomRadius,
                threshold: this.edgeBloomThreshold
            });
        }
        if (this.environmentBloomPass) {
            this.environmentBloomPass.strength = this.environmentBloomStrength; // Update strength for environment bloom
            this.environmentBloomPass.radius = this.environmentBloomRadius; // Update radius for environment bloom
            this.environmentBloomPass.threshold = this.environmentBloomThreshold; // Update threshold for environment bloom
            console.log('Environment bloom updated:', {
                strength: this.environmentBloomStrength,
                radius: this.environmentBloomRadius,
                threshold: this.environmentBloomThreshold
            });
        }
        console.log('Bloom passes update completed'); // Log completion of bloom pass updates
    }

    /**
     * Handle input from a Spacemouse device.
     * @param {number} x - Movement along the x-axis.
     * @param {number} y - Movement along the y-axis.
     * @param {number} z - Movement along the z-axis.
     * @param {number} rx - Rotation around the x-axis.
     * @param {number} ry - Rotation around the y-axis.
     * @param {number} rz - Rotation around the z-axis.
     */
    handleSpacemouseInput(x, y, z, rx, ry, rz) {
        if (!this.camera || !this.controls) {
            console.warn('Camera or controls not initialized for Spacemouse input'); // Warn if not initialized
            return; // Skip input handling if not initialized
        }

        // Update camera position based on Spacemouse input
        this.camera.position.x += x * TRANSLATION_SPEED;
        this.camera.position.y += y * TRANSLATION_SPEED;
        this.camera.position.z += z * TRANSLATION_SPEED;
        
        // Update camera rotation based on Spacemouse input
        this.camera.rotation.x += rx * ROTATION_SPEED;
        this.camera.rotation.y += ry * ROTATION_SPEED;
        this.camera.rotation.z += rz * ROTATION_SPEED;

        this.controls.update(); // Update controls after position/rotation changes
    }

    /**
     * Debugging utility for node labels.
     */
    debugLabels() {
        console.log('Debugging labels'); // Log when debugging labels
        console.log('Total labels:', this.nodeLabels.size); // Log total number of labels

        // Update camera matrices for frustum calculations
        this.camera.updateMatrixWorld();
        this.camera.updateProjectionMatrix();

        // Create a frustum based on the camera's projection matrix
        const frustum = new THREE.Frustum();
        const cameraViewProjectionMatrix = new THREE.Matrix4();
        cameraViewProjectionMatrix.multiplyMatrices(
            this.camera.projectionMatrix,
            this.camera.matrixWorldInverse
        );
        frustum.setFromProjectionMatrix(cameraViewProjectionMatrix); // Set the frustum based on the matrix

        // Log information for each label
        this.nodeLabels.forEach((label, nodeId) => {
            console.log(`Label for node ${nodeId}:`, {
                position: label.position.toArray(), // Log position of the label
                visible: label.visible, // Log visibility of the label
                inFrustum: frustum.containsPoint(label.position), // Check if label is within the camera's frustum
                material: {
                    color: label.material.color.getHex(), // Log color of the label's material
                    opacity: label.material.opacity, // Log opacity of the label's material
                    transparent: label.material.transparent, // Log transparency setting
                    visible: label.material.visible // Log visibility of the label's material
                },
                geometry: {
                    type: label.geometry.type, // Log geometry type
                    parameters: label.geometry.parameters // Log geometry parameters
                }
            });
        });
    }
}
