// Previous imports unchanged...
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { NodeManager } from './nodes.js';
import { EffectsManager } from './effects.js';
import { visualizationSettings } from '../services/visualizationSettings.js';
import { initXRSession, addXRButton, handleXRSession } from '../xr/xrSetup.js';
import { initXRInteraction } from '../xr/xrInteraction.js';

// Constants for input sensitivity
const TRANSLATION_SPEED = 0.01;
const ROTATION_SPEED = 0.01;
const VR_MOVEMENT_SPEED = 0.05;

export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called with graphDataManager:', !!graphDataManager);
        if (!graphDataManager) {
            throw new Error('GraphDataManager is required for WebXRVisualization');
        }
        this.graphDataManager = graphDataManager;

        // Wait for settings before initializing
        this.initialized = false;
        this.pendingInitialization = true;

        // Store references that will be initialized once settings are received
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.nodeManager = null;
        this.controls = null;
        this.xrSessionManager = null;
        this.canvas = null;

        // Position tracking
        this.nodePositions = new Map();
        this.positionBuffer = null;

        // Bind methods
        this.onWindowResize = this.onWindowResize.bind(this);
        this.animate = this.animate.bind(this);
        this.updateVisualization = this.updateVisualization.bind(this);
        this.handleSpacemouseInput = this.handleSpacemouseInput.bind(this);
        this.renderFrame = this.renderFrame.bind(this);
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        this.handleBinaryUpdate = this.handleBinaryUpdate.bind(this);

        // Add event listeners
        window.addEventListener('binaryPositionUpdate', this.handleBinaryUpdate);
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
    }

    handleBinaryUpdate(event) {
        if (!this.initialized || !this.nodeManager) return;

        const { positions } = event.detail;
        
        // Update position cache
        this.nodeManager.nodes.forEach((node, index) => {
            if (positions[index]) {
                const pos = positions[index];
                // Update node position directly
                node.position.set(pos[0], pos[1], pos[2]);
                // Cache position for quick access
                this.nodePositions.set(node.id, {
                    position: [pos[0], pos[1], pos[2]],
                    velocity: [pos[3], pos[4], pos[5]]
                });
            }
        });

        // Update edges if needed (they might need to follow node positions)
        if (this.nodeManager.updateEdgePositions) {
            this.nodeManager.updateEdgePositions();
        }
    }

    updateVisualization(data) {
        if (!this.nodeManager || !data) {
            console.warn('Cannot update visualization: missing manager or data');
            return;
        }

        console.log(`Updating visualization with ${data.nodes?.length || 0} nodes and ${data.edges?.length || 0} edges`);

        // Handle full graph updates (structure changes)
        if (data.nodes || data.edges) {
            if (Array.isArray(data.nodes)) {
                console.log('Updating nodes');
                this.nodeManager.updateNodes(data.nodes);
                // Update position cache
                data.nodes.forEach(node => {
                    this.nodePositions.set(node.id, {
                        position: [node.x, node.y, node.z],
                        velocity: [node.vx || 0, node.vy || 0, node.vz || 0]
                    });
                });
            }
            
            if (Array.isArray(data.edges)) {
                console.log('Updating edges');
                this.nodeManager.updateEdges(data.edges);
            }
        }
    }

    async initializeXR() {
        try {
            // Enable XR on renderer
            this.renderer.xr.enabled = true;

            // Initialize XR session manager
            this.xrSessionManager = await initXRSession(this.renderer, this.scene, this.camera);
            
            if (this.xrSessionManager) {
                // Add XR button to the scene
                await addXRButton(this.xrSessionManager);
                console.log('XR initialization complete');
            } else {
                console.warn('XR not supported or initialization failed');
            }
        } catch (error) {
            console.error('Error initializing XR:', error);
        }
    }

    renderFrame() {
        // Update controls if not in XR mode
        if (this.controls && (!this.xrSessionManager || !this.renderer.xr.isPresenting)) {
            this.controls.update();
        }

        // Update XR session if active
        if (this.xrSessionManager && this.renderer.xr.isPresenting) {
            this.xrSessionManager.update();
        }

        // Update node labels using cached positions
        if (this.nodeManager) {
            this.nodeManager.updateLabelOrientations(this.camera);
        }

        // Render scene
        this.renderer.render(this.scene, this.camera);
    }

    animate() {
        // Use requestAnimationFrame for non-XR rendering
        if (!this.renderer.xr.isPresenting) {
            requestAnimationFrame(this.animate);
            this.renderFrame();
        }
    }

    initializeSettings() {
        console.log('Initializing settings');
        
        const settings = visualizationSettings.getSettings();
        if (!settings?.visualization) {
            console.warn('No visualization settings available');
            return;
        }

        const vis = settings.visualization;
        
        // Add strong ambient light for better visibility
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
        this.scene.add(ambientLight);

        // Add directional light for shadows and highlights
        const directionalLight = new THREE.DirectionalLight(0xffffff, 2.0);
        directionalLight.position.set(10, 20, 10);
        this.scene.add(directionalLight);

        // Add hemisphere light for better ambient illumination
        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
        this.scene.add(hemisphereLight);

        // Add point lights for better illumination
        const pointLight1 = new THREE.PointLight(0xffffff, 1.0, 300);
        pointLight1.position.set(100, 100, 100);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xffffff, 1.0, 300);
        pointLight2.position.set(-100, -100, -100);
        this.scene.add(pointLight2);

        // Set fog from settings
        this.scene.fog = new THREE.FogExp2(0x000000, vis.fog_density);

        console.log('Scene settings initialized with settings from server');
    }

    onWindowResize() {
        if (this.camera && this.renderer && this.canvas) {
            // Update canvas size
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            
            // Update camera
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            
            // Update renderer
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }

    setupEventListeners() {
        console.log('Setting up event listeners');
        
        window.addEventListener('graphDataUpdated', (event) => {
            console.log('Received graphDataUpdated event:', event.detail);
            if (event.detail && Array.isArray(event.detail.nodes)) {
                this.updateVisualization(event.detail);
            } else {
                console.warn('Invalid graph data in event:', event.detail);
            }
        });

        window.addEventListener('resize', this.onWindowResize);
    }

    handleSpacemouseInput(x, y, z) {
        if (!this.camera) return;

        this.camera.position.x += x * TRANSLATION_SPEED;
        this.camera.position.y += y * TRANSLATION_SPEED;
        this.camera.position.z += z * TRANSLATION_SPEED;

        if (this.controls) {
            this.controls.target.copy(this.camera.position).add(
                new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion)
            );
            this.controls.update();
        }
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        
        // Remove event listeners
        window.removeEventListener('binaryPositionUpdate', this.handleBinaryUpdate);
        window.removeEventListener('resize', this.onWindowResize);
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

        // Clear position cache
        this.nodePositions.clear();
        this.positionBuffer = null;

        // Clean up renderer
        this.renderer.setAnimationLoop(null);

        // Clean up managers
        if (this.nodeManager) {
            this.nodeManager.dispose();
        }

        // Clean up DOM elements
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }

        // Dispose of Three.js resources
        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }
}
