import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { NodeManager } from './nodes.js';
import { EffectsManager } from './effects.js';
import { LayoutManager } from './layout.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01;
const ROTATION_SPEED = 0.01;

export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        // Initialize the scene, camera, and renderer
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.position.set(0, 0, 500);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.setClearColor(0x000000);

        this.controls = null;
        this.animationFrameId = null;
        this.lastPositionUpdate = 0;
        this.positionUpdateThreshold = 16; // ~60fps for smooth updates

        // Get initial settings
        const settings = visualizationSettings.getSettings();

        // Initialize managers with settings
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());
        this.effectsManager = new EffectsManager(
            this.scene, 
            this.camera, 
            this.renderer,
            visualizationSettings.getEnvironmentSettings()
        );

        // Initialize layout manager with settings
        const layoutSettings = visualizationSettings.getLayoutSettings();
        this.layoutManager = new LayoutManager(layoutSettings);
        console.log('Layout manager initialized with settings:', layoutSettings);

        // Initialize settings
        this.initializeSettings();

        // Add event listeners for graph data and position updates
        window.addEventListener('graphDataUpdated', (event) => {
            console.log('Received graphDataUpdated event:', event.detail);
            this.updateVisualization();
        });

        window.addEventListener('visualizationSettingsUpdated', (event) => {
            console.log('Received visualizationSettingsUpdated event:', event.detail);
            this.updateSettings(event.detail);
        });

        // Handle position updates from layout manager
        window.addEventListener('positionUpdate', (event) => {
            console.log('Received position update');
            if (this.graphDataManager.websocketService && this.graphDataManager.websocketService.socket) {
                // Send binary position data directly through websocket
                this.graphDataManager.websocketService.socket.send(event.detail);
            }
        });

        // Handle incoming position updates from other clients
        window.addEventListener('nodePositionsUpdated', (event) => {
            console.log('Received position update from server');
            if (Array.isArray(event.detail)) {
                this.nodeManager.updateNodePositions(event.detail);
            }
        });

        console.log('WebXRVisualization constructor completed');
    }

    updateSettings(settings) {
        console.log('Updating settings:', settings);
        if (!settings) return;

        // Update visualization settings
        Object.entries(settings).forEach(([key, value]) => {
            if (typeof value === 'object') {
                // Handle nested settings objects
                Object.entries(value).forEach(([subKey, subValue]) => {
                    const fullKey = `${key}_${subKey}`;
                    this.updateVisualFeatures(fullKey, subValue);
                });
            } else {
                this.updateVisualFeatures(key, value);
            }
        });
    }

    initializeSettings() {
        console.log('Initializing settings');
        this.fogDensity = 0.001; // Reduced fog density
        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        
        // Lighting settings
        this.ambientLightIntensity = 50;
        this.directionalLightIntensity = 5.0;
        this.directionalLightColor = 0xffffff;
        this.ambientLightColor = 0x404040;
        
        this.addLights();
        console.log('Settings initialized');
    }

    addLights() {
        console.log('Adding lights to the scene');

        // Create and configure ambient light
        this.ambientLight = new THREE.AmbientLight(this.ambientLightColor, this.ambientLightIntensity);
        this.scene.add(this.ambientLight);

        // Create and configure main directional light
        this.directionalLight = new THREE.DirectionalLight(this.directionalLightColor, this.directionalLightIntensity);
        this.directionalLight.position.set(50, 50, 50);
        this.scene.add(this.directionalLight);

        // Add additional directional lights for better coverage
        const frontLight = new THREE.DirectionalLight(0xffffff, 3.0);
        frontLight.position.set(0, 0, 100);
        this.scene.add(frontLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 3.0);
        backLight.position.set(0, 0, -100);
        this.scene.add(backLight);

        const topLight = new THREE.DirectionalLight(0xffffff, 3.0);
        topLight.position.set(0, 100, 0);
        this.scene.add(topLight);

        // Add point lights for additional illumination
        const pointLight1 = new THREE.PointLight(0xffffff, 2.0, 1000);
        pointLight1.position.set(100, 100, 100);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xffffff, 2.0, 1000);
        pointLight2.position.set(-100, -100, -100);
        this.scene.add(pointLight2);

        console.log('Lights added to the scene');
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

        this.effectsManager.initPostProcessing();
        this.effectsManager.createHologramStructure();

        window.addEventListener('resize', this.onWindowResize.bind(this), false);

        this.animate();
        console.log('Three.js initialization completed');
    }

    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        this.controls.update();

        this.effectsManager.animate();
        this.nodeManager.updateLabelOrientations(this.camera);
        this.effectsManager.render();
    }

    onWindowResize() {
        console.log('Window resized');
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.effectsManager.onResize(window.innerWidth, window.innerHeight);
    }

    updateVisualization() {
        console.log('Updating visualization');
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData || !graphData.nodes || !graphData.edges) {
            console.warn('Invalid graph data:', graphData);
            return;
        }

        try {
            // Update visual elements
            this.nodeManager.updateNodes(graphData.nodes);
            this.nodeManager.updateEdges(graphData.edges);
            
            // Initialize layout manager if needed
            if (!this.layoutManager.isInitialized) {
                console.log('Initializing layout manager with nodes:', graphData.nodes);
                this.layoutManager.initializePositions(graphData.nodes);
                this.layoutManager.startContinuousSimulation(graphData);
            }

            // Apply force-directed layout
            this.layoutManager.applyForceDirectedLayout(graphData.nodes, graphData.edges);
            
            // Update node positions
            this.nodeManager.updateNodePositions(graphData.nodes);
            
        } catch (error) {
            console.error('Error updating visualization:', error);
            console.error('Error stack:', error.stack);
        }
    }

    updateVisualFeatures(control, value) {
        console.log(`Updating visual feature: ${control} = ${value}`);
        if (!control || typeof control !== 'string') {
            console.warn('Invalid control parameter:', control);
            return;
        }
        
        try {
            // Delegate updates to appropriate managers
            if (control.startsWith('node') || control.startsWith('edge')) {
                this.nodeManager.updateFeature(control, value);
            } else if (control.startsWith('bloom') || control.startsWith('hologram')) {
                this.effectsManager.updateFeature(control, value);
            } else if (control.startsWith('forceDirected')) {
                console.log('Updating layout feature:', control, value);
                this.layoutManager.updateFeature(control, value);
                this.graphDataManager.updateForceDirectedParams(control.replace('forceDirected', ''), value);
            }

            // Handle lighting and other scene-level features
            switch (control) {
                case 'ambientLightIntensity':
                    this.ambientLight.intensity = value;
                    break;
                case 'directionalLightIntensity':
                    this.directionalLight.intensity = value;
                    break;
                case 'ambientLightColor':
                    this.ambientLight.color.setHex(value);
                    break;
                case 'directionalLightColor':
                    this.directionalLight.color.setHex(value);
                    break;
                case 'fogDensity':
                    this.scene.fog.density = value;
                    break;
            }
        } catch (error) {
            console.error('Error updating visual feature:', error);
            console.error('Control:', control, 'Value:', value);
            console.error('Stack:', error.stack);
        }
    }

    handleSpacemouseInput(x, y, z, rx, ry, rz) {
        if (!this.camera || !this.controls) {
            console.warn('Camera or controls not initialized for Spacemouse input');
            return;
        }

        // Apply translation
        this.camera.position.x += x * TRANSLATION_SPEED;
        this.camera.position.y += y * TRANSLATION_SPEED;
        this.camera.position.z += z * TRANSLATION_SPEED;
        
        // Apply rotation
        this.camera.rotation.x += rx * ROTATION_SPEED;
        this.camera.rotation.y += ry * ROTATION_SPEED;
        this.camera.rotation.z += rz * ROTATION_SPEED;

        this.controls.update();
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        this.nodeManager.dispose();
        this.effectsManager.dispose();
        this.layoutManager.stopSimulation();

        this.renderer.dispose();
        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }

    handleNodeDrag(nodeId, position) {
        // Update local node position
        this.nodeManager.updateNodePosition(nodeId, position);

        const now = Date.now();
        if (now - this.lastPositionUpdate >= this.positionUpdateThreshold) {
            this.lastPositionUpdate = now;
            
            // Get all node positions for synchronization
            const positions = this.nodeManager.getNodePositions();
            
            // Create binary position data with positions and velocities (24 bytes per node)
            const buffer = new ArrayBuffer(positions.length * 24);
            const view = new Float32Array(buffer);
            
            positions.forEach((pos, index) => {
                const offset = index * 6;
                // Position
                view[offset] = pos.position.x;
                view[offset + 1] = pos.position.y;
                view[offset + 2] = pos.position.z;
                // Velocity
                view[offset + 3] = pos.velocity.x;
                view[offset + 4] = pos.velocity.y;
                view[offset + 5] = pos.velocity.z;
            });

            // Dispatch binary position update
            window.dispatchEvent(new CustomEvent('positionUpdate', {
                detail: buffer
            }));
        }
    }

    handleBinaryPositionUpdate(buffer) {
        const positions = new Float32Array(buffer);
        const updates = [];
        
        // Each position update contains 6 float values (x,y,z, vx,vy,vz)
        for (let i = 0; i < positions.length; i += 6) {
            updates.push({
                position: new THREE.Vector3(
                    positions[i],
                    positions[i + 1],
                    positions[i + 2]
                ),
                velocity: new THREE.Vector3(
                    positions[i + 3],
                    positions[i + 4],
                    positions[i + 5]
                )
            });
        }

        // Fast update through node manager
        this.nodeManager.updateNodePositions(updates);
    }
}
