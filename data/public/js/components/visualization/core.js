import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { NodeManager } from './nodes.js';
import { EffectsManager } from './effects.js';
import { LayoutManager } from './layout.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';
import { initXRSession, handleXRSession } from '../../xr/xrSetup.js';
import { initXRInteraction, handleXRInput, XRLabelManager } from '../../xr/xrInteraction.js';

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01;
const ROTATION_SPEED = 0.01;

function updateNodeDynamics(nodeManager, updates, isInitialLayout, timeStep) {
    if (isInitialLayout) {
        console.log('Applying initial layout positions and velocities');
        nodeManager.resetSimulation();
    }

    nodeManager.updateNodeDynamics(updates);

    if (timeStep > 0) {
        nodeManager.setTimeStep(timeStep);
    }

    if (nodeManager.isInteractive()) {
        nodeManager.updatePhysics(updates);
    }
}

export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        // Initialize the scene, camera, and renderer
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.position.set(0, 1.6, 3); // Set initial position at standing height

        // Initialize renderer with XR support
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true, // Enable alpha for AR
            logarithmicDepthBuffer: true // Better depth precision for XR
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.xr.enabled = true;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Initialize XR components
        this.xrLabelManager = new XRLabelManager(this.scene, this.camera);
        this.xrControllers = null;
        this.xrHands = null;

        this.controls = null;
        this.animationFrameId = null;
        this.lastPositionUpdate = 0;
        this.positionUpdateThreshold = 16;

        this.previousPositions = new Map();
        this.previousTimes = new Map();
        this.lastUpdateTime = performance.now();

        // Initialize managers with settings from service
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());
        this.effectsManager = new EffectsManager(
            this.scene,
            this.camera,
            this.renderer,
            visualizationSettings.getEnvironmentSettings()
        );
        this.layoutManager = new LayoutManager(visualizationSettings.getLayoutSettings());

        // Initialize settings and add event listeners
        this.initializeSettings();
        this.setupEventListeners();

        console.log('WebXRVisualization constructor completed');
    }

    setupEventListeners() {
        window.addEventListener('graphDataUpdated', (event) => {
            if (event.detail && Array.isArray(event.detail.nodes)) {
                this.updateVisualization(event.detail);
            }
        });

        window.addEventListener('visualizationSettingsUpdated', (event) => {
            this.updateSettings(event.detail);
        });

        window.addEventListener('positionUpdate', (event) => {
            if (this.graphDataManager.isGraphDataValid() && this.graphDataManager.websocketService) {
                this.graphDataManager.websocketService.send(event.detail);
            }
        });

        window.addEventListener('binaryPositionUpdate', (event) => {
            this.handleBinaryPositionUpdate(event.detail);
        });

        // XR-specific event listeners
        window.addEventListener('xrSelectStart', (event) => {
            this.handleXRSelect(event.detail);
        });

        window.addEventListener('xrSelectEnd', (event) => {
            this.handleXRSelectEnd(event.detail);
        });
    }

    initializeSettings() {
        console.log('Initializing settings');
        const envSettings = visualizationSettings.getEnvironmentSettings();
        
        // Initialize fog
        this.fogDensity = envSettings.fogDensity;
        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        
        // Initialize lighting
        this.ambientLightIntensity = 50;
        this.directionalLightIntensity = 5.0;
        this.directionalLightColor = 0xffffff;
        this.ambientLightColor = 0x404040;
        
        // Add ambient light
        this.ambientLight = new THREE.AmbientLight(this.ambientLightColor, this.ambientLightIntensity);
        this.scene.add(this.ambientLight);

        // Add directional light
        this.directionalLight = new THREE.DirectionalLight(
            this.directionalLightColor,
            this.directionalLightIntensity
        );
        this.directionalLight.position.set(5, 5, 5);
        this.directionalLight.castShadow = true;
        this.scene.add(this.directionalLight);

        // Add point lights for better illumination
        const pointLight1 = new THREE.PointLight(0xffffff, 1, 100);
        pointLight1.position.set(10, 10, 10);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xffffff, 1, 100);
        pointLight2.position.set(-10, -10, -10);
        this.scene.add(pointLight2);
    }

    initThreeJS() {
        console.log('Initializing Three.js with XR support');
        const container = document.getElementById('scene-container');
        if (!container) {
            console.error("Could not find 'scene-container' element");
            return;
        }

        container.appendChild(this.renderer.domElement);

        // Initialize XR
        initXRSession(this.renderer, this.scene, this.camera);
        const xrComponents = initXRInteraction(this.scene, this.camera, this.renderer, this.handleXRSelect.bind(this));
        this.xrControllers = xrComponents.controllers;
        this.xrHands = xrComponents.hands;

        // Initialize non-XR controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Disable OrbitControls when in XR
        this.renderer.xr.addEventListener('sessionstart', () => {
            this.controls.enabled = false;
        });

        this.renderer.xr.addEventListener('sessionend', () => {
            this.controls.enabled = true;
        });

        this.effectsManager.initPostProcessing();
        this.effectsManager.createHologramStructure();

        window.addEventListener('resize', this.onWindowResize.bind(this), false);

        this.animate();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.effectsManager.handleResize();
    }

    animate() {
        // Use XR animation loop
        this.renderer.setAnimationLoop((timestamp, frame) => {
            // Update non-XR controls if not in XR session
            if (!this.renderer.xr.isPresenting) {
                this.controls.update();
            }

            // Update XR interactions if in XR session
            if (frame) {
                handleXRInput(frame, this.renderer.xr.getReferenceSpace(), 
                    this.xrControllers, this.xrHands, this.scene, this.camera);
            }

            this.effectsManager.animate();
            this.nodeManager.updateLabelOrientations(this.camera);
            this.xrLabelManager.update(); // Update XR-specific labels

            // Render the scene
            if (this.renderer.xr.isPresenting) {
                this.renderer.render(this.scene, this.camera);
            } else {
                this.effectsManager.render();
            }
        });
    }

    handleXRSelect(event) {
        // Handle XR selection events (controller triggers or hand pinch)
        const intersection = event.intersection;
        if (intersection) {
            const object = intersection.object;
            if (object.userData.nodeId) {
                this.nodeManager.handleNodeSelect(object.userData.nodeId);
            }
        }
    }

    handleXRSelectEnd(event) {
        // Handle XR selection end events
        this.nodeManager.handleNodeSelectEnd();
    }

    updateVisualization(graphData) {
        if (this.nodeManager && graphData) {
            // Update nodes
            if (Array.isArray(graphData.nodes)) {
                this.nodeManager.updateNodes(graphData.nodes);
            }
            
            // Update edges if available
            if (Array.isArray(graphData.edges)) {
                this.nodeManager.updateEdges(graphData.edges);
            }
        }
    }

    updateSettings(settings) {
        // Update each manager with its specific settings
        this.nodeManager.updateFeature(visualizationSettings.getNodeSettings());
        this.effectsManager.updateFeature(visualizationSettings.getEnvironmentSettings());
        this.layoutManager.updateFeature(visualizationSettings.getLayoutSettings());
    }

    handleSpacemouseInput(x, y, z) {
        if (!this.camera || this.renderer.xr.isPresenting) return;

        // Translation
        this.camera.position.x += x * TRANSLATION_SPEED;
        this.camera.position.y += y * TRANSLATION_SPEED;
        this.camera.position.z += z * TRANSLATION_SPEED;

        // Update controls target
        if (this.controls) {
            this.controls.target.copy(this.camera.position).add(
                new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion)
            );
            this.controls.update();
        }
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        this.renderer.setAnimationLoop(null);

        this.nodeManager.dispose();
        this.effectsManager.dispose();
        this.layoutManager.stopSimulation();
        this.xrLabelManager.dispose();

        if (this.xrControllers) {
            this.xrControllers.forEach(controller => {
                this.scene.remove(controller);
            });
        }

        if (this.xrHands) {
            this.xrHands.forEach(hand => {
                this.scene.remove(hand);
            });
        }

        this.renderer.dispose();
        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }
}
