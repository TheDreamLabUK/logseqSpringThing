import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { NodeManager } from './nodes.js';
import { EffectsManager } from './effects.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';
import { initXRSession, addXRButton, handleXRSession } from '../../xr/xrSetup.js';
import { initXRInteraction } from '../../xr/xrInteraction.js';

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01;
const ROTATION_SPEED = 0.01;
const VR_MOVEMENT_SPEED = 0.05;

export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        // Initialize the scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.matrixAutoUpdate = true;
        
        // Set initial camera position for desktop mode
        this.camera.position.set(0, 1.6, 3);

        // Initialize renderer with XR support
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            logarithmicDepthBuffer: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Initialize managers with settings from service
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());
        this.effectsManager = new EffectsManager(
            this.scene,
            this.camera,
            this.renderer,
            visualizationSettings.getEnvironmentSettings()
        );

        this.controls = null;
        this.xrSessionManager = null;
        this.xrControllers = [];
        this.xrHands = [];
        this.xrLabelManager = null;

        // Bind methods
        this.onWindowResize = this.onWindowResize.bind(this);
        this.animate = this.animate.bind(this);
        this.updateVisualization = this.updateVisualization.bind(this);

        // Initialize settings and add event listeners
        this.initializeSettings();
        this.setupEventListeners();

        console.log('WebXRVisualization constructor completed');
    }

    onWindowResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            if (this.effectsManager) {
                this.effectsManager.handleResize();
            }
        }
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
            // Apply position updates received from server
            if (event.detail) {
                this.applyPositionUpdate(event.detail);
            }
        });
    }

    applyPositionUpdate(buffer) {
        try {
            const dataView = new Float32Array(buffer);
            const isInitialLayout = dataView[0] === 1.0;
            
            // Skip the first float (isInitialLayout flag)
            for (let i = 1; i < dataView.length; i += 6) {
                const nodeId = Math.floor((i - 1) / 6);
                const x = dataView[i];
                const y = dataView[i + 1];
                const z = dataView[i + 2];
                
                // Update node position in the visualization
                const mesh = this.nodeManager.nodeMeshes.get(nodeId);
                if (mesh) {
                    mesh.position.set(x, y, z);
                    // Update connected edges
                    this.nodeManager.updateEdgesForNode(nodeId);
                }
            }
        } catch (error) {
            console.error('Error applying position update:', error);
        }
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

    async initThreeJS() {
        console.log('Initializing Three.js');
        const container = document.getElementById('scene-container');
        if (!container) {
            console.error("Could not find 'scene-container' element");
            return;
        }

        // Setup renderer with proper stacking context
        this.renderer.domElement.style.position = 'absolute';
        this.renderer.domElement.style.top = '0';
        this.renderer.domElement.style.left = '0';
        this.renderer.domElement.style.zIndex = '0';
        container.appendChild(this.renderer.domElement);

        // Create a separate div for OrbitControls
        const controlsContainer = document.createElement('div');
        controlsContainer.style.position = 'absolute';
        controlsContainer.style.top = '0';
        controlsContainer.style.left = '0';
        controlsContainer.style.width = '100%';
        controlsContainer.style.height = '100%';
        controlsContainer.style.zIndex = '1';
        container.appendChild(controlsContainer);

        // Initialize controls with optimized settings
        this.controls = new OrbitControls(this.camera, controlsContainer);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.rotateSpeed = 0.8;
        this.controls.panSpeed = 0.8;
        this.controls.zoomSpeed = 0.8;

        // Setup pointer events
        const updatePointerEvents = (isInteracting) => {
            if (!this.renderer.xr.isPresenting) {
                this.renderer.domElement.style.pointerEvents = isInteracting ? 'auto' : 'none';
                controlsContainer.style.pointerEvents = isInteracting ? 'auto' : 'none';
            }
        };

        container.addEventListener('mouseenter', () => updatePointerEvents(true));
        container.addEventListener('mouseleave', () => updatePointerEvents(false));
        
        // Initialize click handling
        this.nodeManager.initClickHandling(this.renderer);

        // Initialize basic XR support
        this.xrSessionManager = await initXRSession(this.renderer, this.scene, this.camera, this.effectsManager);

        // Initialize effects after XR setup
        if (this.effectsManager) {
            this.effectsManager.initPostProcessing();
        }

        // Add resize listener
        window.addEventListener('resize', this.onWindowResize);

        // Start animation loop
        this.animate();
        
        // Add XR button if supported
        if (this.xrSessionManager) {
            await addXRButton(this.xrSessionManager);
        }
    }

    animate() {
        const renderFrame = (timestamp, frame) => {
            // Update controls if enabled
            if (this.controls && this.controls.enabled) {
                this.controls.update();
            }

            // Update labels
            this.nodeManager.updateLabelOrientations(this.camera);

            // Render scene with effects in both desktop and XR modes
            if (this.effectsManager) {
                this.effectsManager.render();
            } else {
                this.renderer.render(this.scene, this.camera);
            }
        };

        this.renderer.setAnimationLoop(renderFrame);
    }

    updateVisualization(graphData) {
        if (!this.nodeManager || !graphData) return;

        console.log('Updating visualization with new graph data');

        // Update visual representation
        if (Array.isArray(graphData.nodes)) {
            this.nodeManager.updateNodes(graphData.nodes);
        }
        
        if (Array.isArray(graphData.edges)) {
            this.nodeManager.updateEdges(graphData.edges);
        }
    }

    updateSettings(settings) {
        console.log('Updating visualization settings:', settings);
        
        if (settings.visual) {
            // Update visual settings
            const visualSettings = {
                nodeColor: settings.visual.nodeColor,
                edgeColor: settings.visual.edgeColor,
                hologramColor: settings.visual.hologramColor,
                minNodeSize: settings.visual.minNodeSize,
                maxNodeSize: settings.visual.maxNodeSize,
                hologramScale: settings.visual.hologramScale,
                hologramOpacity: settings.visual.hologramOpacity,
                edgeOpacity: settings.visual.edgeOpacity,
                fogDensity: settings.visual.fogDensity
            };
            this.nodeManager.updateFeature(visualSettings);
            
            // Update fog density
            if (this.scene.fog && settings.visual.fogDensity !== undefined) {
                this.scene.fog.density = settings.visual.fogDensity;
            }
        }

        if (settings.material) {
            // Update material settings
            const materialSettings = {
                metalness: settings.material.metalness,
                roughness: settings.material.roughness,
                clearcoat: settings.material.clearcoat,
                clearcoatRoughness: settings.material.clearcoatRoughness,
                opacity: settings.material.opacity,
                emissiveMinIntensity: settings.material.emissiveMin,
                emissiveMaxIntensity: settings.material.emissiveMax
            };
            this.nodeManager.updateMaterial(materialSettings);
        }

        if (settings.bloom) {
            // Update bloom settings
            const bloomSettings = {
                nodeBloomStrength: settings.bloom.nodeStrength,
                nodeBloomRadius: settings.bloom.nodeRadius,
                nodeBloomThreshold: settings.bloom.nodeThreshold,
                edgeBloomStrength: settings.bloom.edgeStrength,
                edgeBloomRadius: settings.bloom.edgeRadius,
                edgeBloomThreshold: settings.bloom.edgeThreshold,
                environmentBloomStrength: settings.bloom.envStrength,
                environmentBloomRadius: settings.bloom.envRadius,
                environmentBloomThreshold: settings.bloom.envThreshold
            };
            this.effectsManager.updateBloom(bloomSettings);
        }

        if (settings.fisheye) {
            // Update fisheye settings
            const fisheyeSettings = {
                enabled: settings.fisheye.enabled,
                strength: settings.fisheye.strength,
                radius: settings.fisheye.radius,
                focusPoint: [
                    settings.fisheye.focusX,
                    settings.fisheye.focusY,
                    settings.fisheye.focusZ
                ]
            };
            this.effectsManager.updateFisheye(fisheyeSettings);
        }
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

        window.removeEventListener('resize', this.onWindowResize);

        this.nodeManager.dispose();
        this.effectsManager.dispose();
        
        if (this.xrLabelManager) {
            this.xrLabelManager.dispose();
        }

        if (this.xrControllers) {
            this.xrControllers.forEach(controller => {
                if (controller) {
                    this.scene.remove(controller);
                }
            });
        }

        if (this.xrHands) {
            this.xrHands.forEach(hand => {
                if (hand) {
                    this.scene.remove(hand);
                }
            });
        }

        this.renderer.dispose();
        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }
}
