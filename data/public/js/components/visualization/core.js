// Previous imports remain the same...
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { NodeManager } from './nodes.js';
import { EffectsManager } from './effects.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';
import { initXRSession, addXRButton } from '../../xr/xrSetup.js';
import { initXRInteraction } from '../../xr/xrInteraction.js';

// Constants remain the same...
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

        // Initialize the scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.matrixAutoUpdate = true;
        this.camera.position.set(0, 1.6, 3);

        // Initialize renderer with WebGL2 check
        try {
            const canvas = document.createElement('canvas');
            const contextAttributes = {
                alpha: true,
                antialias: true,
                powerPreference: "high-performance",
                failIfMajorPerformanceCaveat: false
            };

            // Try WebGL 2 first
            let gl = canvas.getContext('webgl2', contextAttributes);
            let isWebGL2 = !!gl;

            if (!gl) {
                // Fallback to WebGL 1
                console.warn('WebGL 2 not available, falling back to WebGL 1');
                gl = canvas.getContext('webgl', contextAttributes) ||
                     canvas.getContext('experimental-webgl', contextAttributes);
                isWebGL2 = false;
            }

            if (!gl) {
                throw new Error('WebGL not supported');
            }

            this.renderer = new THREE.WebGLRenderer({
                canvas: canvas,
                context: gl,
                antialias: true,
                alpha: true,
                logarithmicDepthBuffer: true,
                powerPreference: "high-performance"
            });

            // Configure renderer based on capabilities
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = isWebGL2 ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;

            // Enable XR if available
            if (navigator.xr) {
                this.renderer.xr.enabled = true;
            } else {
                console.warn('WebXR not supported');
            }

            console.log(`Renderer initialized with ${isWebGL2 ? 'WebGL 2' : 'WebGL 1'}`);
        } catch (error) {
            console.error('Failed to initialize renderer:', error);
            throw error;
        }

        // Rest of the constructor remains the same...
        // Initialize managers
        console.log('Initializing NodeManager');
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());
        
        // Initialize effects manager
        console.log('Initializing EffectsManager');
        try {
            this.effectsManager = new EffectsManager(this.scene, this.camera, this.renderer);
            this.effectsEnabled = true;
        } catch (error) {
            console.error('Failed to initialize effects manager:', error);
            this.effectsManager = null;
            this.effectsEnabled = false;
        }

        this.controls = null;
        this.xrSessionManager = null;
        this.xrControllers = [];
        this.xrHands = [];
        this.xrLabelManager = null;
        this.isXRActive = false;

        // Bind methods
        this.onWindowResize = this.onWindowResize.bind(this);
        this.animate = this.animate.bind(this);
        this.updateVisualization = this.updateVisualization.bind(this);
        this.fallbackRender = this.fallbackRender.bind(this);
        this.handleSpacemouseInput = this.handleSpacemouseInput.bind(this);

        // Initialize settings and add event listeners
        this.initializeSettings();
        this.setupEventListeners();

        // Request initial graph data
        console.log('Requesting initial graph data');
        const initialData = this.graphDataManager.getGraphData();
        if (initialData && Array.isArray(initialData.nodes)) {
            console.log('Initial graph data available:', initialData);
            this.updateVisualization(initialData);
        } else {
            console.log('No initial graph data, waiting for updates');
            // Request data from websocket
            if (this.graphDataManager.websocketService) {
                console.log('Requesting data from websocket');
                this.graphDataManager.websocketService.send({ type: 'getInitialData' });
            }
        }

        console.log('WebXRVisualization constructor completed');
    }

    onWindowResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            if (this.effectsManager && this.effectsEnabled) {
                try {
                    this.effectsManager.handleResize();
                } catch (error) {
                    console.error('Error handling effects resize:', error);
                    this.effectsEnabled = false;
                }
            }
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

        window.addEventListener('visualizationSettingsUpdated', (event) => {
            console.log('Received settings update:', event.detail);
            this.updateSettings(event.detail);
        });

        window.addEventListener('positionUpdate', (event) => {
            if (this.graphDataManager.isGraphDataValid() && this.graphDataManager.websocketService) {
                console.log('Sending position update to server');
                this.graphDataManager.websocketService.send(event.detail);
            }
        });

        window.addEventListener('binaryPositionUpdate', (event) => {
            if (event.detail) {
                console.log('Applying binary position update');
                this.applyPositionUpdate(event.detail);
            }
        });

        // XR session state listeners
        if (this.renderer.xr) {
            this.renderer.xr.addEventListener('sessionstart', () => {
                console.log('XR session started');
                this.isXRActive = true;
                this.initializeXREffects();
            });

            this.renderer.xr.addEventListener('sessionend', () => {
                console.log('XR session ended');
                this.isXRActive = false;
                this.initializeDesktopEffects();
            });
        }
    }

    async initializeDesktopEffects() {
        if (this.effectsManager) {
            try {
                await this.effectsManager.initPostProcessing(false);
                this.effectsEnabled = true;
            } catch (error) {
                console.error('Failed to initialize desktop effects:', error);
                this.effectsEnabled = false;
            }
        }
    }

    async initializeXREffects() {
        if (this.effectsManager) {
            try {
                await this.effectsManager.initPostProcessing(true);
                this.effectsEnabled = true;
            } catch (error) {
                console.error('Failed to initialize XR effects:', error);
                this.effectsEnabled = false;
            }
        }
    }

    fallbackRender() {
        const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
        this.renderer.render(this.scene, currentCamera);
    }

    applyPositionUpdate(buffer) {
        try {
            const dataView = new Float32Array(buffer);
            const isInitialLayout = dataView[0] === 1.0;
            
            for (let i = 1; i < dataView.length; i += 6) {
                const nodeId = Math.floor((i - 1) / 6);
                const x = dataView[i];
                const y = dataView[i + 1];
                const z = dataView[i + 2];
                
                const mesh = this.nodeManager.nodeMeshes.get(nodeId);
                if (mesh) {
                    mesh.position.set(x, y, z);
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
        
        this.fogDensity = envSettings.fogDensity;
        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        
        this.ambientLightIntensity = 50;
        this.directionalLightIntensity = 5.0;
        this.directionalLightColor = 0xffffff;
        this.ambientLightColor = 0x404040;
        
        this.ambientLight = new THREE.AmbientLight(this.ambientLightColor, this.ambientLightIntensity);
        this.scene.add(this.ambientLight);

        this.directionalLight = new THREE.DirectionalLight(
            this.directionalLightColor,
            this.directionalLightIntensity
        );
        this.directionalLight.position.set(5, 5, 5);
        this.directionalLight.castShadow = true;
        this.scene.add(this.directionalLight);

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

        this.renderer.domElement.style.position = 'absolute';
        this.renderer.domElement.style.top = '0';
        this.renderer.domElement.style.left = '0';
        this.renderer.domElement.style.zIndex = '0';
        container.appendChild(this.renderer.domElement);

        const controlsContainer = document.createElement('div');
        controlsContainer.style.position = 'absolute';
        controlsContainer.style.top = '0';
        controlsContainer.style.left = '0';
        controlsContainer.style.width = '100%';
        controlsContainer.style.height = '100%';
        controlsContainer.style.zIndex = '1';
        container.appendChild(controlsContainer);

        this.controls = new OrbitControls(this.camera, controlsContainer);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.rotateSpeed = 0.8;
        this.controls.panSpeed = 0.8;
        this.controls.zoomSpeed = 0.8;

        const updatePointerEvents = (isInteracting) => {
            if (!this.renderer.xr.isPresenting) {
                this.renderer.domElement.style.pointerEvents = isInteracting ? 'auto' : 'none';
                controlsContainer.style.pointerEvents = isInteracting ? 'auto' : 'none';
            }
        };

        container.addEventListener('mouseenter', () => updatePointerEvents(true));
        container.addEventListener('mouseleave', () => updatePointerEvents(false));
        
        this.nodeManager.initClickHandling(this.renderer);

        // Initialize desktop effects first
        await this.initializeDesktopEffects();

        // Initialize XR support after effects
        this.xrSessionManager = await initXRSession(this.renderer, this.scene, this.camera);

        window.addEventListener('resize', this.onWindowResize);

        this.animate();
        
        if (this.xrSessionManager) {
            await addXRButton(this.xrSessionManager);
        }
    }

    animate() {
        const renderFrame = (timestamp, frame) => {
            if (this.controls && this.controls.enabled && !this.isXRActive) {
                this.controls.update();
            }

            this.nodeManager.updateLabelOrientations(this.camera);

            // Try effects rendering first, fallback to normal if needed
            if (this.effectsManager && this.effectsEnabled) {
                try {
                    this.effectsManager.render();
                } catch (error) {
                    console.error('Error in effects rendering:', error);
                    this.effectsEnabled = false;
                    this.fallbackRender();
                }
            } else {
                this.fallbackRender();
            }
        };

        this.renderer.setAnimationLoop(renderFrame);
    }

    updateVisualization(graphData) {
        if (!this.nodeManager || !graphData) {
            console.warn('Cannot update visualization: missing manager or data');
            return;
        }

        console.log(`Updating visualization with ${graphData.nodes?.length || 0} nodes and ${graphData.edges?.length || 0} edges`);

        if (Array.isArray(graphData.nodes)) {
            console.log('Updating nodes');
            this.nodeManager.updateNodes(graphData.nodes);
        }
        
        if (Array.isArray(graphData.edges)) {
            console.log('Updating edges');
            this.nodeManager.updateEdges(graphData.edges);
        }
    }

    updateSettings(settings) {
        console.log('Updating visualization settings:', settings);
        
        if (settings.visual) {
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
            
            if (this.scene.fog && settings.visual.fogDensity !== undefined) {
                this.scene.fog.density = settings.visual.fogDensity;
            }
        }

        if (settings.material) {
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

        if (settings.bloom && this.effectsEnabled) {
            try {
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
            } catch (error) {
                console.error('Error updating bloom settings:', error);
                this.effectsEnabled = false;
            }
        }

        if (settings.fisheye && this.effectsEnabled) {
            try {
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
            } catch (error) {
                console.error('Error updating fisheye settings:', error);
                this.effectsEnabled = false;
            }
        }
    }

    handleSpacemouseInput(x, y, z) {
        if (!this.camera || this.renderer.xr.isPresenting) return;

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
        this.renderer.setAnimationLoop(null);

        window.removeEventListener('resize', this.onWindowResize);

        this.nodeManager.dispose();
        if (this.effectsManager) {
            this.effectsManager.dispose();
        }
        
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
