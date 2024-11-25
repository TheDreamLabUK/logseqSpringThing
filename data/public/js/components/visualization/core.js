import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { NodeManager } from './nodes.js';
import { EffectsManager } from './effects.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';
import { initXRSession, addXRButton } from '../../xr/xrSetup.js';
import { initXRInteraction } from '../../xr/xrInteraction.js';

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

        // Initialize the scene with a dark gray background for better contrast
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);
        
        // Create camera with optimized initial position for graph viewing
        this.camera = new THREE.PerspectiveCamera(
            50, // Narrower FOV for better depth perception
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        // Position camera further back for better overview of large graphs
        this.camera.position.set(0, 15, 50);
        this.camera.lookAt(0, 0, 0);

        // Create and initialize canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;

        // Initialize renderer with WebGL2 and HDR support
        try {
            const contextAttributes = {
                alpha: false,
                antialias: true,
                powerPreference: "high-performance",
                failIfMajorPerformanceCaveat: false,
                preserveDrawingBuffer: true // Important for texture updates
            };

            // Try WebGL 2 first for better performance
            let gl = this.canvas.getContext('webgl2', contextAttributes);
            let isWebGL2 = !!gl;

            if (!gl) {
                console.warn('WebGL 2 not available, falling back to WebGL 1');
                gl = this.canvas.getContext('webgl', contextAttributes) ||
                     this.canvas.getContext('experimental-webgl', contextAttributes);
                isWebGL2 = false;
            }

            if (!gl) {
                throw new Error('WebGL not supported');
            }

            // Create renderer with optimized settings
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                context: gl,
                antialias: true,
                alpha: false,
                logarithmicDepthBuffer: true,
                powerPreference: "high-performance",
                preserveDrawingBuffer: true
            });

            // Configure renderer for HDR and optimal quality
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = isWebGL2 ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
            this.renderer.setClearColor(0x111111, 1);
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1.2;

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

        // Initialize scene container with proper styling
        const container = document.getElementById('scene-container');
        if (container) {
            container.style.position = 'absolute';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100%';
            container.style.height = '100%';
            container.style.overflow = 'hidden';
            container.style.backgroundColor = '#111111';
            
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            container.appendChild(this.canvas);
        }

        // Initialize texture loader with proper settings
        this.textureLoader = new THREE.TextureLoader();
        this.textureLoader.crossOrigin = 'anonymous';

        // Initialize managers
        console.log('Initializing NodeManager');
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());
        
        // Initialize effects manager with error handling
        console.log('Initializing EffectsManager');
        try {
            this.effectsManager = new EffectsManager(this.scene, this.camera, this.renderer);
            this.effectsEnabled = true;
        } catch (error) {
            console.error('Failed to initialize effects manager:', error);
            this.effectsManager = null;
            this.effectsEnabled = false;
        }

        // Initialize controls with optimized settings
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.rotateSpeed = 0.8;
        this.controls.panSpeed = 0.8;
        this.controls.zoomSpeed = 0.8;
        this.controls.target.set(0, 0, 0);

        // XR-related properties
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

        // Initialize scene settings and start animation
        this.initializeSettings();
        this.setupEventListeners();
        this.animate();

        console.log('WebXRVisualization constructor completed');
    }

    initializeSettings() {
        console.log('Initializing settings');
        
        // Add a grid helper for spatial reference
        const gridHelper = new THREE.GridHelper(100, 100, 0x555555, 0x282828);
        gridHelper.position.y = -10; // Move grid down for better perspective
        this.scene.add(gridHelper);

        // Add ambient light for base illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Add directional light for shadows and highlights
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        
        // Increase shadow map size for better quality
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 200;
        directionalLight.shadow.bias = -0.0001;
        
        this.scene.add(directionalLight);

        // Add point lights for better illumination
        const pointLight1 = new THREE.PointLight(0xffffff, 0.3, 200);
        pointLight1.position.set(50, 50, 50);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xffffff, 0.3, 200);
        pointLight2.position.set(-50, -50, -50);
        this.scene.add(pointLight2);

        // Add very subtle fog for depth
        const envSettings = visualizationSettings.getEnvironmentSettings();
        this.scene.fog = new THREE.FogExp2(0x111111, envSettings.fogDensity || 0.001);
    }

    async initThreeJS() {
        console.log('Initializing Three.js');
        const container = document.getElementById('scene-container');
        if (!container) {
            console.error("Could not find 'scene-container' element");
            return;
        }

        // Initialize controls if not already initialized
        if (!this.controls) {
            this.controls = new OrbitControls(this.camera, this.canvas);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.1;
            this.controls.rotateSpeed = 0.8;
            this.controls.panSpeed = 0.8;
            this.controls.zoomSpeed = 0.8;
        }
        
        this.nodeManager.initClickHandling(this.renderer);

        // Initialize desktop effects first
        await this.initializeDesktopEffects();

        // Initialize XR support after effects
        this.xrSessionManager = await initXRSession(this.renderer, this.scene, this.camera);

        window.addEventListener('resize', this.onWindowResize);
        
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
            
            // Update effects if enabled
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

    fallbackRender() {
        const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
        // Clear the scene before rendering
        this.renderer.clear(true, true, true);
        // Render the scene directly without effects
        this.renderer.render(this.scene, currentCamera);
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

    applyPositionUpdate(update) {
        if (!this.nodeManager) {
            console.warn('Cannot apply position update: NodeManager not initialized');
            return;
        }

        try {
            // Assuming update is a binary buffer containing node positions
            if (update instanceof ArrayBuffer) {
                const positions = new Float32Array(update);
                this.nodeManager.updateNodePositions(positions);
            } else {
                console.warn('Invalid position update format:', update);
        }
        } catch (error) {
            console.error('Error applying position update:', error);
        }
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        this.renderer.setAnimationLoop(null);

        window.removeEventListener('resize', this.onWindowResize);

        // Dispose of textures
        if (this.textureLoader) {
            THREE.Cache.clear(); // Clear texture cache
        }

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

        // Clean up canvas
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }

        this.renderer.dispose();
        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }
}
