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
                preserveDrawingBuffer: true,
                xrCompatible: true // Enable XR compatibility
            };

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

            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                context: gl,
                antialias: true,
                alpha: false,
                logarithmicDepthBuffer: true,
                powerPreference: "high-performance",
                preserveDrawingBuffer: true
            });

            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = isWebGL2 ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
            this.renderer.setClearColor(0x222222, 1);
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1.2;

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
            container.style.backgroundColor = '#222222'; // Match renderer background
            
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            container.appendChild(this.canvas);
        }

        // Initialize managers
        console.log('Initializing NodeManager');
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());

        // Initialize controls with optimized settings
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.rotateSpeed = 0.8;
        this.controls.panSpeed = 0.8;
        this.controls.zoomSpeed = 0.8;
        this.controls.target.set(0, 0, 0);

        // Initialize XR
        this.xrSessionManager = null;
        this.initializeXR();

        // Bind methods
        this.onWindowResize = this.onWindowResize.bind(this);
        this.animate = this.animate.bind(this);
        this.updateVisualization = this.updateVisualization.bind(this);
        this.handleSpacemouseInput = this.handleSpacemouseInput.bind(this);

        // Initialize scene settings and start animation
        this.initializeSettings();
        this.setupEventListeners();
        this.animate();

        console.log('WebXRVisualization constructor completed');
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
                
                // Set up XR animation loop
                this.renderer.setAnimationLoop((timestamp, frame) => {
                    // Update XR session
                    if (this.xrSessionManager) {
                        this.xrSessionManager.update(timestamp, frame);
                    }

                    // Update controls and labels
                    if (this.controls && !this.renderer.xr.isPresenting) {
                        this.controls.update();
                    }
                    if (this.nodeManager) {
                        this.nodeManager.updateLabelOrientations(this.camera);
                    }

                    // Render scene
                    this.renderer.render(this.scene, this.camera);
                });

                console.log('XR initialization complete');
            } else {
                console.warn('XR not supported or initialization failed');
            }
        } catch (error) {
            console.error('Error initializing XR:', error);
        }
    }

    initializeSettings() {
        console.log('Initializing settings');
        
        // Add strong ambient light for base illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
        this.scene.add(ambientLight);

        // Add directional light for shadows and highlights
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
        directionalLight.position.set(5, 10, 5);
        this.scene.add(directionalLight);

        // Add hemisphere light for better ambient illumination
        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
        this.scene.add(hemisphereLight);

        // Add point lights for better illumination
        const pointLight1 = new THREE.PointLight(0xffffff, 0.8, 100);
        pointLight1.position.set(20, 20, 20);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xffffff, 0.8, 100);
        pointLight2.position.set(-20, -20, -20);
        this.scene.add(pointLight2);

        // Set very subtle fog for depth
        const envSettings = visualizationSettings.getEnvironmentSettings();
        this.scene.fog = new THREE.FogExp2(0x222222, envSettings.fogDensity || 0.0002);

        console.log('Scene settings initialized with enhanced lighting');
    }

    animate() {
        const renderFrame = () => {
            if (this.controls) {
                this.controls.update();
            }

            if (this.nodeManager) {
                this.nodeManager.updateLabelOrientations(this.camera);
            }

            // Direct rendering without effects
            this.renderer.render(this.scene, this.camera);
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

        window.addEventListener('visualizationSettingsUpdated', (event) => {
            console.log('Received settings update:', event.detail);
            this.updateSettings(event.detail);
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

    updateSettings(settings) {
        console.log('Updating visualization settings:', settings);
        
        if (settings.visual) {
            const visualSettings = {
                nodeColor: settings.visual.nodeColor,
                edgeColor: settings.visual.edgeColor,
                minNodeSize: settings.visual.minNodeSize,
                maxNodeSize: settings.visual.maxNodeSize,
                edgeOpacity: settings.visual.edgeOpacity,
                fogDensity: settings.visual.fogDensity
            };
            this.nodeManager.updateFeature(visualSettings);
            
            if (this.scene.fog && settings.visual.fogDensity !== undefined) {
                this.scene.fog.density = settings.visual.fogDensity;
            }
        }

        if (settings.material) {
            this.nodeManager.updateMaterial(settings.material);
        }
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        this.renderer.setAnimationLoop(null);

        window.removeEventListener('resize', this.onWindowResize);

        if (this.nodeManager) {
            this.nodeManager.dispose();
        }

        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }

        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }
}
