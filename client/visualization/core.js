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

        // Bind methods
        this.onWindowResize = this.onWindowResize.bind(this);
        this.animate = this.animate.bind(this);
        this.updateVisualization = this.updateVisualization.bind(this);
        this.handleSpacemouseInput = this.handleSpacemouseInput.bind(this);
        this.renderFrame = this.renderFrame.bind(this);
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);

        // Listen for settings
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
    }

    handleSettingsUpdate(event) {
        console.log('Received visualization settings update:', event.detail);
        
        if (this.pendingInitialization) {
            this.pendingInitialization = false;
            this.initializeVisualization(event.detail);
        } else {
            this.updateFromSettings(event.detail);
        }
    }

    initializeVisualization(settings) {
        if (!settings?.visualization) {
            console.error('Cannot initialize visualization without settings');
            return;
        }

        const vis = settings.visualization;

        // Initialize the scene with settings
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Create camera with settings
        this.camera = new THREE.PerspectiveCamera(
            50, // Wider FOV for better overview
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        this.camera.position.set(0, 75, 200);
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
                xrCompatible: true
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
            this.renderer.setClearColor(0x000000, 1);
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1.5;

            console.log(`Renderer initialized with ${isWebGL2 ? 'WebGL 2' : 'WebGL 1'}`);
        } catch (error) {
            console.error('Failed to initialize renderer:', error);
            throw error;
        }

        // Initialize scene container
        const container = document.getElementById('scene-container');
        if (container) {
            container.style.position = 'absolute';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100%';
            container.style.height = '100%';
            container.style.overflow = 'hidden';
            container.style.backgroundColor = '#000000';
            
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            container.appendChild(this.canvas);
        }

        // Initialize managers with settings
        console.log('Initializing NodeManager with settings');
        this.nodeManager = new NodeManager(this.scene, this.camera, visualizationSettings.getNodeSettings());

        // Initialize controls with settings
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.rotateSpeed = 0.4;
        this.controls.panSpeed = 0.6;
        this.controls.zoomSpeed = 1.2;
        this.controls.minDistance = 50;
        this.controls.maxDistance = 500;
        this.controls.target.set(0, 0, 0);

        // Initialize scene settings
        this.initializeSettings();
        this.setupEventListeners();

        // Load initial graph data if available
        const initialData = this.graphDataManager.getGraphData();
        if (initialData && initialData.nodes.length > 0) {
            console.log('Loading initial graph data');
            this.updateVisualization(initialData);
        }

        // Start animation
        this.animate();

        // Initialize XR after everything else is set up
        this.initializeXR().then(() => {
            console.log('WebXRVisualization initialization completed');
            this.initialized = true;
        });
    }

    updateFromSettings(settings) {
        if (!this.initialized) return;

        const vis = settings.visualization;
        if (!vis) return;

        // Update scene properties
        if (vis.fog_density !== undefined && this.scene.fog) {
            this.scene.fog.density = vis.fog_density;
        }

        // Update node manager
        if (this.nodeManager) {
            this.nodeManager.updateFromSettings(settings);
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

        // Update XR session if in XR mode
        if (this.xrSessionManager && this.renderer.xr.isPresenting) {
            this.xrSessionManager.update();
        }

        // Update node labels
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
            this.updateFromSettings(event.detail);
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
        this.renderer.setAnimationLoop(null);

        window.removeEventListener('resize', this.onWindowResize);
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

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
