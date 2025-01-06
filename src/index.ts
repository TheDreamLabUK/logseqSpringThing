import * as THREE from 'three';
import { Scene, PerspectiveCamera, WebGLRenderer, Clock, Mesh, Material } from 'three';
import { Settings } from './types/settings';
import { SettingsStore } from './state/SettingsStore';
import { defaultSettings } from './state/defaultSettings';
import { HologramManager } from './managers/HologramManager';
import { GraphDataManager } from './state/graphData';
import { createLogger } from '../core/logger';
import { initializeLogger } from './core/logger';

export class GraphVisualization {
    private logger = createLogger('GraphVisualization');
    private settings: Settings;
    private settingsStore: SettingsStore;
    private scene: Scene;
    private camera: PerspectiveCamera;
    private renderer: WebGLRenderer;
    private hologramManager: HologramManager;
    private clock: Clock;
    private graphDataManager: GraphDataManager;
    private isDataLoaded: boolean = false;

    constructor(initialSettings?: Settings) {
        console.log('Starting GraphVisualization initialization');
        
        try {
            this.settings = initialSettings || defaultSettings;
            console.log('Using settings:', this.settings);

            // Initialize basic Three.js components
            this.scene = new Scene();
            this.camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            this.camera.position.z = 5;
            
            // Create and setup renderer
            this.renderer = new WebGLRenderer({
                antialias: true,
                alpha: true,
                powerPreference: "high-performance"
            });
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(this.renderer.domElement);

            // Create a simple test cube
            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0xff0000,
                wireframe: true 
            });
            const cube = new THREE.Mesh(geometry, material);
            this.scene.add(cube);
            console.log('Added test cube to scene');

            // Initialize clock
            this.clock = new Clock();

            // Setup resize handler
            window.addEventListener('resize', this.onWindowResize.bind(this));

            console.log('Starting animation loop');
            requestAnimationFrame(this.animate);

        } catch (error) {
            console.error('Initialization error:', error);
            throw error;
        }
    }

    private validateSettings(): boolean {
        if (!this.settings?.visualization?.hologram) {
            this.logger.error('Missing hologram visualization settings');
            return false;
        }
        if (!this.settings?.nodes?.material) {
            this.logger.error('Missing node material settings');
            return false;
        }
        return true;
    }

    private onWindowResize(): void {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    private async loadGraphData(): Promise<void> {
        this.logger.debug('Starting to load graph data...');
        try {
            const graphData = await this.graphDataManager.loadInitialGraphData();
            this.logger.debug('Graph data loaded:', graphData);

            this.isDataLoaded = true;
            this.logger.debug('Graph data loaded successfully. Creating scene objects...');

            // Log scene state before adding holograms
            this.logger.debug('Scene state before holograms:', {
                sceneChildren: this.scene?.children.length
            });

            // Check if holograms are enabled
            if (this.settings?.visualization?.hologram?.enabled) {
                // Create HologramManager after data is loaded
                this.logger.debug('Creating HologramManager');
                this.hologramManager = new HologramManager(this.settings, this.scene);
                
                // Log scene state after adding holograms
                this.logger.debug('Scene state after holograms:', {
                    sceneChildren: this.scene?.children.length,
                    hologramGroup: this.hologramManager.getGroup()?.children.length
                });
            } else {
                this.logger.warn('Holograms are disabled in settings, skipping HologramManager creation');
            }

            this.logger.debug('Starting animation loop');
            requestAnimationFrame(this.animate);
        } catch (error) {
            this.logger.error('Error loading graph data:', error);
        }
    }

    private animate = (): void => {
        try {
            // Simple rotation of the test cube
            const cube = this.scene.children[0];
            if (cube) {
                cube.rotation.x += 0.01;
                cube.rotation.y += 0.01;
            }

            // Basic render
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }

            requestAnimationFrame(this.animate);
        } catch (error) {
            console.error('Animation error:', error);
        }
    };

    public initialize(): void {
        // Additional initialization if needed
    }

    public dispose(): void {
        // Stop animation first
        cancelAnimationFrame(this.animate as any);
        
        // Remove event listeners
        window.removeEventListener('resize', this.onWindowResize);
        
        // Dispose of Three.js resources
        this.scene.traverse((object) => {
            if (object instanceof Mesh) {
                object.geometry.dispose();
                if (object.material instanceof Material) {
                    object.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
        this.scene.clear();
        
        const canvas = this.renderer.domElement;
        canvas.parentElement?.removeChild(canvas);
    }

    private testRender(): void {
        console.log('Testing minimal render');
        
        try {
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;

            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0xff0000,
                wireframe: true  // Make it wireframe to rule out any material issues
            });
            const cube = new THREE.Mesh(geometry, material);
            scene.add(cube);
            
            // Log pre-render state
            console.log('Test render pre-state:', {
                hasScene: !!scene,
                hasCamera: !!camera,
                sceneChildren: scene.children.length,
                cameraPosition: camera.position.toArray(),
                rendererContext: !!this.renderer.getContext()
            });

            this.renderer.render(scene, camera);
            console.log('Test render successful');
            
            // Only start animation if test render succeeds
            this.isDataLoaded = true;  // Set this for now to allow animation
            requestAnimationFrame(this.animate);
        } catch (error) {
            console.error('Test render failed:', {
                error,
                stack: error.stack,
                message: error.message
            });
        }
    }

    private testShader(): void {
        const vertexShader = `
            void main() {
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            void main() {
                gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
            }
        `;

        try {
            const geometry = new THREE.PlaneGeometry(1, 1);
            const material = new THREE.ShaderMaterial({
                vertexShader,
                fragmentShader
            });
            const mesh = new THREE.Mesh(geometry, material);
            
            const scene = new THREE.Scene();
            scene.add(mesh);
            
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;
            
            this.renderer.render(scene, camera);
            console.log('Shader test successful');
        } catch (error) {
            console.error('Shader test failed:', {
                error,
                stack: error.stack,
                message: error.message
            });
        }
    }
} 