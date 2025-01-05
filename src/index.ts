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
        // Initialize logger first
        initializeLogger();
        
        this.logger.debug('Initializing GraphVisualization with settings:', initialSettings);
        this.settingsStore = SettingsStore.getInstance();
        this.settings = initialSettings || defaultSettings;
        this.graphDataManager = GraphDataManager.getInstance();
        
        if (!this.validateSettings()) {
            throw new Error('Invalid settings configuration');
        }

        // Initialize Three.js components
        this.scene = new Scene();
        this.camera = new PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.renderer = new WebGLRenderer({ 
            antialias: true,
            alpha: true,
            powerPreference: "high-performance"
        });
        this.clock = new Clock();

        // Setup renderer
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000000, 0);
        document.body.appendChild(this.renderer.domElement);

        // Setup camera
        this.camera.position.z = 5;

        // Setup event listeners
        window.addEventListener('resize', this.onWindowResize.bind(this));

        // Load graph data
        this.loadGraphData();
    }

    private validateSettings(): boolean {
        if (!this.settings?.visualization?.hologram) {
            console.error('Missing hologram visualization settings');
            return false;
        }
        if (!this.settings?.nodes?.material) {
            console.error('Missing node material settings');
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
            await this.graphDataManager.loadInitialGraphData();
            this.isDataLoaded = true;
            this.logger.debug('Graph data loaded successfully');

            // Create HologramManager after data is loaded
            this.hologramManager = new HologramManager(this.settings, this.scene);
            this.logger.debug('HologramManager created');

            requestAnimationFrame(this.animate);
        } catch (error) {
            this.logger.error('Error loading graph data:', error);
        }
    }

    private animate = (): void => {
        if (!this.isDataLoaded) {
            this.logger.warn('Graph data not loaded yet. Skipping animation frame.');
            requestAnimationFrame(this.animate);
            return;
        }

        const delta = this.clock.getDelta();

        // Update scene
        if (this.hologramManager) {
            this.hologramManager.update(delta);
        } else {
            console.warn('HologramManager not initialized yet.');
        }

        // Render
        if (this.renderer && this.scene && this.camera) {
            try {
                this.renderer.render(this.scene, this.camera);
            } catch (error) {
                console.error('Render error:', error);
            }
        }

        requestAnimationFrame(this.animate);
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
} 