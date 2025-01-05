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
        
        this.logger.debug('Initializing GraphVisualization with settings:', {
            initialSettings,
            defaultSettings,
            hasHologramSettings: !!initialSettings?.visualization?.hologram,
            hologramEnabled: initialSettings?.visualization?.hologram?.enabled
        });
        
        try {
            this.settingsStore = SettingsStore.getInstance();
            this.settings = initialSettings || defaultSettings;
            
            // Log merged settings
            this.logger.debug('Using settings:', {
                hasHologram: !!this.settings?.visualization?.hologram,
                hologramEnabled: this.settings?.visualization?.hologram?.enabled,
                fullSettings: this.settings
            });
            
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

            this.logger.debug('Creating WebGL renderer');
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

            this.logger.debug('GraphVisualization initialized successfully');

            // Load graph data
            this.loadGraphData();
        } catch (error) {
            this.logger.error('Error in GraphVisualization constructor:', error);
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
        const frameId = Math.random();
        this.logger.debug(`Animation frame ${frameId} started`);

        if (!this.isDataLoaded) {
            this.logger.warn(`Frame ${frameId}: Graph data not loaded yet. Skipping.`);
            requestAnimationFrame(this.animate);
            return;
        }

        try {
            const delta = this.clock.getDelta();

            if (this.hologramManager) {
                this.hologramManager.update(delta);
            }

            // Try rendering
            try {
                if (this.renderer && this.scene && this.camera) {
                    this.renderer.render(this.scene, this.camera);
                }
            } catch (renderError) {
                this.logger.error(`Frame ${frameId}: Render error:`, {
                    error: renderError,
                    scene: {
                        children: this.scene?.children.length,
                        camera: !!this.camera,
                        renderer: !!this.renderer
                    }
                });
            }

            requestAnimationFrame(this.animate);
        } catch (error) {
            this.logger.error(`Frame ${frameId}: Animation error:`, error);
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
} 