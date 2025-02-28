import {
  Scene,
  PerspectiveCamera,
  // WebGLRenderer,
  Vector3,
  // Quaternion,
  // Matrix4,
  // Group,
  // Mesh,
  // MeshBasicMaterial,
  // Color,
  // Raycaster,
  // Object3D,
  // Euler,
  // MathUtils,
} from 'three';
import { createLogger, createErrorMetadata, createDataMetadata } from '../core/logger';
import { Settings } from '../types/settings/base';
import { defaultSettings } from '../state/defaultSettings';
import { XRHandWithHaptics } from '../types/xr';
import { EdgeManager } from './EdgeManager';
import { NodeManagerFacade } from './node/NodeManagerFacade';
import { graphDataManager } from '../state/graphData';
import { MetadataVisualizer } from './MetadataVisualizer';
import { GraphData } from '../core/types';
import { WebSocketService } from '../websocket/websocketService';
import { NodeMetadata } from '../types/metadata';
import { MaterialFactory } from './factories/MaterialFactory';
// import { SettingsStore } from '../state/SettingsStore'; // Commented out as it's unused
// import { debugState } from '../core/debugState'; // Commented out as it's unused
import { GraphDataManager } from '../state/graphData';
import { Node } from '../core/types';

const logger = createLogger('VisualizationController');

type VisualizationCategory = 'visualization' | 'physics' | 'rendering';
type PendingUpdate = { category: VisualizationCategory; value: any };

export class VisualizationController {
    private static instance: VisualizationController | null = null;
    private currentSettings: Settings;
    private edgeManager: EdgeManager | null = null;
    private nodeManager: NodeManagerFacade | null = null;
    private metadataVisualizer: MetadataVisualizer | null = null;
    private isInitialized: boolean = false;
    private pendingUpdates: Map<string, PendingUpdate> = new Map();
    private lastUpdateTime: number = performance.now();
    private websocketService: WebSocketService;

    private constructor() {
        // Initialize with complete default settings
        this.currentSettings = defaultSettings;
        this.websocketService = WebSocketService.getInstance();
        
        // Subscribe to graph data updates
        graphDataManager.subscribe((data: GraphData) => {
            if (this.isInitialized) {
                if (this.nodeManager) {
                    this.nodeManager.updateNodes(data.nodes);
                }
                if (this.edgeManager) {
                    this.edgeManager.updateEdges(data.edges);
                }
            } else {
                // Queue updates until initialized
                if (import.meta.env.DEV) logger.debug('Queuing updates until initialization');
            }
        });

        // Subscribe to websocket binary updates
        this.websocketService.onBinaryMessage((nodes) => {
            if (this.nodeManager && this.isInitialized) {
                // Convert binary node data to the format expected by updateNodePositions
                const updates = nodes.map(node => ({
                    id: node.id.toString(),
                    data: {
                        position: node.position,
                        velocity: node.velocity
                    }
                }));
                this.nodeManager.updateNodePositions(updates);
            }
        });
    }

    public initializeScene(scene: Scene, camera: PerspectiveCamera): void {
        logger.info('Initializing visualization scene');
        
        // Ensure camera can see nodes
        camera.layers.enable(0);
        logger.debug('Camera layers configured');
        
        // Enable WebSocket debugging
        this.currentSettings.system.debug.enabled = true;
        this.currentSettings.system.debug.enableWebsocketDebug = true;
        
        // Connect to websocket first
        this.websocketService.connect().then(() => {
            logger.info('WebSocket connected, enabling binary updates');
            graphDataManager.enableBinaryUpdates();
            
            // Send initial request for data
            this.websocketService.sendMessage({ 
                type: 'requestInitialData',
                timestamp: Date.now()
            });
            
            // Initially disable randomization until data loading is finished
            this.websocketService.sendMessage({
                type: 'enableRandomization',
                enabled: false
            });
        }).catch(error => {
            logger.error('Failed to connect WebSocket:', createErrorMetadata(error));
        });
        
        const materialFactory = MaterialFactory.getInstance();
        this.nodeManager = NodeManagerFacade.getInstance(
            scene,
            camera,
            materialFactory.getNodeMaterial(this.currentSettings)
        );
        this.edgeManager = new EdgeManager(scene, this.currentSettings, this.nodeManager.getNodeInstanceManager());
        this.metadataVisualizer = new MetadataVisualizer(camera, scene, this.currentSettings);
        this.isInitialized = true;
        
        if (import.meta.env.DEV) logger.debug('Scene managers initialized');

        // Initialize with current graph data (if any)
        const currentData = graphDataManager.getGraphData();
        if (currentData.nodes.length > 0 && this.nodeManager) {
            this.nodeManager.updateNodes(currentData.nodes);
        }

        // Start animation loop
        this.animate();

        logger.info('Scene initialization complete');
    }

    public static getInstance(): VisualizationController {
        if (!VisualizationController.instance) {
            VisualizationController.instance = new VisualizationController();
        }
        return VisualizationController.instance;
    }

    public updateSetting(path: string, value: any): void {
        const parts = path.split('.');
        const category = parts[0] as VisualizationCategory;
        
        if (!['visualization', 'physics', 'rendering'].includes(category)) {
            return;
        }

        if (!this.isInitialized) {
            logger.debug(`Queuing setting update for ${path}`);
            this.pendingUpdates.set(path, { category, value });
            return;
        }

        let current = this.currentSettings as any;
        for (let i = 0; i < parts.length - 1; i++) {
            const part = parts[i];
            if (!(part in current)) {
                current[part] = {};
            }
            current = current[part];
        }

        current[parts[parts.length - 1]] = value;
        this.applySettingUpdate(category);
    }

    public updateSettings(category: VisualizationCategory, settings: Partial<Settings>): void {
        if (!this.isInitialized) {
            logger.debug(`Queuing bulk settings update for ${category}`);
            this.pendingUpdates.set(category, { category, value: settings });
            return;
        }

        switch (category) {
            case 'visualization':
                if (settings.visualization) {
                    this.currentSettings.visualization = {
                        ...this.currentSettings.visualization,
                        ...settings.visualization
                    };
                    this.applyVisualizationUpdates();
                }
                break;
            case 'physics':
                if (settings.visualization?.physics) {
                    this.currentSettings.visualization.physics = {
                        ...this.currentSettings.visualization.physics,
                        ...settings.visualization.physics
                    };
                    this.updatePhysicsSimulation();
                }
                break;
            case 'rendering':
                if (settings.visualization?.rendering) {
                    this.currentSettings.visualization.rendering = {
                        ...this.currentSettings.visualization.rendering,
                        ...settings.visualization.rendering
                    };
                    this.updateRenderingQuality();
                }
                break;
        }
    }

    public getSettings(category: VisualizationCategory): Partial<Settings> {
        const baseVisualization = {
            nodes: { ...this.currentSettings.visualization.nodes },
            edges: { ...this.currentSettings.visualization.edges },
            physics: { ...this.currentSettings.visualization.physics },
            rendering: { ...this.currentSettings.visualization.rendering },
            animations: { ...this.currentSettings.visualization.animations },
            labels: { ...this.currentSettings.visualization.labels },
            bloom: { ...this.currentSettings.visualization.bloom },
            hologram: { ...this.currentSettings.visualization.hologram }
        };

        switch (category) {
            case 'visualization':
                return {
                    visualization: { ...this.currentSettings.visualization }
                };
            case 'physics':
                return {
                    visualization: {
                        ...baseVisualization,
                        physics: { ...this.currentSettings.visualization.physics }
                    }
                };
            case 'rendering':
                return {
                    visualization: {
                        ...baseVisualization,
                        rendering: { ...this.currentSettings.visualization.rendering }
                    }
                };
            default:
                return {
                    visualization: baseVisualization
                };
        }
    }

    public handleHandInput(hand: XRHandWithHaptics): void {
        if (!this.isInitialized || !hand) return;

        const pinchStrength = hand.pinchStrength || 0;
        const gripStrength = hand.gripStrength || 0;

        if (pinchStrength > (this.currentSettings.xr.pinchThreshold || 0.5)) {
            logger.debug('Pinch gesture detected', { strength: pinchStrength });
        }

        if (gripStrength > (this.currentSettings.xr.dragThreshold || 0.5)) {
            logger.debug('Grip gesture detected', { strength: gripStrength });
        }

        if (hand.hand?.joints) {
            logger.debug('Processing hand joints');
        }
    }

    private applySettingUpdate(category: VisualizationCategory): void {
        if (!this.isInitialized) {
            logger.debug(`Queuing category update for ${category}`);
            return;
        }

        logger.debug(`Updating ${category} settings`);
        
        switch (category) {
            case 'visualization':
                this.applyVisualizationUpdates();
                break;
            case 'physics':
                this.updatePhysicsSimulation();
                break;
            case 'rendering':
                this.updateRenderingQuality();
                break;
        }
    }

    private applyVisualizationUpdates(): void {
        if (!this.isInitialized) return;
        this.updateNodeAppearance();
        this.updateEdgeAppearance();
        // Update metadata visualization
        if (this.metadataVisualizer) {
            this.updateMetadataVisualization();
        }
    }

    private updateNodeAppearance(): void {
        if (!this.isInitialized) return;
        logger.debug('Updating node appearance');
        if (this.nodeManager) {
            this.nodeManager.handleSettingsUpdate(this.currentSettings);
        }
    }

    private updateEdgeAppearance(): void {
        if (!this.isInitialized) {
            logger.debug('Queuing edge appearance update');
            return;
        }

        if (this.edgeManager) {
            this.edgeManager.handleSettingsUpdate(this.currentSettings);
            logger.debug('Edge appearance updated');
        } else {
            logger.warn('EdgeManager not initialized');
        }
    }

    private updatePhysicsSimulation(): void {
        if (!this.isInitialized) return;
        logger.debug('Updating physics simulation');
    }

    private updateRenderingQuality(): void {
        if (!this.isInitialized) return;
        logger.debug('Updating rendering quality');
    }

    public updateNodePositions(nodes: any[]): void {
        if (this.nodeManager) {
            this.nodeManager.updateNodePositions(nodes);
        }
    }

    /**
     * Randomly distributes all nodes in 3D space and triggers WebSocket updates
     * @param radius The radius of the sphere within which to distribute nodes
     */
    public randomizeNodePositions(radius: number = 7): void {
        // Enforce maximum radius to prevent explosion
        radius = Math.min(radius, 5); // Reduced maximum radius to prevent explosion

        if (!this.nodeManager || !this.isInitialized) {
            logger.warn('Cannot randomize node positions - Node manager not initialized');
            return;
        }

        logger.info('Randomizing node positions with radius:', createDataMetadata({ radius }));
        
        // Get node data from the graph manager
        const graphData = GraphDataManager.getInstance().getGraphData();
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            logger.warn('No nodes found to randomize');
            return;
        }
        
        // Create node updates with random positions
        const updates = graphData.nodes.map((node: Node) => {
            // Generate random position within a sphere
            const theta = Math.random() * Math.PI * 2; // Random angle around Y axis
            const phi = Math.acos((Math.random() * 2) - 1); // Random angle from Y axis
            // Use square root rather than cube root for more central clustering
            const r = radius * Math.sqrt(Math.random()); // Random distance from center
            
            // Convert spherical to Cartesian coordinates
            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            // Use near-zero initial velocities to prevent explosion
            const vx = 0;
            const vy = 0;
            const vz = (Math.random() - 0.5) * 0.01 - (z * 0.001);
            
            return {
                id: node.id,
                data: {
                    position: new Vector3(x, y, z),
                    velocity: new Vector3(vx, vy, vz)
                }
            };
        });
        
        // Update local node positions
        this.nodeManager.updateNodePositions(updates);
        
        // Send updates to server via WebSocket
        if (this.websocketService) {
            updates.forEach((update) => {
                // Send updates to server one at a time with minimal velocity
                this.websocketService.sendNodeUpdates([{ 
                    id: update.id,
                    position: update.data.position,
                    velocity: update.data.velocity
                }]);
            });
            
            // Enable server-side randomization to ensure physics simulation applies
            // Removed enableRandomization toggle to prevent physics instability
        } else {
            logger.warn('WebSocket service not available, node positions only updated locally');
        }
    }

    private animate = (): void => {
        if (!this.isInitialized) return;

        requestAnimationFrame(this.animate);
        const currentTime = performance.now();
        const deltaTime = (currentTime - this.lastUpdateTime) / 1000;
        this.update(deltaTime);
    }

    public update(deltaTime: number): void {
        if (this.isInitialized) {
            const currentTime = performance.now();
            if (deltaTime === 0) {
                deltaTime = (currentTime - this.lastUpdateTime) / 1000;
            }
            this.lastUpdateTime = currentTime;
            if (this.nodeManager) {
                this.nodeManager.update(deltaTime);
            }

            // Update edge animations
            if (this.edgeManager) {
                this.edgeManager.update(deltaTime);
            }
            
            // Update metadata visualization
            if (this.metadataVisualizer) {
                this.updateMetadataVisualization();
            }
        }
    }

    public dispose(): void {
        // Dispose of managers and cleanup websocket
        if (this.metadataVisualizer) {
            this.metadataVisualizer?.dispose();
            this.metadataVisualizer = null;
        }
        this.nodeManager?.dispose();
        this.edgeManager?.dispose();
        this.websocketService.dispose();
        this.isInitialized = false;
        VisualizationController.instance = null;
    }

    private updateMetadataVisualization(): void {
        if (!this.isInitialized || !this.metadataVisualizer || !this.nodeManager) return;
        
        const currentData = graphDataManager.getGraphData();
        currentData.nodes.forEach(node => {
            if (node.data?.metadata) {
                const metadata: NodeMetadata = {
                    id: node.id,
                    name: node.data.metadata.name || 'Unnamed',
                    commitAge: Math.floor((Date.now() - (node.data.metadata.lastModified || Date.now())) / (1000 * 60 * 60 * 24)),
                    hyperlinkCount: node.data.metadata.hyperlinkCount || 0,
                    fileSize: node.data.metadata.fileSize || 0,
                    nodeSize: Math.min(50, Math.max(1, Math.log10((node.data.metadata.fileSize || 1024) / 1024) * 10)), // Scale based on file size (1-50)
                    importance: 1.0, // Default importance
                    position: {
                        x: node.data.position.x || 0,
                        y: node.data.position.y || 0,
                        z: node.data.position.z || 0
                    }
                };
                this.metadataVisualizer?.createMetadataLabel(metadata, node.id);
                const position = this.nodeManager?.getNodeInstanceManager().getNodePosition(node.id);
                if (position) {
                    this.metadataVisualizer?.updateMetadataPosition(node.id, position);
                }
            }
        });
    }
}
