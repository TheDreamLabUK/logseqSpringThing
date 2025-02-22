import {
    Object3D,
    Camera,
    Scene,
    Vector3,
    Sprite,
    SpriteMaterial,
    Texture
} from 'three';
import { NodeMetadata } from '../../../types/metadata';
import { createLogger } from '../../../core/logger';

const logger = createLogger('NodeMetadataManager');

interface MetadataLabel {
    sprite: Sprite;
    metadata: NodeMetadata;
    lastUpdateDistance: number;
    lastVisible?: boolean;
}

export class NodeMetadataManager {
    private static instance: NodeMetadataManager;
    private labels: Map<string, MetadataLabel> = new Map();
    private VISIBILITY_THRESHOLD = 100;  // Increased maximum distance for label visibility
    private readonly UPDATE_INTERVAL = 2;        // More frequent updates
    private readonly LABEL_SCALE = 0.5;         // Base scale for labels
    private frameCount = 0;

    private worldPosition = new Vector3();
    private labelCanvas: HTMLCanvasElement;
    private labelContext: CanvasRenderingContext2D;
    private scene: Scene;

    private constructor(scene: Scene) {
        // Create canvas for label textures
        this.labelCanvas = document.createElement('canvas');
        this.labelCanvas.width = 256;
        this.labelCanvas.height = 128;
        
        const context = this.labelCanvas.getContext('2d');
        if (!context) {
            throw new Error('Failed to get 2D context for label canvas');
        }
        this.labelContext = context;
        
        // Set up default text style
        this.labelContext.textAlign = 'center';
        this.labelContext.textBaseline = 'middle';
        this.labelContext.font = 'bold 24px Arial';
        
        this.scene = scene;
    }

    public static getInstance(scene?: Scene): NodeMetadataManager {
        if (!NodeMetadataManager.instance) {
            NodeMetadataManager.instance = new NodeMetadataManager(scene || new Scene());
        }
        return NodeMetadataManager.instance;
    }

    private createLabelTexture(metadata: NodeMetadata): Texture {
        // Clear canvas
        this.labelContext.clearRect(0, 0, this.labelCanvas.width, this.labelCanvas.height);

        // Draw background
        this.labelContext.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.labelContext.fillRect(0, 0, this.labelCanvas.width, this.labelCanvas.height);

        // Draw text
        this.labelContext.fillStyle = 'white';
        this.labelContext.fillText(
            metadata.name || 'Unknown',
            this.labelCanvas.width / 2,
            this.labelCanvas.height / 2
        );

        // Create texture
        const texture = new Texture(this.labelCanvas);
        texture.needsUpdate = true;
        return texture;
    }

    public async createMetadataLabel(metadata: NodeMetadata): Promise<Object3D> {
        const texture = this.createLabelTexture(metadata);
        const material = new SpriteMaterial({
            map: texture,
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        const sprite = new Sprite(material);
        sprite.scale.set(this.LABEL_SCALE * 2, this.LABEL_SCALE, 1);
        sprite.renderOrder = 1; // Ensure labels render on top

        // Enable both layers for desktop mode
        sprite.layers.enable(0);
        sprite.layers.enable(1);

        const label: MetadataLabel = {
            sprite,
            metadata,
            lastUpdateDistance: Infinity
        };

        // Add to scene
        this.scene.add(sprite);

        this.labels.set(metadata.id, label);
        return sprite;
    }

    public update(camera: Camera): void {
        this.frameCount++;
        if (this.frameCount % this.UPDATE_INTERVAL !== 0) return;

        const cameraPosition = camera.position;

        this.labels.forEach((label) => {
            const { sprite, metadata } = label;
            
            // Get actual world position from metadata
            this.worldPosition.set(
                metadata.position.x || 0,
                metadata.position.y || 0,
                metadata.position.z || 0
            );
            
            // Update sprite position
            sprite.position.copy(this.worldPosition);
            
            const distance = this.worldPosition.distanceTo(cameraPosition);

            // Update visibility based on distance
            const visible = distance < this.VISIBILITY_THRESHOLD;
            sprite.visible = visible;

            if (label.lastVisible !== visible) {
                label.lastVisible = visible;
            }

            if (visible) {
                // Scale based on distance
                const scale = Math.max(0.5, 1 - (distance / this.VISIBILITY_THRESHOLD));
                sprite.scale.set(
                    this.LABEL_SCALE * scale * 2,
                    this.LABEL_SCALE * scale,
                    1
                );

                // Make sprite face camera
                sprite.lookAt(cameraPosition);
            }

            // Update last known distance
            label.lastUpdateDistance = distance;
        });
    }

    public updateMetadata(id: string, metadata: NodeMetadata): void {
        const label = this.labels.get(id);
        if (!label) {
            this.createMetadataLabel(metadata);
            return;
        }

        // Update metadata
        label.metadata = metadata;

        // Update texture
        const texture = this.createLabelTexture(metadata);
        (label.sprite.material as SpriteMaterial).map?.dispose();
        (label.sprite.material as SpriteMaterial).map = texture;
    }

    public updatePosition(id: string, position: Vector3): void {
        const label = this.labels.get(id);
        if (!label) {
            logger.debug(`No label found for node ${id}`);
            return;
        }

        // Update metadata position
        label.metadata.position = { x: position.x, y: position.y, z: position.z };
        // Update sprite position
        label.sprite.position.copy(position);
    }

    public updateVisibilityThreshold(threshold: number): void {
        if (threshold > 0) {
            this.VISIBILITY_THRESHOLD = threshold;
            logger.debug(`Updated visibility threshold to ${threshold}`);
        }
    }

    public setXRMode(enabled: boolean): void {
        this.labels.forEach((label) => {
            const sprite = label.sprite;
            if (enabled) {
                // XR mode - only layer 1
                sprite.layers.disable(0);
                sprite.layers.enable(1);
            } else {
                // Desktop mode - both layers
                sprite.layers.enable(0);
                sprite.layers.enable(1);
            }
        });
    }

    public removeLabel(id: string): void {
        const label = this.labels.get(id);
        if (!label) return;

        // Clean up resources
        (label.sprite.material as SpriteMaterial).map?.dispose();
        label.sprite.material.dispose();

        // Remove from scene
        this.scene.remove(label.sprite);
        
        // Remove from tracking
        this.labels.delete(id);
    }

    public dispose(): void {
        // Clean up all labels
        this.labels.forEach((label) => {
            (label.sprite.material as SpriteMaterial).map?.dispose();
            label.sprite.material.dispose();
            
            // Remove from scene
            this.scene.remove(label.sprite);
        });
        this.labels.clear();

        // Reset singleton
        NodeMetadataManager.instance = null!;
        logger.info('Disposed NodeMetadataManager');
    }
}