import {
    Group,
    Scene,
    PerspectiveCamera,
    Vector3,
    Color,
    Object3D,
    SphereGeometry,
    MeshBasicMaterial,
    Mesh
} from 'three';
import { NodeMetadata } from '../types/metadata';
import { Settings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
import { createLogger, Logger } from '../core/logger';
import { debugState } from '../core/debugState';
import { UnifiedTextRenderer } from './UnifiedTextRenderer';

interface MetadataLabelGroup extends Group {
    name: string;
    userData: {
        isMetadata: boolean;
        nodeId?: string;
    };
}

export type MetadataLabelCallback = (group: MetadataLabelGroup) => void;

export class MetadataVisualizer {
    private scene: Scene;
    private labelGroup: Group;
    private settings: Settings;
    private textRenderer: UnifiedTextRenderer;
    private metadataGroups: Map<string, MetadataLabelGroup>;
    private logger: Logger;
    private debugHelpers: Map<string, Object3D>;

    constructor(camera: PerspectiveCamera, scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
        this.metadataGroups = new Map();
        this.logger = createLogger('MetadataVisualizer');
        
        this.debugHelpers = new Map();
        
        this.logger.info('Initializing MetadataVisualizer with settings:', {
            enableLabels: settings.visualization.labels.enableLabels,
            textColor: settings.visualization.labels.textColor,
            desktopFontSize: settings.visualization.labels.desktopFontSize
        });

        this.labelGroup = new Group();
        this.scene.add(this.labelGroup);
        
        // Initialize text renderer
        this.textRenderer = new UnifiedTextRenderer(camera, scene, settings.visualization.labels);
        
        // Enable both layers by default for desktop mode
        this.labelGroup.layers.enable(0);
        this.labelGroup.layers.enable(1);
        
        // Set initial layer mode
        this.setXRMode(platformManager.isXRMode);
        
        // Listen for XR mode changes
        platformManager.on('xrmodechange', (enabled: boolean) => {
            this.setXRMode(enabled);
        });
    }

    public async createMetadataLabel(metadata: NodeMetadata, nodeId: string): Promise<MetadataLabelGroup> {
        const group = new Group() as MetadataLabelGroup;
        group.name = 'metadata-label';
        group.userData = { 
            isMetadata: true,
            nodeId
        };

        // Format file size
        const fileSizeFormatted = metadata.fileSize > 1024 * 1024 
            ? `${(metadata.fileSize / (1024 * 1024)).toFixed(1)}MB`
            : metadata.fileSize > 1024
                ? `${(metadata.fileSize / 1024).toFixed(1)}KB`
                : `${metadata.fileSize}B`;

        this.logger.info('Creating metadata label:', {
            nodeId,
            metadata: {
                name: metadata.name,
                fileSize: fileSizeFormatted,
                nodeSize: metadata.nodeSize,
                hyperlinkCount: metadata.hyperlinkCount
            }
        });

        // Create text labels using UnifiedTextRenderer
        const labelTexts = [
            `${metadata.name} (${fileSizeFormatted})`,
            `Size: ${metadata.nodeSize.toFixed(1)}`,
            `${metadata.hyperlinkCount} links`
        ];

        const labelPositions = [1.5, 1.0, 0.5]; // Y positions for each label

        labelTexts.forEach((text, index) => {
            const position = new Vector3(0, labelPositions[index], 0);
            const labelId = `${nodeId}-label-${index}`;
            this.textRenderer.updateLabel(
                labelId,
                text,
                position,
                new Color(this.settings.visualization.labels.textColor)
            );
        });

        this.metadataGroups.set(nodeId, group);
        return group;
    }

    private setGroupLayer(group: Object3D, enabled: boolean): void {
        if (enabled) {
            group.traverse(child => {
                child.layers.disable(0);
                child.layers.enable(1);
            });
            group.layers.disable(0);
            group.layers.enable(1);
        } else {
            group.traverse(child => {
                child.layers.enable(0);
                child.layers.enable(1);
            });
            group.layers.enable(0);
            group.layers.enable(1);
        }
    }

    public setXRMode(enabled: boolean): void {
        this.setGroupLayer(this.labelGroup, enabled);
        // Text renderer handles its own XR mode
    }

    public updateMetadataPosition(nodeId: string, position: Vector3): void {
        const group = this.metadataGroups.get(nodeId);
        if (group) {
            group.position.copy(position);
            
            // Update text positions
            const labelPositions = [1.5, 1.0, 0.5];
            labelPositions.forEach((yOffset, index) => {
                const labelId = `${nodeId}-label-${index}`;
                const labelPosition = position.clone().add(new Vector3(0, yOffset, 0));
                this.textRenderer.updateLabel(labelId, '', labelPosition); // Text content remains unchanged
                this.logger.debug('Updating label position:', {
                    nodeId,
                    labelId,
                    position: [labelPosition.x, labelPosition.y, labelPosition.z],
                    yOffset
                });
                
                // Only show debug helpers when debug is enabled
                if (debugState.isEnabled()) {
                    const debugId = `${labelId}-debug`;
                    let debugSphere = this.debugHelpers.get(debugId) as Mesh | undefined;
                    if (!debugSphere) {
                        const geometry = new SphereGeometry(0.1);
                        const material = new MeshBasicMaterial({ color: 0xff0000 });
                        debugSphere = new Mesh(geometry, material) as Mesh;
                        this.labelGroup.add(debugSphere);
                        this.debugHelpers.set(debugId, debugSphere);
                    }
                    debugSphere.position.copy(labelPosition);
                    debugSphere.visible = true;
                }
            });
        }
    }

    public removeMetadata(nodeId: string): void {
        const group = this.metadataGroups.get(nodeId);
        if (group) {
            this.labelGroup.remove(group);
            this.metadataGroups.delete(nodeId);
            
            // Remove text labels
            [0, 1, 2].forEach(index => {
                const labelId = `${nodeId}-label-${index}`;
                this.textRenderer.removeLabel(labelId);
                const debugId = `${labelId}-debug`;
                
                if (debugState.isEnabled()) {
                    // Remove debug helpers
                    const debugHelper = this.debugHelpers.get(debugId);
                    if (debugHelper) this.labelGroup.remove(debugHelper);
                }
                if (this.debugHelpers.has(debugId)) this.debugHelpers.delete(debugId);
            });
        }
    }

    public dispose(): void {
        this.metadataGroups.forEach(group => {
            if (group.userData.nodeId) {
                this.removeMetadata(group.userData.nodeId);
            }
        });
        this.metadataGroups.clear();
        this.textRenderer.dispose();
        if (this.labelGroup.parent) {
            // Clean up debug helpers
            this.debugHelpers.forEach(helper => {
                this.labelGroup.remove(helper);
            });
            this.debugHelpers.clear();
            
            this.labelGroup.parent.remove(this.labelGroup);
        }
    }
}
