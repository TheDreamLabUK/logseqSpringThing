import {
    Group,
    Scene,
    PerspectiveCamera,
    Vector3,
    Color,
    Object3D
} from 'three';
import { NodeMetadata } from '../types/metadata';
import { Settings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
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

    constructor(camera: PerspectiveCamera, scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
        this.metadataGroups = new Map();
        
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
            this.labelGroup.parent.remove(this.labelGroup);
        }
    }
}
