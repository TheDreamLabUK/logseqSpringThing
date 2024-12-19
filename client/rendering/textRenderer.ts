import type { LabelSettings } from '../core/types';
import { settingsManager } from '../state/settings';
import { createLogger } from '../utils/logger';
import * as THREE from 'three';

const logger = createLogger('TextRenderer');

interface LabelState {
    text: string;
    position: THREE.Vector3;
    visible: boolean;
    boundingBox?: THREE.Object3D;
}

export class TextRenderer {
    private labels: Map<string, THREE.Group>;
    private camera: THREE.Camera;
    private labelStates: Map<string, LabelState>;
    private unsubscribers: Array<() => void> = [];
    private projMatrix: THREE.Matrix4;
    private viewMatrix: THREE.Matrix4;
    private currentSettings: LabelSettings;

    constructor(camera: THREE.Camera) {
        this.camera = camera;
        this.labels = new Map();
        this.labelStates = new Map();
        this.projMatrix = new THREE.Matrix4();
        this.viewMatrix = new THREE.Matrix4();
        this.currentSettings = settingsManager.getCurrentSettings().labels;
        this.setupSettingsSubscriptions();
    }

    private setupSettingsSubscriptions(): void {
        Object.keys(this.currentSettings).forEach(setting => {
            const unsubscribe = settingsManager.subscribe('labels', setting as keyof LabelSettings, (value) => {
                this.handleSettingChange(setting as keyof LabelSettings, value);
            });
            this.unsubscribers.push(unsubscribe);
        });
    }

    private handleSettingChange(setting: keyof LabelSettings, value: any): void {
        try {
            switch (setting) {
                case 'desktopFontSize':
                    this.updateFontSize(value as number);
                    break;
                case 'textColor':
                    this.updateTextColor(value as string);
                    break;
                case 'enableLabels':
                    this.updateLabelVisibility(value as boolean);
                    break;
                default:
                    // Other settings handled elsewhere
                    break;
            }
        } catch (error) {
            logger.error(`Error handling setting change for ${setting}:`, error);
        }
    }

    private updateFontSize(fontSize: number): void {
        if (!this.labels) return;
        
        this.labels.forEach((group) => {
            group.children.forEach((child) => {
                if (child instanceof THREE.Mesh && child.userData.text) {
                    const material = child.material as THREE.MeshBasicMaterial;
                    material.dispose();
                    
                    // Create new text geometry with updated font size
                    const geometry = this.createTextGeometry(child.userData.text, {
                        fontSize,
                        position: child.position.clone()
                    });
                    
                    // Replace old geometry
                    child.geometry.dispose();
                    child.geometry = geometry;
                }
            });
        });
    }

    private createTextGeometry(text: string, { fontSize, position }: { fontSize: number; position: THREE.Vector3 }): THREE.BufferGeometry {
        // Create a simple plane geometry as a placeholder
        // In a real implementation, this would create actual text geometry based on the font and text
        const width = fontSize * text.length * 0.5;
        const height = fontSize;
        
        // Create vertices for a simple plane
        const vertices = new Float32Array([
            -width/2 + position.x, -height/2 + position.y, position.z,  // bottom left
            width/2 + position.x, -height/2 + position.y, position.z,   // bottom right
            width/2 + position.x, height/2 + position.y, position.z,    // top right
            -width/2 + position.x, height/2 + position.y, position.z    // top left
        ]);

        // Create UVs
        const uvs = new Float32Array([
            0, 0,  // bottom left
            1, 0,  // bottom right
            1, 1,  // top right
            0, 1   // top left
        ]);

        // Create indices
        const indices = new Uint16Array([
            0, 1, 2,  // first triangle
            0, 2, 3   // second triangle
        ]);

        // Create normals (facing forward in this case)
        const normals = new Float32Array([
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ]);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        return geometry;
    }

    private updateTextColor(newColor: string): void {
        if (!this.labels) return;
        
        this.labels.forEach((group) => {
            group.children.forEach((child) => {
                if (child instanceof THREE.Mesh) {
                    const material = child.material as THREE.MeshBasicMaterial;
                    material.color.set(newColor);
                    material.dispose();
                    material.color.set(newColor);
                }
            });
        });
    }

    private updateLabelVisibility(visible: boolean): void {
        // Update visibility for all labels
        this.labels.forEach((labelGroup) => {
            labelGroup.visible = visible;
        });
    }

    public updateLabel(id: string, text: string, position: THREE.Vector3): void {
        try {
            let labelGroup = this.labels.get(id);
            if (!labelGroup) {
                labelGroup = new THREE.Group();
                this.labels.set(id, labelGroup);
            }

            const state: LabelState = {
                text,
                position: position.clone(),
                visible: true
            };
            this.labelStates.set(id, state);

            // Update style and position
            this.updateLabelStyle(labelGroup, state);
        } catch (error) {
            logger.error('Error updating label:', error);
        }
    }

    private updateLabelStyle(labelGroup: THREE.Group, state: LabelState): void {
        try {
            // Update position
            labelGroup.position.copy(state.position);

            // Update text content and style
            // Implementation depends on how you're rendering text
            // (e.g., using HTML elements, sprites, or geometry)

            // Update visibility
            labelGroup.visible = this.currentSettings.enableLabels && state.visible;

            // Update bounding box for culling
            state.boundingBox = labelGroup;
        } catch (error) {
            logger.error('Error updating label style:', error);
        }
    }

    public removeLabel(id: string): void {
        try {
            const labelGroup = this.labels.get(id);
            if (labelGroup) {
                // Clean up THREE.js objects
                this.clearLabels();
                this.labels.delete(id);
                this.labelStates.delete(id);
            }
        } catch (error) {
            logger.error('Error removing label:', error);
        }
    }

    private clearLabels(): void {
        if (!this.labels) return;
        
        this.labels.forEach((group) => {
            while (group.children.length > 0) {
                const child = group.children[0];
                group.remove(child);
                if (child instanceof THREE.Mesh) {
                    child.geometry.dispose();
                    if (child.material instanceof THREE.Material) {
                        child.material.dispose();
                    }
                }
            }
        });
    }

    public update(): void {
        try {
            // Update projection and view matrices
            this.camera.updateMatrixWorld();
            this.projMatrix.copy(this.camera.projectionMatrix);
            this.viewMatrix.copy(this.camera.matrixWorldInverse);

            // Update label positions and visibility
            this.labelStates.forEach((state, id) => {
                const labelGroup = this.labels.get(id);
                if (!labelGroup) return;

                // Update label position and style
                this.updateLabelStyle(labelGroup, state);
            });
        } catch (error) {
            logger.error('Error updating labels:', error);
        }
    }

    public dispose(): void {
        try {
            // Clean up THREE.js objects
            this.clearLabels();
            this.labels.clear();
            this.labelStates.clear();

            // Clean up subscribers
            this.unsubscribers.forEach(unsubscribe => unsubscribe());
            this.unsubscribers = [];
        } catch (error) {
            logger.error('Error disposing TextRenderer:', error);
        }
    }
}
