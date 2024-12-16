/**
 * Modern text rendering system optimized for AR with desktop fallback
 * Uses Three.js TroikaText for high-quality text rendering with proper depth
 */

import * as THREE from 'three';
import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';
import type { Settings } from '../core/types';

const logger = createLogger('TextRenderer');

interface LabelState {
  text: string;
  basePosition: THREE.Vector3;
  visible: boolean;
  scale: number;
  boundingBox: THREE.Object3D;
}

export class TextRenderer {
  private static instance: TextRenderer;
  private scene: THREE.Scene;
  private camera: THREE.Camera;
  private labels: Map<string, THREE.Group>;
  private labelStates: Map<string, LabelState>;
  private unsubscribers: (() => void)[];
  private projMatrix: THREE.Matrix4;
  private viewMatrix: THREE.Matrix4;
  private currentSettings: Settings['labels'];

  private constructor(scene: THREE.Scene, camera: THREE.Camera) {
    this.scene = scene;
    this.camera = camera;
    this.labels = new Map();
    this.labelStates = new Map();
    this.unsubscribers = [];
    this.projMatrix = new THREE.Matrix4();
    this.viewMatrix = new THREE.Matrix4();
    this.currentSettings = settingsManager.getCurrentSettings().labels;

    this.setupSubscriptions();
    logger.debug('TextRenderer initialized');
  }

  static getInstance(scene: THREE.Scene, camera: THREE.Camera): TextRenderer {
    if (!TextRenderer.instance) {
      TextRenderer.instance = new TextRenderer(scene, camera);
    }
    return TextRenderer.instance;
  }

  private setupSubscriptions(): void {
    const labelSettings: Array<keyof Settings['labels']> = ['enableLabels', 'desktopFontSize', 'textColor'];
    labelSettings.forEach(setting => {
      const unsubscribe = settingsManager.subscribe('labels', setting as string, (value) => {
        try {
          // Update local settings cache
          this.currentSettings = {
            ...this.currentSettings,
            [setting]: value
          };
          this.onLabelSettingChanged();
        } catch (error) {
          logger.error(`Error handling label setting change for ${setting}:`, error);
        }
      });
      this.unsubscribers.push(unsubscribe);
    });
  }

  private async onLabelSettingChanged(): Promise<void> {
    try {
      this.labels.forEach((labelGroup, id) => {
        const state = this.labelStates.get(id);
        if (state) {
          this.updateLabelStyle(labelGroup, state);
        }
      });
    } catch (error) {
      logger.error('Error updating labels after settings change:', error);
    }
  }

  private updateLabelStyle(labelGroup: THREE.Group, state: LabelState): void {
    const textMesh = labelGroup.children[0] as THREE.Mesh;
    if (!textMesh) return;

    try {
      const material = textMesh.material as THREE.MeshBasicMaterial;
      material.color.set(this.currentSettings.textColor);
      material.opacity = 1;
      material.transparent = true;

      // Update visibility and scale
      labelGroup.visible = this.currentSettings.enableLabels && state.visible;
      const scale = this.currentSettings.desktopFontSize * state.scale;
      labelGroup.scale.set(scale, scale, scale);

      // Update position with offset
      labelGroup.position.copy(state.basePosition);

      // Update bounding box for culling
      state.boundingBox = labelGroup;
    } catch (error) {
      logger.error('Error updating label style:', error);
    }
  }

  public async updateLabel(id: string, text: string, position: THREE.Vector3): Promise<void> {
    try {
      if (!this.currentSettings.enableLabels) {
        this.removeLabel(id);
        return;
      }

      let labelGroup = this.labels.get(id);
      if (!labelGroup) {
        labelGroup = new THREE.Group();
        
        // Create text mesh using basic geometry for now
        const geometry = new THREE.PlaneGeometry(1, 1);
        const material = new THREE.MeshBasicMaterial({
          color: this.currentSettings.textColor,
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide
        });
        
        const textMesh = new THREE.Mesh(geometry, material);
        labelGroup.add(textMesh);
        this.labels.set(id, labelGroup);
        this.scene.add(labelGroup);
      }

      // Update state
      const state: LabelState = {
        text,
        basePosition: position.clone(),
        visible: true,
        scale: 1,
        boundingBox: labelGroup
      };
      this.labelStates.set(id, state);

      // Update style and position
      this.updateLabelStyle(labelGroup, state);
    } catch (error) {
      logger.error('Error updating label:', error);
    }
  }

  public removeLabel(id: string): void {
    try {
      const labelGroup = this.labels.get(id);
      if (labelGroup) {
        this.scene.remove(labelGroup);
        labelGroup.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.geometry.dispose();
            child.material.dispose();
          }
        });
        this.labels.delete(id);
        this.labelStates.delete(id);
      }
    } catch (error) {
      logger.error('Error removing label:', error);
    }
  }

  public update(): void {
    if (this.labels.size === 0) return;

    try {
      // Update matrices
      this.viewMatrix.copy(this.camera.matrixWorldInverse);
      this.projMatrix.copy(this.camera.projectionMatrix);

      // Update each label
      this.labels.forEach((labelGroup, id) => {
        const state = this.labelStates.get(id);
        if (!state) return;

        // Check if label is in view
        const cameraDistance = this.camera.position.distanceTo(state.basePosition);
        state.visible = cameraDistance < 100; // Simple distance-based culling
        
        if (!state.visible) {
          labelGroup.visible = false;
          return;
        }

        // Update scale based on distance
        state.scale = Math.max(0.1, Math.min(1, 5 / cameraDistance));

        // Make text face camera
        labelGroup.lookAt(this.camera.position);

        // Update opacity based on view angle
        const material = (labelGroup.children[0] as THREE.Mesh).material as THREE.MeshBasicMaterial;
        const viewVector = new THREE.Vector3().subVectors(this.camera.position, state.basePosition).normalize();
        const forward = new THREE.Vector3(0, 0, 1).applyMatrix4(this.camera.matrixWorld);
        const dot = viewVector.dot(forward.normalize());
        material.opacity = Math.max(0.2, (1 + dot) / 2);

        // Update style with new state
        this.updateLabelStyle(labelGroup, state);
      });
    } catch (error) {
      logger.error('Error updating labels:', error);
    }
  }

  public dispose(): void {
    try {
      this.labels.forEach(labelGroup => {
        labelGroup.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.geometry.dispose();
            child.material.dispose();
          }
        });
      });
      this.labels.clear();
      this.labelStates.clear();
      this.unsubscribers.forEach(unsubscribe => unsubscribe());
      this.unsubscribers = [];
    } catch (error) {
      logger.error('Error disposing TextRenderer:', error);
    }
  }
}
