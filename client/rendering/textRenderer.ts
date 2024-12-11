/**
 * Text rendering for node labels using sprites
 */

import * as THREE from 'three';
import { Vector3, VisualizationSettings } from '../core/types';
import { createLogger } from '../core/utils';
import { settingsManager } from '../state/settings';
import { FONT_URL } from '../core/constants';

// Logger will be used for debugging font loading and text rendering
const _logger = createLogger('TextRenderer');

// Add FontFace API type declarations
declare global {
  interface Document {
    fonts: FontFaceSet;
  }

  interface FontFaceSet extends Set<FontFace> {
    readonly ready: Promise<FontFaceSet>;
    readonly status: 'loading' | 'loaded';
    check(font: string, text?: string): boolean;
    load(font: string, text?: string): Promise<FontFace[]>;
  }
}

export class TextRenderer {
  private static instance: TextRenderer;
  private scene: THREE.Scene;
  private camera: THREE.Camera;
  
  // Font texture and canvas
  private canvas: HTMLCanvasElement;
  private context: CanvasRenderingContext2D;
  private fontLoaded: boolean = false;
  
  // Label management
  private labels: Map<string, THREE.Sprite>;
  private labelPool: THREE.Sprite[];
  private settings: VisualizationSettings;

  private constructor(scene: THREE.Scene, camera: THREE.Camera) {
    this.scene = scene;
    this.camera = camera;
    this.labels = new Map();
    this.labelPool = [];
    this.settings = settingsManager.getSettings();

    // Create canvas for text rendering
    this.canvas = document.createElement('canvas');
    const context = this.canvas.getContext('2d');
    if (!context) {
      throw new Error('Failed to get 2D context for text rendering');
    }
    this.context = context;

    this.initialize();
  }

  static getInstance(scene: THREE.Scene, camera: THREE.Camera): TextRenderer {
    if (!TextRenderer.instance) {
      TextRenderer.instance = new TextRenderer(scene, camera);
    }
    return TextRenderer.instance;
  }

  private async initialize(): Promise<void> {
    await this.loadFont();
    this.setupEventListeners();
  }

  private async loadFont(): Promise<void> {
    try {
      // Load font using FontFace API
      const font = new FontFace(
        'LabelFont',
        `url(${FONT_URL})`
      );

      const loadedFont = await font.load();
      document.fonts.add(loadedFont);
      this.fontLoaded = true;
      _logger.log('Font loaded successfully');
    } catch (error) {
      _logger.error('Failed to load font:', error);
      throw error;
    }
  }

  private setupEventListeners(): void {
    settingsManager.subscribe(settings => {
      this.settings = settings;
      this.updateAllLabels();
    });
  }

  private createTextTexture(text: string): THREE.Texture {
    // Set canvas size
    const fontSize = 48;
    this.context.font = `${fontSize}px LabelFont`;
    const metrics = this.context.measureText(text);
    const width = Math.ceil(metrics.width);
    const height = Math.ceil(fontSize * 1.4); // Add some padding

    this.canvas.width = width;
    this.canvas.height = height;

    // Clear canvas
    this.context.fillStyle = 'transparent';
    this.context.fillRect(0, 0, width, height);

    // Draw text
    this.context.font = `${fontSize}px LabelFont`;
    this.context.textAlign = 'center';
    this.context.textBaseline = 'middle';
    this.context.fillStyle = this.settings.labelColor;
    this.context.fillText(text, width / 2, height / 2);

    // Create texture
    const texture = new THREE.Texture(this.canvas);
    texture.needsUpdate = true;
    return texture;
  }

  private createLabelSprite(text: string): THREE.Sprite {
    // Reuse sprite from pool if available
    let sprite = this.labelPool.pop();
    
    if (!sprite) {
      const spriteMaterial = new THREE.SpriteMaterial({
        transparent: true,
        depthTest: false
      });
      sprite = new THREE.Sprite(spriteMaterial);
    }

    // Update sprite texture
    const texture = this.createTextTexture(text);
    (sprite.material as THREE.SpriteMaterial).map = texture;

    // Set sprite scale based on text dimensions
    const scale = this.settings.labelSize * 0.01;
    sprite.scale.set(
      this.canvas.width * scale,
      this.canvas.height * scale,
      1
    );

    return sprite;
  }

  /**
   * Create or update a label for a node
   */
  updateLabel(id: string, text: string, position: Vector3): void {
    if (!this.fontLoaded || !this.settings.showLabels) {
      return;
    }

    let label = this.labels.get(id);
    if (!label) {
      label = this.createLabelSprite(text);
      this.labels.set(id, label);
      this.scene.add(label);
    } else {
      // Update existing label
      const texture = this.createTextTexture(text);
      (label.material as THREE.SpriteMaterial).map?.dispose();
      (label.material as THREE.SpriteMaterial).map = texture;
    }

    // Update position
    label.position.set(position.x, position.y + 1.5, position.z); // Offset above node
    label.material.opacity = this.calculateOpacity(position);
  }

  /**
   * Remove a label
   */
  removeLabel(id: string): void {
    const label = this.labels.get(id);
    if (label) {
      this.scene.remove(label);
      (label.material as THREE.SpriteMaterial).map?.dispose();
      this.labelPool.push(label); // Return to pool for reuse
      this.labels.delete(id);
    }
  }

  /**
   * Update all labels (e.g., after settings change)
   */
  private updateAllLabels(): void {
    if (!this.fontLoaded || !this.settings.showLabels) {
      return;
    }

    this.labels.forEach((__label, id) => {
      const label = this.labels.get(id);
      if (label) {
        const opacity = this.calculateOpacity(label.position);
        label.material.opacity = opacity;
      }
    });
  }

  /**
   * Update label positions and visibility
   */
  update(): void {
    if (!this.fontLoaded || !this.settings.showLabels) {
      return;
    }

    // Update label opacity based on distance to camera
    this.labels.forEach((__label, id) => {
      const label = this.labels.get(id);
      if (label) {
        const opacity = this.calculateOpacity(label.position);
        label.material.opacity = opacity;
      }
    });
  }

  /**
   * Calculate label opacity based on distance to camera
   */
  private calculateOpacity(position: Vector3): number {
    const distance = this.camera.position.distanceTo(new THREE.Vector3(position.x, position.y, position.z));
    const maxDistance = 100;
    const minDistance = 10;
    
    if (distance > maxDistance) return 0;
    if (distance < minDistance) return 1;
    
    return 1 - ((distance - minDistance) / (maxDistance - minDistance));
  }

  /**
   * Clear all labels
   */
  clear(): void {
    this.labels.forEach((__label, id) => {
      this.removeLabel(id);
    });
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    this.clear();
    this.labels.forEach(label => {
      label.material.dispose();
      (label.material as THREE.SpriteMaterial).map?.dispose();
    });
    this.labelPool.forEach(label => {
      label.material.dispose();
      (label.material as THREE.SpriteMaterial).map?.dispose();
    });
    this.canvas.remove();
  }
}
