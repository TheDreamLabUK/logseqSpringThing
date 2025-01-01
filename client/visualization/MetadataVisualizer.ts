import * as THREE from 'three';
import { MaterialFactory } from '../rendering/factories/MaterialFactory';
import { Settings } from '../types/settings';
import { createLogger } from '../core/utils';

const logger = createLogger('MetadataVisualizer');

export interface MetadataOptions {
    position?: THREE.Vector3;
    rotation?: THREE.Euler;
    scale?: THREE.Vector3;
    text?: string;
    fontSize?: number;
    color?: number;
    backgroundColor?: number;
    opacity?: number;
}

export interface MetadataProperties {
    title?: string;
    description?: string;
    tags?: string[];
    properties?: Record<string, string | number | boolean>;
}

export class MetadataVisualizer {
    private scene: THREE.Scene;
    private materialFactory: MaterialFactory;
    private settings: Settings;
    private metadataObjects: Map<string, THREE.Group>;
    private canvas: HTMLCanvasElement;
    private context: CanvasRenderingContext2D;

    constructor(scene: THREE.Scene, materialFactory: MaterialFactory, settings: Settings) {
        this.scene = scene;
        this.materialFactory = materialFactory;
        this.settings = settings;
        this.metadataObjects = new Map();

        this.canvas = document.createElement('canvas');
        const ctx = this.canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context');
        }
        this.context = ctx;
    }

    createMetadata(id: string, properties: MetadataProperties, options: MetadataOptions = {}): THREE.Group {
        const group = new THREE.Group();

        // Create text texture
        const text = this.formatMetadata(properties);
        const texture = this.createTextTexture(text, options);

        // Create plane with text
        const geometry = new THREE.PlaneGeometry(1, 1);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            opacity: options.opacity ?? 1.0,
            side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);

        if (options.position) {
            group.position.copy(options.position);
        }
        if (options.rotation) {
            group.rotation.copy(options.rotation);
        }
        if (options.scale) {
            group.scale.copy(options.scale);
        }

        group.add(mesh);
        this.metadataObjects.set(id, group);
        this.scene.add(group);

        return group;
    }

    updateMetadata(id: string, properties: Partial<MetadataProperties>, options: Partial<MetadataOptions> = {}): void {
        const group = this.metadataObjects.get(id);
        if (!group) {
            logger.warn(`Metadata object ${id} not found`);
            return;
        }

        const mesh = group.children[0] as THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
        if (!mesh) return;

        if (properties) {
            const text = this.formatMetadata(properties as MetadataProperties);
            const texture = this.createTextTexture(text, options);
            mesh.material.map = texture;
        }

        if (options.position) {
            group.position.copy(options.position);
        }
        if (options.rotation) {
            group.rotation.copy(options.rotation);
        }
        if (options.scale) {
            group.scale.copy(options.scale);
        }
        if (options.opacity !== undefined) {
            mesh.material.opacity = options.opacity;
        }
    }

    private formatMetadata(properties: MetadataProperties): string {
        const lines: string[] = [];

        if (properties.title) {
            lines.push(`Title: ${properties.title}`);
        }
        if (properties.description) {
            lines.push(`Description: ${properties.description}`);
        }
        if (properties.tags && properties.tags.length > 0) {
            lines.push(`Tags: ${properties.tags.join(', ')}`);
        }
        if (properties.properties) {
            Object.entries(properties.properties).forEach(([key, value]) => {
                lines.push(`${key}: ${value}`);
            });
        }

        return lines.join('\n');
    }

    private createTextTexture(text: string, options: Partial<MetadataOptions> = {}): THREE.CanvasTexture {
        const fontSize = options.fontSize || 24;
        const textColor = options.color ? `#${options.color.toString(16)}` : '#ffffff';
        const bgColor = options.backgroundColor ? `#${options.backgroundColor.toString(16)}` : '#000000';

        this.context.font = `${fontSize}px Arial`;
        const lines = text.split('\n');
        const lineHeight = fontSize * 1.2;
        const padding = fontSize * 0.5;

        // Measure text dimensions
        let maxWidth = 0;
        for (const line of lines) {
            const metrics = this.context.measureText(line);
            maxWidth = Math.max(maxWidth, metrics.width);
        }

        // Set canvas size
        this.canvas.width = maxWidth + padding * 2;
        this.canvas.height = lines.length * lineHeight + padding * 2;

        // Draw background
        this.context.fillStyle = bgColor;
        this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw text
        this.context.font = `${fontSize}px Arial`;
        this.context.fillStyle = textColor;
        this.context.textBaseline = 'top';

        lines.forEach((line, i) => {
            this.context.fillText(line, padding, padding + i * lineHeight);
        });

        // Create texture
        const texture = new THREE.CanvasTexture(this.canvas);
        texture.needsUpdate = true;

        return texture;
    }

    removeMetadata(id: string): void {
        const group = this.metadataObjects.get(id);
        if (group) {
            const mesh = group.children[0] as THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
            if (mesh) {
                mesh.material.map?.dispose();
                mesh.material.dispose();
                mesh.geometry.dispose();
            }
            this.scene.remove(group);
            this.metadataObjects.delete(id);
        }
    }

    getMetadata(id: string): THREE.Group | undefined {
        return this.metadataObjects.get(id);
    }

    clear(): void {
        this.metadataObjects.forEach((group, id) => {
            this.removeMetadata(id);
        });
    }

    dispose(): void {
        this.clear();
        this.canvas.remove();
    }
}
