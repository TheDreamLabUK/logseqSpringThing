import * as THREE from 'three';
import { MaterialSettings } from '../../core/types';
import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';

export class MaterialFactory {
    private static instance: MaterialFactory;
    private materialCache: Map<string, THREE.Material>;

    private constructor() {
        this.materialCache = new Map();
    }

    public static getInstance(): MaterialFactory {
        if (!MaterialFactory.instance) {
            MaterialFactory.instance = new MaterialFactory();
        }
        return MaterialFactory.instance;
    }

    public createNodeMaterial(settings: MaterialSettings): THREE.Material {
        const cacheKey = this.createCacheKey(settings);
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        let material: THREE.Material;

        if (settings.type === 'phong') {
            material = new THREE.MeshPhongMaterial();
            if (settings.color) material.color = settings.color;
            if (settings.transparent !== undefined) material.transparent = settings.transparent;
            if (settings.opacity !== undefined) material.opacity = settings.opacity;
            if (settings.side !== undefined) material.side = settings.side;
        } else if (settings.type === 'hologram') {
            material = new HologramShaderMaterial({
                color: settings.color,
                opacity: settings.opacity,
                glowIntensity: settings.glowIntensity
            });
        } else {
            material = new THREE.MeshBasicMaterial();
            if (settings.color) material.color = settings.color;
            if (settings.transparent !== undefined) material.transparent = settings.transparent;
            if (settings.opacity !== undefined) material.opacity = settings.opacity;
            if (settings.side !== undefined) material.side = settings.side;
        }

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public createNodeMaterial(type: string): THREE.Material {
        const key = `node-${type}`;
        if (this.materialCache.has(key)) {
            return this.materialCache.get(key)!;
        }

        let material: THREE.Material;
        switch (type) {
            case 'basic':
                material = new THREE.MeshBasicMaterial();
                break;
            case 'phong':
                material = new THREE.MeshPhongMaterial();
                break;
            default:
                material = new THREE.MeshBasicMaterial();
        }

        this.materialCache.set(key, material);
        return material;
    }

    public createHologramMaterial(): HologramShaderMaterial {
        return new HologramShaderMaterial();
    }

    public getMetadataMaterial(color: THREE.Color): THREE.Material {
        const key = `metadata-${color.getHexString()}`;
        if (this.materialCache.has(key)) {
            return this.materialCache.get(key)!;
        }

        const material = new THREE.MeshBasicMaterial({ color });
        this.materialCache.set(key, material);
        return material;
    }

    public updateMaterial(type: string, settings: any): void {
        const material = this.materialCache.get(`node-${type}`);
        if (material) {
            if ('color' in material) {
                (material as THREE.MeshBasicMaterial | THREE.MeshPhongMaterial).color.set(settings.color);
            }
            if ('transparent' in material) {
                material.transparent = settings.transparent;
                material.opacity = settings.opacity;
            }
        }
    }

    private createCacheKey(settings: MaterialSettings): string {
        return `${settings.type}_${settings.color?.getHexString()}_${settings.opacity}_${settings.transparent}_${settings.side}`;
    }

    public dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
