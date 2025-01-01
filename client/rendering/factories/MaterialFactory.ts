import * as THREE from 'three';
import { MaterialSettings } from '../../core/types';
import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';

export class MaterialFactory {
    private static instance: MaterialFactory;
    private materialCache = new Map<string, THREE.Material>();

    private constructor() {}

    static getInstance(): MaterialFactory {
        if (!MaterialFactory.instance) {
            MaterialFactory.instance = new MaterialFactory();
        }
        return MaterialFactory.instance;
    }

    createNodeMaterial(settings: MaterialSettings): THREE.Material {
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

    createHologramMaterial(): HologramShaderMaterial {
        return new HologramShaderMaterial();
    }

    getMetadataMaterial(color: THREE.Color): THREE.Material {
        const cacheKey = `metadata_${color.getHexString()}`;
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new THREE.MeshBasicMaterial();
        material.color = color;
        material.transparent = true;
        material.opacity = 0.8;
        material.side = THREE.DoubleSide;

        this.materialCache.set(cacheKey, material);
        return material;
    }

    private createCacheKey(settings: MaterialSettings): string {
        return `${settings.type}_${settings.color?.getHexString()}_${settings.opacity}_${settings.transparent}_${settings.side}`;
    }

    dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
