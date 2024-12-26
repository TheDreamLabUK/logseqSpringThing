import { Color, DoubleSide, Material, MeshBasicMaterial, MeshPhongMaterial } from 'three';
import { Settings } from '../../types/settings';
import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';

type CachedMaterial = Material | HologramShaderMaterial;

export class MaterialFactory {
    private static instance: MaterialFactory;
    private materialCache = new Map<string, CachedMaterial>();

    private constructor() {}

    static getInstance(): MaterialFactory {
        if (!MaterialFactory.instance) {
            MaterialFactory.instance = new MaterialFactory();
        }
        return MaterialFactory.instance;
    }

    getNodeMaterial(settings: Settings): Material {
        const cacheKey = 'node-basic';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization.nodes.baseColor,
            transparent: true,
            opacity: settings.visualization.nodes.opacity
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    getPhongNodeMaterial(): Material {
        const cacheKey = 'node-phong';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshPhongMaterial({
            color: 0x4fc3f7,
            shininess: 30,
            specular: 0x004ba0,
            transparent: true,
            opacity: 0.9,
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    getMetadataMaterial(): Material {
        const cacheKey = 'metadata';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: new Color('#00ff00'),
            transparent: true,
            opacity: 0.8,
            side: DoubleSide
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    getHologramMaterial(settings: Settings): HologramShaderMaterial {
        const cacheKey = 'hologram';
        const cached = this.materialCache.get(cacheKey);
        if (cached && cached instanceof HologramShaderMaterial) {
            return cached;
        }

        const material = new HologramShaderMaterial(settings);
        this.materialCache.set(cacheKey, material);
        return material;
    }

    updateMaterial(type: string, settings: Settings): void {
        const material = this.materialCache.get(type);
        if (!material) return;

        switch (type) {
            case 'node-basic':
            case 'node-phong':
                if (material instanceof MeshBasicMaterial || material instanceof MeshPhongMaterial) {
                    material.color.set(settings.visualization.nodes.baseColor);
                    material.opacity = settings.visualization.nodes.opacity;
                }
                break;
            case 'hologram':
                if (material instanceof HologramShaderMaterial) {
                    const hexColor = settings.visualization.hologram.ringColor;
                    const hex = hexColor.replace('#', '');
                    const r = parseInt(hex.substring(0, 2), 16) / 255;
                    const g = parseInt(hex.substring(2, 4), 16) / 255;
                    const b = parseInt(hex.substring(4, 6), 16) / 255;
                    material.uniforms.color.value.set(r, g, b);
                    material.uniforms.opacity.value = settings.visualization.hologram.ringOpacity;
                }
                break;
        }
    }

    dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }

    getEdgeMaterial(settings: Settings): Material {
        const cacheKey = 'edge';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: new Color(settings.visualization.edges.color),
            opacity: settings.visualization.edges.opacity,
            transparent: true
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }
}
