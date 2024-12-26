import { Color, DoubleSide, Material, MeshBasicMaterial, MeshPhongMaterial } from 'three';
import { Settings } from '../../types/settings';
import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';

export class MaterialFactory {
    private static instance: MaterialFactory;
    private materialCache = new Map<string, Material>();

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
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as HologramShaderMaterial;
        }

        const material = new HologramShaderMaterial({
            uniforms: {
                color: { value: new Color(settings.visualization.hologram.ringColor) },
                opacity: { value: settings.visualization.hologram.ringOpacity },
                time: { value: 0 },
                pulseSpeed: { value: 1.0 },
                pulseIntensity: { value: 0.2 }
            }
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    updateMaterial(type: string, settings: Settings): void {
        const material = this.materialCache.get(type);
        if (!material) return;

        switch (type) {
            case 'node-basic':
            case 'node-phong':
                (material as MeshBasicMaterial | MeshPhongMaterial).color.set(settings.visualization.nodes.baseColor);
                material.opacity = settings.visualization.nodes.opacity;
                break;
            case 'hologram':
                const hologramMaterial = material as HologramShaderMaterial;
                hologramMaterial.uniforms.color.value = new Color(settings.visualization.hologram.ringColor);
                hologramMaterial.uniforms.opacity.value = settings.visualization.hologram.ringOpacity;
                break;
        }
    }

    dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
