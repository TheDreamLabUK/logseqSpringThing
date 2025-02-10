import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';
import { Color, Material, MeshBasicMaterial, LineBasicMaterial } from 'three';

export class MaterialFactory {
    private static instance: MaterialFactory;
    private materialCache: Map<string, Material>;

    private constructor() {
        this.materialCache = new Map();
    }

    public static getInstance(): MaterialFactory {
        if (!MaterialFactory.instance) {
            MaterialFactory.instance = new MaterialFactory();
        }
        return MaterialFactory.instance;
    }

    private hexToRgb(hex: string): Color {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (!result) {
            return new Color(0xffffff);
        }
        return new Color(`#${result[1]}${result[2]}${result[3]}`);
    }

    public createHologramMaterial(settings: any): HologramShaderMaterial {
        const cacheKey = 'hologram';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as HologramShaderMaterial;
        }

        const material = new HologramShaderMaterial(settings);
        
        if (settings.visualization?.hologram?.ringColor) {
            const materialColor = this.hexToRgb(settings.visualization.hologram.ringColor);
            material.uniforms.color.value = materialColor;
        }

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getHologramMaterial(settings: any): HologramShaderMaterial {
        return this.createHologramMaterial(settings);
    }

    public getSceneSphereMaterial(settings: any): Material {
        const cacheKey = 'scene-sphere';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }
        const material = new MeshBasicMaterial({
            wireframe: true,
            color: settings.visualization?.hologram?.ringColor || 0xffffff,
            transparent: true,
            opacity: settings.visualization?.hologram?.opacity || 0.8
        });
        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getRingMaterial(settings: any): Material {
        const cacheKey = 'ring';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }
        const material = this.getHologramMaterial(settings);
        material.transparent = true;
        material.opacity = settings.visualization?.hologram?.opacity || 0.6;
        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getNodeMaterial(settings: any, context: 'ar' | 'desktop' = 'desktop'): Material {
        const cacheKey = `node-${context}`;
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization?.nodes?.baseColor || 0x4287f5,
            transparent: true,
            opacity: settings.visualization?.nodes?.opacity || 0.9,
            wireframe: true,
            depthTest: true
        });
        
        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getMetadataMaterial(): Material {
        const cacheKey = 'metadata';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public updateMaterial(type: string, settings: any): void {
        const material = this.materialCache.get(type);
        if (!material) return;

        switch (type) {
            case 'node-basic':
            case 'node-phong': {
                const nodeMaterial = material as LineBasicMaterial;
                nodeMaterial.color = this.hexToRgb(settings.visualization?.nodes?.baseColor || '#4287f5');
                nodeMaterial.opacity = settings.visualization?.nodes?.opacity || 0.9;
                nodeMaterial.needsUpdate = true;
                break;
            }
            case 'edge':
                (material as LineBasicMaterial).color = this.hexToRgb(settings.visualization?.edges?.color || '#6e7c91');
                break;
            case 'hologram':
                if (material instanceof HologramShaderMaterial) {
                    material.uniforms.color.value = this.hexToRgb(settings.visualization?.hologram?.ringColor || '#ffffff');
                }
                break;
        }
    }

    public dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
