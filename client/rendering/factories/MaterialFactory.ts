import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';
import { Color, Material, MeshBasicMaterial, MeshPhongMaterial, LineBasicMaterial } from 'three';

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
        
        if (settings.visualization?.hologram?.color) {
            const materialColor = this.hexToRgb(settings.visualization.hologram.color);
            material.uniforms.color.value = materialColor;
        }

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getHologramMaterial(settings: any): HologramShaderMaterial {
        return this.createHologramMaterial(settings);
    }

    public getNodeMaterial(settings: any): Material {
        const cacheKey = 'node-basic';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization?.node?.color || 0xffffff
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getPhongNodeMaterial(): Material {
        const cacheKey = 'node-phong';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshPhongMaterial({
            color: 0xffffff
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getEdgeMaterial(settings: any): Material {
        const cacheKey = 'edge';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new LineBasicMaterial({
            color: settings.visualization?.edge?.color || 0xffffff
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
            case 'node-phong':
                (material as MeshBasicMaterial | MeshPhongMaterial).color = this.hexToRgb(settings.visualization?.node?.color || '#ffffff');
                break;
            case 'edge':
                (material as LineBasicMaterial).color = this.hexToRgb(settings.visualization?.edge?.color || '#ffffff');
                break;
            case 'hologram':
                if (material instanceof HologramShaderMaterial) {
                    material.uniforms.color.value = this.hexToRgb(settings.visualization?.hologram?.color || '#ffffff');
                }
                break;
        }
    }

    public dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
