import { HologramShaderMaterial } from '../materials/HologramShaderMaterial';
import { Color, Material, MeshBasicMaterial, MeshPhongMaterial, _LineBasicMaterial as LineBasicMaterial, _PointsMaterial as PointsMaterial, _Side as Side } from 'three';
import { Settings } from '../../types/settings';

export class MaterialFactory {
    private static instance: MaterialFactory;
    private materialCache: Map<string, Material>;
    private _settings: Settings;

    private constructor(_settings: Settings) {
        this._settings = _settings;
        this.materialCache = new Map();
    }

    public static getInstance(_settings: Settings): MaterialFactory {
        if (!MaterialFactory.instance) {
            MaterialFactory.instance = new MaterialFactory(_settings);
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

    public createHologramMaterial(_settings: Settings): HologramShaderMaterial {
        const cacheKey = 'hologram';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as HologramShaderMaterial;
        }

        const material = new HologramShaderMaterial(_settings);
        
        if (_settings.visualization?.hologram?.color) {
            const materialColor = this.hexToRgb(_settings.visualization.hologram.color);
            material.uniforms.color.value = materialColor;
        }

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getHologramMaterial(_settings: Settings): HologramShaderMaterial {
        return this.createHologramMaterial(_settings);
    }

    public getNodeMaterial(_settings: Settings): MeshBasicMaterial {
        const cacheKey = 'node-basic';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as MeshBasicMaterial;
        }

        const material = new MeshBasicMaterial({
            color: _settings.visualization?.node?.color || 0xffffff
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getPhongNodeMaterial(): MeshPhongMaterial {
        const cacheKey = 'node-phong';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as MeshPhongMaterial;
        }

        const material = new MeshPhongMaterial({
            color: 0xffffff
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getEdgeMaterial(_settings: Settings): LineBasicMaterial {
        const cacheKey = 'edge';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as LineBasicMaterial;
        }

        const material = new LineBasicMaterial({
            color: _settings.visualization?.edge?.color || 0xffffff
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getMetadataMaterial(): MeshBasicMaterial {
        const cacheKey = 'metadata';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey) as MeshBasicMaterial;
        }

        const material = new MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public updateMaterial(type: string, _settings: Settings): void {
        const material = this.materialCache.get(type);
        if (!material) return;

        switch (type) {
            case 'node-basic':
            case 'node-phong':
                (material as MeshBasicMaterial | MeshPhongMaterial).color = this.hexToRgb(_settings.visualization?.node?.color || '#ffffff');
                break;
            case 'edge':
                (material as LineBasicMaterial).color = this.hexToRgb(_settings.visualization?.edge?.color || '#ffffff');
                break;
            case 'hologram':
                if (material instanceof HologramShaderMaterial) {
                    material.uniforms.color.value = this.hexToRgb(_settings.visualization?.hologram?.color || '#ffffff');
                }
                break;
        }
    }

    public dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
