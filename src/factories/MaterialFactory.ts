import { 
    Material, 
    MeshBasicMaterial, 
    MeshPhongMaterial, 
    DoubleSide 
} from 'three';
import { NodeMaterialSettings } from '../types/settings';

export class MaterialFactory {
    private materialCache: Map<string, Material>;

    constructor() {
        this.materialCache = new Map();
    }

    private createCacheKey(settings: NodeMaterialSettings): string {
        return `${settings.type}-${settings.color}-${settings.transparent}-${settings.opacity}`;
    }

    private createBasicMaterial(settings: NodeMaterialSettings): Material {
        return new MeshBasicMaterial({
            color: settings.color || 0xffffff,
            transparent: settings.transparent ?? false,
            opacity: settings.opacity ?? 1.0,
            side: settings.side || DoubleSide
        });
    }

    createHologramMaterial(settings?: NodeMaterialSettings): Material {
        // Implementation needed for hologram material
        return new MeshBasicMaterial({ 
            color: settings?.color || 0x00ff00,
            transparent: true,
            opacity: 0.5
        });
    }

    createNodeMaterial(settings: NodeMaterialSettings): Material {
        if (!settings) {
            console.warn('No material settings provided, using default');
            return new MeshBasicMaterial({ color: 0xffffff });
        }

        const cacheKey = this.createCacheKey(settings);
        
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        let material: Material;

        try {
            switch (settings.type) {
                case 'phong':
                    material = this.createPhongMaterial(settings);
                    break;
                case 'hologram':
                    material = this.createHologramMaterial(settings);
                    break;
                default:
                    material = this.createBasicMaterial(settings);
            }

            this.materialCache.set(cacheKey, material);
            return material;
        } catch (error) {
            console.error(`Failed to create material: ${error}`);
            return new MeshBasicMaterial({ color: 0xff0000 }); // Fallback red material
        }
    }

    private createPhongMaterial(settings: NodeMaterialSettings): MeshPhongMaterial {
        return new MeshPhongMaterial({
            color: settings.color || 0xffffff,
            transparent: settings.transparent ?? false,
            opacity: settings.opacity ?? 1.0,
            side: settings.side || DoubleSide
        });
    }
} 