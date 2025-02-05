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

    public getNodeMaterial(settings: any, context: 'ar' | 'desktop' = 'desktop'): Material {
        const cacheKey = `node-${context}`;
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        let material: Material;
        
        if (context === 'ar') {
            // AR (Meta Quest) - Performance optimized
            // AR (Meta Quest) - Performance optimized with minimal features
            // AR (Meta Quest) - Performance optimized with minimal features
            material = new MeshBasicMaterial({
                color: settings.visualization?.nodes?.baseColor || 0x4287f5,
                transparent: true,
                opacity: settings.visualization?.nodes?.opacity || 0.9,
                depthWrite: true,
                side: 0, // FrontSide for better performance
                depthTest: true
            });
        } else {
            // Desktop/VR - Full features with better visual quality
            material = new MeshPhongMaterial({
                color: settings.visualization?.nodes?.baseColor || 0x4287f5,
                transparent: true,
                opacity: settings.visualization?.nodes?.opacity || 0.9,
                depthWrite: true,
                side: 2, // DoubleSide for better visuals
                shininess: 30,
                specular: 0x444444,
                depthTest: true
            });
        }

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getPhongNodeMaterial(settings: any): Material {
        const cacheKey = 'node-phong';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization?.nodes?.baseColor || 0x4287f5,
            transparent: true,
            opacity: settings.visualization?.nodes?.opacity || 0.9,
            depthWrite: true
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getSceneSphereMaterial(type: string, settings: any): Material {
        const cacheKey = `scene-sphere-${type}`;
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization?.hologram?.color || 0x4287f5,
            transparent: true,
            opacity: 0.3,
            depthWrite: false
        });

        (material as any).wireframe = true;

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getRingMaterial(settings: any): Material {
        const cacheKey = 'ring';
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization?.hologram?.color || 0x4287f5,
            transparent: true,
            opacity: 0.15,
            depthWrite: false
        });

        this.materialCache.set(cacheKey, material);
        return material;
    }

    public getEdgeMaterial(settings: any, context: 'ar' | 'desktop' = 'desktop'): Material {
        const cacheKey = `edge-${context}`;
        if (this.materialCache.has(cacheKey)) {
            return this.materialCache.get(cacheKey)!;
        }

        const material = new MeshBasicMaterial({
            color: settings.visualization?.edges?.color || 0x6e7c91,
            transparent: true,
            opacity: context === 'ar' ? 0.8 : 0.9,
            depthWrite: true,
            depthTest: true,
            side: 2  // DoubleSide for better visibility
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
                const nodeMaterial = material as MeshBasicMaterial;
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
                    material.uniforms.color.value = this.hexToRgb(settings.visualization?.hologram?.color || '#ffffff');
                }
                break;
            case 'scene-sphere-inner':
            case 'scene-sphere-middle':
            case 'scene-sphere-outer':
                const sphereMaterial = material as MeshBasicMaterial;
                sphereMaterial.color = this.hexToRgb(settings.visualization?.hologram?.color || '#4287f5');
                sphereMaterial.needsUpdate = true;
                break;
            case 'ring':
                const ringMaterial = material as MeshBasicMaterial;
                ringMaterial.color = this.hexToRgb(settings.visualization?.hologram?.color || '#4287f5');
                ringMaterial.needsUpdate = true;
                break;
        }
    }

    public dispose(): void {
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();
    }
}
