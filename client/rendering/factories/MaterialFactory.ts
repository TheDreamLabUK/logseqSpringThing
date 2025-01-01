import {
    Color,
    Material,
    createMeshBasicMaterial,
    createMeshPhongMaterial,
    createMeshStandardMaterial,
    DoubleSide,
    Side,
    MaterialParameters,
    MeshBasicMaterialParameters,
    MeshPhongMaterialParameters,
    MeshStandardMaterialParameters
} from '../../core/threeTypes';

export interface MaterialSettings {
    color?: Color | number | string;
    opacity?: number;
    transparent?: boolean;
    wireframe?: boolean;
    side?: Side;
    emissive?: Color | number | string;
    metalness?: number;
    roughness?: number;
}

export class MaterialFactory {
    private static defaultSettings: MaterialSettings = {
        color: 0x666666,
        opacity: 1.0,
        transparent: false,
        wireframe: false,
        side: DoubleSide
    };

    static createBasicMaterial(settings: Partial<MaterialSettings> = {}): Material {
        const finalSettings = { ...this.defaultSettings, ...settings } as MeshBasicMaterialParameters;
        return createMeshBasicMaterial(finalSettings);
    }

    static createPhongMaterial(settings: Partial<MaterialSettings> = {}): Material {
        const finalSettings = { ...this.defaultSettings, ...settings } as MeshPhongMaterialParameters;
        return createMeshPhongMaterial(finalSettings);
    }

    static createStandardMaterial(settings: Partial<MaterialSettings> = {}): Material {
        const finalSettings = { ...this.defaultSettings, ...settings } as MeshStandardMaterialParameters;
        return createMeshStandardMaterial(finalSettings);
    }

    static createNodeMaterial(settings: Partial<MaterialSettings> = {}): Material {
        return this.createPhongMaterial(settings);
    }

    static createEdgeMaterial(settings: Partial<MaterialSettings> = {}): Material {
        return this.createBasicMaterial({
            ...settings,
            transparent: true,
            opacity: 0.6
        });
    }

    static createHologramMaterial(settings: Partial<MaterialSettings> = {}): Material {
        return this.createStandardMaterial({
            ...settings,
            transparent: true,
            opacity: 0.3,
            metalness: 0.5,
            roughness: 0.2
        });
    }
}
