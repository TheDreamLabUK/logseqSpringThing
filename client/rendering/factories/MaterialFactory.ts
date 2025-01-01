import {
    Color,
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
    LineBasicMaterial,
    LineBasicMaterialParameters,
    MeshBasicMaterialParameters,
    MeshPhongMaterialParameters,
    MeshStandardMaterialParameters,
    createMeshBasicMaterial,
    createMeshPhongMaterial,
    createMeshStandardMaterial,
    createLineBasicMaterial,
    DoubleSide
} from '../../core/threeTypes';

export type MaterialSettings = {
    color?: Color | number | string;
    opacity?: number;
    transparent?: boolean;
    wireframe?: boolean;
    emissive?: Color | number | string;
    metalness?: number;
    roughness?: number;
};

export class MaterialFactory {
    private static defaultSettings: MaterialSettings = {
        color: 0x666666,
        opacity: 1.0,
        transparent: false,
        wireframe: false
    };

    private static getBasicMaterialParams(settings: Partial<MaterialSettings>) {
        return {
            ...this.defaultSettings,
            ...settings,
            side: DoubleSide
        } as MeshBasicMaterialParameters;
    }

    static createBasicMaterial(settings: Partial<MaterialSettings> = {}): MeshBasicMaterial {
        return createMeshBasicMaterial(this.getBasicMaterialParams(settings));
    }

    static createPhongMaterial(settings: Partial<MaterialSettings> = {}): MeshPhongMaterial {
        return createMeshPhongMaterial(this.getBasicMaterialParams(settings));
    }

    static createStandardMaterial(settings: Partial<MaterialSettings> = {}): MeshStandardMaterial {
        return createMeshStandardMaterial(this.getBasicMaterialParams(settings));
    }

    static createNodeMaterial(settings: Partial<MaterialSettings> = {}): MeshPhongMaterial {
        return this.createPhongMaterial(settings);
    }

    static createEdgeMaterial(settings: Partial<MaterialSettings> = {}): LineBasicMaterial {
        const finalSettings: LineBasicMaterialParameters = {
            ...this.getBasicMaterialParams(settings),
            transparent: true,
            opacity: 0.6
        };
        return createLineBasicMaterial(finalSettings);
    }

    static createHologramMaterial(settings: Partial<MaterialSettings> = {}): MeshStandardMaterial {
        return this.createStandardMaterial({
            ...settings,
            transparent: true,
            opacity: 0.3,
            metalness: 0.5,
            roughness: 0.2
        });
    }
}
