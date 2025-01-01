import {
    BufferGeometry,
    BoxGeometry,
    SphereGeometry,
    TorusGeometry,
    IcosahedronGeometry,
    Vector3
} from 'three';

export class GeometryFactory {
    private static instance: GeometryFactory;
    private geometries: Map<string, BufferGeometry> = new Map();

    private constructor() {}

    public static getInstance(): GeometryFactory {
        if (!GeometryFactory.instance) {
            GeometryFactory.instance = new GeometryFactory();
        }
        return GeometryFactory.instance;
    }

    public getNodeGeometry(quality: 'low' | 'medium' | 'high'): BufferGeometry {
        const key = `node-${quality}`;
        if (this.geometries.has(key)) {
            return this.geometries.get(key)!;
        }

        let geometry: BufferGeometry;
        switch (quality) {
            case 'low':
                geometry = new BoxGeometry(1, 1, 1);
                break;
            case 'medium':
                geometry = new SphereGeometry(0.5, 16, 16);
                break;
            case 'high':
                geometry = new IcosahedronGeometry(0.5, 2);
                break;
            default:
                geometry = new BoxGeometry(1, 1, 1);
        }

        this.geometries.set(key, geometry);
        return geometry;
    }

    public getHologramGeometry(type: 'ring' | 'sphere' | 'icosahedron', quality: 'low' | 'medium' | 'high'): BufferGeometry {
        const key = `hologram-${type}-${quality}`;
        if (this.geometries.has(key)) {
            return this.geometries.get(key)!;
        }

        let geometry: BufferGeometry;
        switch (type) {
            case 'ring':
                geometry = new TorusGeometry(1, 0.02, 16, 100);
                break;
            case 'sphere':
                geometry = new SphereGeometry(1, quality === 'low' ? 16 : quality === 'medium' ? 32 : 64);
                break;
            case 'icosahedron':
                geometry = new IcosahedronGeometry(1, quality === 'low' ? 1 : quality === 'medium' ? 2 : 3);
                break;
            default:
                geometry = new TorusGeometry(1, 0.02, 16, 100);
        }

        this.geometries.set(key, geometry);
        return geometry;
    }

    public dispose(): void {
        this.geometries.forEach(geometry => geometry.dispose());
        this.geometries.clear();
    }
}
