import { BufferGeometry, SphereGeometry, CylinderGeometry } from 'three';

export class GeometryFactory {
    private static instance: GeometryFactory;
    private geometryCache = new Map<string, BufferGeometry>();

    private constructor() {}

    static getInstance(): GeometryFactory {
        if (!GeometryFactory.instance) {
            GeometryFactory.instance = new GeometryFactory();
        }
        return GeometryFactory.instance;
    }

    getNodeGeometry(quality: 'low' | 'medium' | 'high'): BufferGeometry {
        const cacheKey = `node-${quality}`;
        if (this.geometryCache.has(cacheKey)) {
            return this.geometryCache.get(cacheKey)!;
        }

        const segments = {
            low: 8,
            medium: 16,
            high: 32
        }[quality] || 16;

        const geometry = new SphereGeometry(1, segments, segments);
        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }

    getHologramGeometry(type: string, quality: string): BufferGeometry {
        const cacheKey = `hologram-${type}-${quality}`;
        if (this.geometryCache.has(cacheKey)) {
            return this.geometryCache.get(cacheKey)!;
        }

        const segments = {
            low: { ring: 32, sphere: 8 },
            medium: { ring: 64, sphere: 16 },
            high: { ring: 128, sphere: 32 }
        }[quality] || { ring: 64, sphere: 16 };

        let geometry: BufferGeometry;
        switch (type) {
            case 'ring':
                geometry = new SphereGeometry(1, segments.ring, segments.ring);
                break;
            case 'buckminster':
                geometry = new SphereGeometry(1, 20, 20);
                break;
            case 'geodesic':
                geometry = new SphereGeometry(1, 16, 16);
                break;
            case 'triangleSphere':
                geometry = new SphereGeometry(1, segments.sphere, segments.sphere);
                break;
            default:
                geometry = new SphereGeometry(1, segments.sphere, segments.sphere);
        }

        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }

    getEdgeGeometry(): BufferGeometry {
        const cacheKey = 'edge';
        if (this.geometryCache.has(cacheKey)) {
            return this.geometryCache.get(cacheKey)!;
        }

        // CylinderGeometry parameters:
        // radiusTop, radiusBottom, height, radialSegments
        const geometry = new CylinderGeometry(0.05, 0.05, 1, 8);
        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }

    dispose(): void {
        this.geometryCache.forEach(geometry => geometry.dispose());
        this.geometryCache.clear();
    }
}
