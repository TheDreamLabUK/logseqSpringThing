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
            low: { ring: 64, sphere: 32 },
            medium: { ring: 96, sphere: 48 },
            high: { ring: 128, sphere: 64 }
        }[quality] || { ring: 96, sphere: 48 };

        let geometry: BufferGeometry;
        switch (type) {
            case 'ring':
                // Translucent rings at scene scale
                geometry = new SphereGeometry(40, segments.ring, segments.ring);
                break;
            case 'outerSphere':
                geometry = new SphereGeometry(200, segments.sphere, segments.sphere);
                break;
            case 'middleSphere':
                geometry = new SphereGeometry(100, segments.sphere, segments.sphere);
                break;
            case 'innerSphere':
                geometry = new SphereGeometry(40, segments.sphere, segments.sphere);
                break;
            default:
                geometry = new SphereGeometry(40, segments.sphere, segments.sphere);
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
