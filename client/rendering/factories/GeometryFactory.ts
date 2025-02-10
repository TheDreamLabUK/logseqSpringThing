import { 
    BufferGeometry, 
    SphereGeometry, 
    CylinderGeometry, 
    IcosahedronGeometry,
    TorusGeometry
} from 'three';

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

    getNodeGeometry(quality: 'low' | 'medium' | 'high', context: 'ar' | 'desktop' = 'desktop', size: number = 1000): BufferGeometry {
        const cacheKey = `node-${quality}-${context}-${size}`;
        if (this.geometryCache.has(cacheKey)) {
            return this.geometryCache.get(cacheKey)!;
        }

        let geometry: BufferGeometry;
        let segmentCount: number;
        
        switch (quality) {
            case 'low':
               segmentCount = context === 'ar' ? 4 : 8;
                 break;
            case 'medium':
                segmentCount = context === 'ar' ? 8 : 16;
                break;
            case 'high':
                segmentCount = context === 'ar' ? 16 : 24;
                break;
            default:
                segmentCount = 16;
        }
        geometry = new SphereGeometry(size, segmentCount, segmentCount);
        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }

    getHologramGeometry(type: string, quality: string, size: number = 1000): BufferGeometry {
        const cacheKey = `hologram-${type}-${quality}-${size}`;
        if (this.geometryCache.has(cacheKey)) {
            return this.geometryCache.get(cacheKey)!;
        }

        const segments = {
            low: { ring: 32, sphere: 16 },
            medium: { ring: 48, sphere: 24 },
            high: { ring: 64, sphere: 32 }
        }[quality] || { ring: 96, sphere: 48 };

        let geometry: BufferGeometry;
        switch (type) {
            case 'ring':
                geometry = new TorusGeometry(size, size * 0.05, segments.ring, segments.ring * 2);
                break;
            case 'buckminster':
                geometry = new IcosahedronGeometry(size, 2); // Icosahedron for Buckminster Fullerene
                break;
            case 'geodesic':
                geometry = new IcosahedronGeometry(size, 5); // Higher detail Icosahedron
                break;
            case 'triangleSphere':
                geometry = new SphereGeometry(size, segments.sphere, segments.sphere / 2); // Sphere with triangular faces
                break;
            default:
                geometry = new SphereGeometry(size, segments.sphere, segments.sphere / 2);
        }

        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }

    getEdgeGeometry(context: 'ar' | 'desktop' = 'desktop', quality?: 'low' | 'medium' | 'high'): BufferGeometry {
        const cacheKey = `edge-${context}-${quality || 'medium'}`;
        if (this.geometryCache.has(cacheKey)) {
            return this.geometryCache.get(cacheKey)!;
        }

        // Use CylinderGeometry for more reliable edge rendering
        const baseRadius = context === 'ar' ? 0.1 : 0.15; // Reduced base radius for better scaling
        
        // Adjust segments based on quality
        const segments = {
            low: context === 'ar' ? 4 : 6,
            medium: context === 'ar' ? 6 : 8,
            high: context === 'ar' ? 8 : 10
        }[quality || 'medium'];

        const geometry = new CylinderGeometry(baseRadius, baseRadius, 1, segments);
        
        // Rotate 90 degrees to align with Z-axis
        geometry.rotateX(Math.PI / 2);
        
        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }

    dispose(): void {
        this.geometryCache.forEach(geometry => geometry.dispose());
        this.geometryCache.clear();
    }
}
