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

    getNodeGeometry(quality: 'low' | 'medium' | 'high', context: 'ar' | 'desktop' = 'desktop'): BufferGeometry {
        const cacheKey = `node-${quality}-${context}`;
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
        geometry = new SphereGeometry(1, segmentCount, segmentCount);
        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }
    getHologramGeometry(type: string, quality: string, edgeOnly: boolean = false): BufferGeometry {
        const cacheKey = `hologram-${type}-${quality}${edgeOnly ? '-edge' : ''}`;
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
                geometry = new TorusGeometry(1, 0.05, segments.ring, segments.ring * 2);
                break;
            case 'buckminster':
                geometry = new IcosahedronGeometry(1, 2); // Icosahedron for Buckminster Fullerene
                break;
            case 'geodesic':
                geometry = new IcosahedronGeometry(1, 5); // Higher detail Icosahedron
                break;
            case 'triangleSphere':
                geometry = new SphereGeometry(1, 32, 16); // Sphere with triangular faces
                break;
            default:
                geometry = new SphereGeometry(1, segments.sphere, segments.sphere);
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
