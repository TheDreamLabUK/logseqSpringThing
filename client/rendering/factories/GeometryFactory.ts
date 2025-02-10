import { 
    BufferGeometry, 
    SphereGeometry, 
    CylinderGeometry, 
    BufferAttribute
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
        
        if (context === 'ar') {
            // AR (Meta Quest) - Optimized geometry
            switch (quality) {
                case 'high':
                    geometry = new SphereGeometry(1, 8, 6);
                    break;
                case 'medium':
                    geometry = new SphereGeometry(1, 6, 4);
                    break;
                case 'low':
                    geometry = new SphereGeometry(1, 4, 3);
                    break;
            }
            segmentCount = quality === 'high' ? 8 : quality === 'medium' ? 6 : 4;
        } else {
            // Desktop/VR - Full quality geometry
            segmentCount = {
                low: 16,
                medium: 24,
                high: 24
            }[quality] || 16;
            
            geometry = new SphereGeometry(1, segmentCount, segmentCount);
        }

        this.geometryCache.set(cacheKey, geometry);
        return geometry;
    }
    getHologramGeometry(type: string, quality: string, edgeOnly: boolean = false): BufferGeometry {
        const cacheKey = `hologram-${type}-${quality}`;
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

        if (edgeOnly) {
            // Create a simplified edge-only version using thin cylinders
            const edgeGeometry = new BufferGeometry();
            const vertices = [];
            const radius = type === 'ring' ? 40 : 200;
            const segmentCount = segments.ring;

            // Create a circle of vertices
            for (let i = 0; i < segmentCount; i++) {
                const angle1 = (i / segmentCount) * Math.PI * 2;
                const angle2 = ((i + 1) / segmentCount) * Math.PI * 2;

                const x1 = Math.cos(angle1) * radius;
                const y1 = Math.sin(angle1) * radius;
                const x2 = Math.cos(angle2) * radius;
                const y2 = Math.sin(angle2) * radius;

                // Add edge vertices
                vertices.push(x1, y1, 0, x2, y2, 0);
            }

            geometry.dispose();
            edgeGeometry.setAttribute('position', 
                new BufferAttribute(new Float32Array(vertices), 3)
            );
            
            // Cache with edge-only suffix
            this.geometryCache.set(cacheKey + '-edge', edgeGeometry);
            return edgeGeometry;
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
