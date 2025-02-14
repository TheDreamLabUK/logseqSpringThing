import {
    BufferGeometry,
    IcosahedronGeometry,
    SphereGeometry
} from 'three';
import { GeometryFactory } from '../../factories/GeometryFactory';
import { createLogger } from '../../../core/logger';

const logger = createLogger('NodeGeometryManager');

// LOD level definitions
export enum LODLevel {
    HIGH = 0,    // < 10 units: Full detail
    MEDIUM = 1,  // 10-30 units: Medium detail
    LOW = 2      // > 30 units: Low detail
}

interface LODThresholds {
    [LODLevel.HIGH]: number;   // Distance threshold for high detail
    [LODLevel.MEDIUM]: number; // Distance threshold for medium detail
    [LODLevel.LOW]: number;    // Distance threshold for low detail
}

interface GeometryQuality {
    segments: number;  // Number of segments/detail level
    radius: number;    // Base size
}

export class NodeGeometryManager {
    private static instance: NodeGeometryManager;
    private geometryCache: Map<LODLevel, BufferGeometry>;
    private currentLOD: LODLevel = LODLevel.HIGH;
    
    private readonly lodThresholds: LODThresholds = {
        [LODLevel.HIGH]: 10,
        [LODLevel.MEDIUM]: 30,
        [LODLevel.LOW]: 50
    };

    private readonly qualitySettings: Record<LODLevel, GeometryQuality> = {
        [LODLevel.HIGH]: { segments: 16, radius: 1 },
        [LODLevel.MEDIUM]: { segments: 8, radius: 1 },
        [LODLevel.LOW]: { segments: 4, radius: 1 }
    };

    private constructor() {
        GeometryFactory.getInstance(); // Initialize factory
        this.geometryCache = new Map();
        this.initializeGeometries();
    }

    public static getInstance(): NodeGeometryManager {
        if (!NodeGeometryManager.instance) {
            NodeGeometryManager.instance = new NodeGeometryManager();
        }
        return NodeGeometryManager.instance;
    }

    private initializeGeometries(): void {
        // Initialize geometries for each LOD level
        Object.values(LODLevel).forEach((level) => {
            if (typeof level === 'number') {
                const quality = this.qualitySettings[level];
                const geometry = this.createOptimizedGeometry(level, quality);
                this.geometryCache.set(level, geometry);
            }
        });
        logger.info('Initialized geometries for all LOD levels');
    }

    private createOptimizedGeometry(level: LODLevel, quality: GeometryQuality): BufferGeometry {
        // Create geometry based on LOD level
        let geometry: BufferGeometry;

        switch (level) {
            case LODLevel.HIGH:
                // High detail: Icosahedron for smooth spherical shape
                geometry = new IcosahedronGeometry(quality.radius, quality.segments);
                break;

            case LODLevel.MEDIUM:
                // Medium detail: Sphere with reduced segments
                geometry = new SphereGeometry(
                    quality.radius,
                    quality.segments,
                    quality.segments
                );
                break;

            case LODLevel.LOW:
                // Low detail: Simple box for distant nodes
                // Use SphereGeometry with minimal segments for low detail
                geometry = new SphereGeometry(
                    quality.radius,
                    4, // minimal segments
                    4  // minimal segments
                );
                break;

            default:
                logger.warn(`Unknown LOD level: ${level}, falling back to medium quality`);
                geometry = new SphereGeometry(quality.radius, 8, 8);
        }


        return geometry;
    }

    public getGeometryForDistance(distance: number): BufferGeometry {
        // Determine appropriate LOD level based on distance
        let targetLOD = LODLevel.HIGH;

        if (distance > this.lodThresholds[LODLevel.LOW]) {
            targetLOD = LODLevel.LOW;
        } else if (distance > this.lodThresholds[LODLevel.MEDIUM]) {
            targetLOD = LODLevel.MEDIUM;
        }

        // Only update if LOD level changed
        if (targetLOD !== this.currentLOD) {
            this.currentLOD = targetLOD;
            logger.debug(`Switching to LOD level ${targetLOD} for distance ${distance}`);
        }

        return this.geometryCache.get(targetLOD) || this.geometryCache.get(LODLevel.MEDIUM)!;
    }

    public getCurrentLOD(): LODLevel {
        return this.currentLOD;
    }

    public getThresholdForLOD(level: LODLevel): number {
        return this.lodThresholds[level];
    }

    public dispose(): void {
        // Clean up geometries
        this.geometryCache.forEach(geometry => {
            geometry.dispose();
        });
        this.geometryCache.clear();
        logger.info('Disposed all geometries');
    }
}