import {
    BufferGeometry,
    IcosahedronGeometry,
    OctahedronGeometry
} from 'three';
import { GeometryFactory } from '../../factories/GeometryFactory';
import { createLogger } from '../../../core/logger';

const logger = createLogger('NodeGeometryManager');

// LOD level definitions
export enum LODLevel {
    HIGH = 0,    // < 10 meters: Full detail
    MEDIUM = 1,  // 10-50 meters: Medium detail
    LOW = 2      // > 50 meters: Low detail
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
        [LODLevel.HIGH]: 10.0,    // Show full detail when closer than 10 meters
        [LODLevel.MEDIUM]: 50.0,  // Medium detail between 10-50 meters
        [LODLevel.LOW]: 150.0     // Low detail beyond 50 meters
    };

    private readonly qualitySettings: Record<LODLevel, GeometryQuality> = {
        [LODLevel.HIGH]: { segments: 1, radius: 0.12 },   // 12cm radius with 1 subdivision
        [LODLevel.MEDIUM]: { segments: 0, radius: 0.12 }, // 12cm radius basic octahedron
        [LODLevel.LOW]: { segments: 0, radius: 0.1 }      // 10cm octahedron for distance
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
                // High detail: Icosahedron with 1 subdivision
                geometry = new IcosahedronGeometry(quality.radius, 1);
                break;

            case LODLevel.MEDIUM:
                // Medium detail: Basic octahedron
                geometry = new OctahedronGeometry(quality.radius);
                break;

            case LODLevel.LOW:
                // Low detail: Smaller octahedron
                geometry = new OctahedronGeometry(quality.radius);
                break;

            default:
                logger.warn(`Unknown LOD level: ${level}, falling back to medium quality`);
                geometry = new OctahedronGeometry(quality.radius);
        }

        // Compute and adjust bounding sphere for better frustum culling
        geometry.computeBoundingSphere();
        if (geometry.boundingSphere) {
            geometry.boundingSphere.radius *= 1.2;
        }

        return geometry;
    }

    public getGeometryForDistance(distance: number): BufferGeometry {
        // Determine appropriate LOD level based on distance
        let targetLOD = LODLevel.HIGH;
        
        // Use more conservative thresholds for AR mode
        const isAR = window.location.href.includes('ar=true') || 
                    document.querySelector('#xr-button')?.textContent?.includes('Exit AR');
        
        // Apply different thresholds for AR mode
        // Use adjusted thresholds for AR mode
        const mediumThreshold = isAR ? this.lodThresholds[LODLevel.MEDIUM] * 1.5 : this.lodThresholds[LODLevel.MEDIUM];
        
        // Fix the LOD logic to ensure nodes don't vanish as we get closer
        // The closer we are, the higher the detail should be
        if (distance <= this.lodThresholds[LODLevel.HIGH]) {
            // Close distance: use high detail
            targetLOD = LODLevel.HIGH;
        } else if (distance <= mediumThreshold) {
            // Medium distance: use medium detail
            targetLOD = LODLevel.MEDIUM;
        } else {
            // Far distance: use low detail
            targetLOD = LODLevel.LOW;
        }

        // Only update if LOD level changed
        if (targetLOD !== this.currentLOD) {
            this.currentLOD = targetLOD;
            logger.info(`Switching to LOD level ${targetLOD} for distance ${distance.toFixed(2)}`);
        }

        // Always ensure we return a valid geometry
        const geometry = this.geometryCache.get(targetLOD);
        if (!geometry) {
            logger.warn(`No geometry found for LOD level ${targetLOD}, falling back to MEDIUM`);
            return this.geometryCache.get(LODLevel.MEDIUM)!;
        }
        return geometry;
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