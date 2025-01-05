import { 
    Mesh, 
    Material, 
    Group, 
    RingGeometry, 
    BufferGeometry,
    TorusGeometry,
    Scene,
    MeshBasicMaterial,
    Color,
    DoubleSide
} from 'three';
import { Settings } from '../types/settings';
import { MaterialFactory } from '../factories/MaterialFactory';
import { HologramMaterial } from '../materials/HologramMaterial';

export class HologramManager {
    private settings: Settings;
    private isXRMode: boolean;
    private materialFactory: MaterialFactory;
    private group: Group;
    private holograms: Mesh[] = [];
    private materials: HologramMaterial[] = [];

    constructor(settings: Settings, scene: Scene) {
        console.log('HologramManager constructor called.');
        this.settings = settings;
        this.isXRMode = false;
        this.materialFactory = new MaterialFactory();
        this.group = new Group();
        this.createHolograms();
        scene.add(this.group);
    }

    createHolograms(): void {
        console.log('HologramManager.createHolograms called.');
        const hologramSettings = this.settings?.visualization?.hologram;
        
        if (!hologramSettings) {
            console.warn("Hologram settings are not available");
            return;
        }

        const quality = this.isXRMode && this.settings?.xr 
            ? this.settings.xr.quality 
            : 'medium';

        for (let i = 0; i < (hologramSettings.ringCount || 0); i++) {
            const ring = this.createHologramMesh("ring", quality);
            if (!ring) continue;

            const scale = hologramSettings.ringSizes?.[i] || 20;
            ring.scale.set(scale, scale, scale);
            ring.rotateX(Math.PI / 2 * i);
            ring.rotateY(Math.PI / 4 * i);
            ring.userData.rotationSpeed = hologramSettings.ringRotationSpeed * (i + 1);
            
            this.holograms.push(ring);
            this.group.add(ring);
        }
    }

    private createHologramMesh(type: string, quality: string): Mesh | null {
        try {
            const geometry = this.createGeometry(type, quality);
            const material = new MeshBasicMaterial({
                color: 0x00ff00,
                transparent: true,
                opacity: 0.5,
                side: DoubleSide
            });

            return new Mesh(geometry, material);
        } catch (error) {
            console.error(`Failed to create hologram mesh: ${error}`);
            return null;
        }
    }

    private createGeometry(type: string, quality: string): BufferGeometry {
        const segments = this.getSegmentCount(quality);
        
        switch (type.toLowerCase()) {
            case 'ring':
                return new RingGeometry(
                    0.8, // inner radius
                    1.0, // outer radius
                    segments // segments
                );
            case 'torus':
                return new TorusGeometry(
                    1.0, // radius
                    0.1, // tube
                    segments, // radialSegments
                    segments // tubularSegments
                );
            default:
                throw new Error(`Unknown geometry type: ${type}`);
        }
    }

    private getSegmentCount(quality: string): number {
        switch (quality.toLowerCase()) {
            case 'low': return 32;
            case 'medium': return 64;
            case 'high': return 128;
            default: return 64;
        }
    }

    public update(delta: number): void {
        // No time-based updates for now
        
        // Update rotations
        this.holograms.forEach((hologram, index) => {
            if (hologram.userData.rotationSpeed) {
                hologram.rotation.y += hologram.userData.rotationSpeed * delta;
            }
        });
    }

    public getGroup(): Group {
        return this.group;
    }

    public dispose(): void {
        this.holograms.forEach(hologram => {
            hologram.geometry.dispose();
            if (hologram.material instanceof Material) {
                hologram.material.dispose();
            }
        });
        this.materials.forEach(material => material.dispose());
        this.group.clear();
    }
} 