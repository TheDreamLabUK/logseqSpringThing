import * as THREE from 'three';
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
import { createLogger } from '../core/logger';

export class HologramManager {
    private logger = createLogger('HologramManager');
    private settings: Settings;
    private isXRMode: boolean;
    private materialFactory: MaterialFactory;
    private group: Group;
    private holograms: Mesh[] = [];
    private materials: HologramMaterial[] = [];

    constructor(settings: Settings, scene: Scene) {
        this.logger.debug('HologramManager constructor called with settings:', settings);
        try {
            if (!settings?.visualization?.hologram) {
                throw new Error('Missing hologram settings');
            }

            if (!settings.visualization.hologram.enabled) {
                this.logger.warn('Holograms are disabled in settings. Some features may not work.');
            }

            this.settings = settings;
            this.isXRMode = false;
            this.materialFactory = new MaterialFactory();
            this.group = new Group();
            
            if (settings.visualization.hologram.enabled) {
                this.createHolograms();
            }
            
            scene.add(this.group);
            
            this.logger.debug('HologramManager initialized successfully', {
                enabled: settings.visualization.hologram.enabled,
                groupChildren: this.group.children.length
            });
        } catch (error) {
            this.logger.error('Error in HologramManager constructor:', error);
            throw error;
        }

        this.logger.debug('Three.js version:', THREE.REVISION);
    }

    createHolograms(): void {
        this.logger.debug('Creating holograms - start');
        const hologramSettings = this.settings?.visualization?.hologram;
        
        this.logger.debug('Hologram settings:', hologramSettings);
        
        if (!hologramSettings) {
            this.logger.warn("Hologram settings are not available");
            return;
        }

        if (!hologramSettings.enabled) {
            this.logger.warn("Holograms are disabled in settings");
            return;
        }

        // Create a simple test cube instead of a ring
        try {
            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0x00ff00,
                wireframe: true 
            });
            const cube = new THREE.Mesh(geometry, material);
            
            this.holograms.push(cube);
            this.group.add(cube);
            
            this.logger.debug('Created test cube', {
                position: cube.position,
                geometry: cube.geometry.type,
                material: cube.material.type
            });
        } catch (error) {
            this.logger.error('Error creating test cube:', error);
        }

        this.logger.debug('Holograms created', {
            hologramCount: this.holograms.length,
            groupChildren: this.group.children.length
        });
    }

    private createHologramMesh(type: string, quality: string): Mesh | null {
        try {
            this.logger.debug('Creating hologram mesh', { type, quality });
            const geometry = this.createGeometry(type, quality);
            const material = new MeshBasicMaterial({
                color: 0x00ff00,
                transparent: true,
                opacity: 0.5,
                side: DoubleSide
            });

            return new Mesh(geometry, material);
        } catch (error) {
            this.logger.error('Failed to create hologram mesh:', error);
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
        if (!this.holograms.length) {
            return;
        }

        try {
            // Just rotate the first hologram
            const hologram = this.holograms[0];
            hologram.rotation.y += 0.01;
            
            this.logger.debug('Updated hologram rotation', {
                rotation: hologram.rotation.y
            });
        } catch (error) {
            this.logger.error('Error in simple rotation update:', error);
        }
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