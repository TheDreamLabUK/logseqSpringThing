import * as THREE from 'three';
import { HologramShaderMaterial } from '../rendering/materials/HologramShaderMaterial';

export class HologramManager {
    private readonly hologramGroup: THREE.Group;
    private readonly settings: any;
    private isXRMode: boolean = false;

    private readonly segments = {
        low: { ring: 32, sphere: 16 },
        medium: { ring: 64, sphere: 32 },
        high: { ring: 128, sphere: 64 }
    };

    constructor(
        private readonly scene: THREE.Scene,
        settings: any
    ) {
        this.settings = settings;
        this.hologramGroup = new THREE.Group();
        this.scene.add(this.hologramGroup);
        this.initializeGeometries();
        this.createHolographicStructures();
    }

    private initializeGeometries(): void {
        const quality = this.isXRMode ? 'high' : (this.settings.quality || 'medium');
        const segmentCount = this.segments[quality as keyof typeof this.segments] || this.segments.medium;
        
        // Create base geometries
        const ringGeometry = new THREE.TorusGeometry(1, 0.02, segmentCount.ring, segmentCount.ring * 2);
        const sphereGeometry = new THREE.SphereGeometry(1, segmentCount.sphere, segmentCount.sphere);
        
        // Create hologram material
        const material = new HologramShaderMaterial(this.settings);
        
        // Add ring structures
        for (let i = 0; i < 3; i++) {
            const ring = new THREE.Mesh(ringGeometry, material.clone());
            ring.scale.setScalar(1.5 + i * 0.5);
            ring.rotation.x = Math.PI / 3 * i;
            this.hologramGroup.add(ring);
        }
        
        // Add sphere structure
        const sphere = new THREE.Mesh(sphereGeometry, material.clone());
        this.hologramGroup.add(sphere);
    }

    private createHolographicStructures(): void {
        // Clear existing structures
        while (this.hologramGroup.children.length > 0) {
            const child = this.hologramGroup.children[0];
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
            this.hologramGroup.remove(child);
        }
        
        this.initializeGeometries();
    }

    public setXRMode(enabled: boolean): void {
        this.isXRMode = enabled;
        this.createHolographicStructures();
    }

    public update(deltaTime: number): void {
        this.hologramGroup.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material instanceof HologramShaderMaterial) {
                child.rotation.y += deltaTime * 0.1;
                child.material.update(deltaTime);
            }
        });
    }

    public handleHandInteraction(hand: THREE.XRHand): void {
        if (!this.isXRMode) return;

        const indexTip = hand.joints['index-finger-tip'];
        if (!indexTip) return;

        const position = new THREE.Vector3();
        position.set(
            indexTip.position.x,
            indexTip.position.y,
            indexTip.position.z
        );
        position.applyMatrix4(indexTip.matrixWorld);

        this.hologramGroup.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material instanceof HologramShaderMaterial) {
                const distance = position.distanceTo(child.position);
                if (distance < 0.1) {
                    child.material.handleInteraction(position);
                }
            }
        });
    }

    public updateSettings(settings: any): void {
        Object.assign(this.settings, settings);
        this.createHolographicStructures();
    }

    public dispose(): void {
        this.hologramGroup.children.forEach(child => {
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
        });
        this.scene.remove(this.hologramGroup);
    }
}
