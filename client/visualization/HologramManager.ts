import * as THREE from 'three';
import { GeometryFactory } from '../rendering/factories/GeometryFactory';
import { MaterialFactory } from '../rendering/factories/MaterialFactory';

export class HologramManager {
    private readonly hologramGroup: THREE.Group;
    private readonly settings: any;
    private isXRMode: boolean = false;
    private readonly geometryFactory: GeometryFactory;
    private readonly materialFactory: MaterialFactory;
    
    // Store scene objects
    private rings: THREE.Mesh[] = [];
    private spheres: THREE.Mesh[] = [];
    private ringQuaternions: THREE.Quaternion[] = [];

    constructor(
        private readonly scene: THREE.Scene,
        settings: any
    ) {
        this.settings = settings;
        this.hologramGroup = new THREE.Group();
        this.scene.add(this.hologramGroup);
        this.hologramGroup.layers.set(this.isXRMode ? 1 : 0);
        
        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        
        this.createHolographicStructures();
    }

    private createHolographicStructures(): void {
        // Clean up existing objects
        this.rings.forEach(ring => {
            ring.geometry.dispose();
            ring.material.dispose();
            this.hologramGroup.remove(ring);
        });
        this.spheres.forEach(sphere => {
            sphere.geometry.dispose();
            sphere.material.dispose();
            this.hologramGroup.remove(sphere);
        });
        this.rings = [];
        this.spheres = [];

        const quality = this.isXRMode ? 'high' : (this.settings.quality || 'medium');

        // Create spheres
        const sphereRadii = [40, 100, 200];
        sphereRadii.forEach(radius => {
            const geometry = this.geometryFactory.getHologramGeometry('innerSphere', quality);
            const material = this.materialFactory.getSceneSphereMaterial(this.settings);
            const sphere = new THREE.Mesh(geometry, material);
            sphere.scale.setScalar(radius / 40); // Base geometry is radius 40
            sphere.layers.set(this.isXRMode ? 1 : 0);
            this.hologramGroup.add(sphere);
            this.spheres.push(sphere);
        });

        // Create rings
        sphereRadii.forEach(radius => {
            const scale = radius / 40; // Base ring geometry is radius 40
            for (let i = 0; i < 3; i++) {
                const geometry = this.geometryFactory.getHologramGeometry('ring', quality);
                const material = this.materialFactory.getRingMaterial(this.settings);
                const ring = new THREE.Mesh(geometry, material);
                
                ring.scale.setScalar(scale);
                ring.layers.set(this.isXRMode ? 1 : 0);
                const quaternion = new THREE.Quaternion();
                quaternion.setFromEuler(new THREE.Euler(Math.PI / 3 * i, Math.PI / 4 * i, 0));
                ring.quaternion.copy(quaternion);
                this.ringQuaternions.push(quaternion);
                
                this.hologramGroup.add(ring);
                this.rings.push(ring);
            }
        });
    }
    public setXRMode(enabled: boolean): void {
        this.isXRMode = enabled;
        this.hologramGroup.layers.set(enabled ? 1 : 0);
        [...this.rings, ...this.spheres].forEach(mesh => {
            mesh.layers.set(enabled ? 1 : 0);
        });
        this.createHolographicStructures();
    }


    public update(deltaTime: number): void {
        // Update ring rotations using quaternions
        const rotationQuat = new THREE.Quaternion();
        rotationQuat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), deltaTime * 0.1);
        
        this.rings.forEach((ring, index) => {
            this.ringQuaternions[index].multiply(rotationQuat);
            ring.quaternion.copy(this.ringQuaternions[index]);
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

        [...this.rings, ...this.spheres].forEach(mesh => {
            const distance = position.distanceTo(mesh.position);
            if (distance < 0.1) {
                if (mesh.material instanceof THREE.MeshBasicMaterial) {
                    const originalOpacity = mesh.material.opacity;
                    mesh.material.opacity = Math.min(originalOpacity * 2, 1);
                    setTimeout(() => {
                        mesh.material.opacity = originalOpacity;
                    }, 200);
                }
            }
        });
    }

    public updateSettings(settings: any): void {
        Object.assign(this.settings, settings);
        this.createHolographicStructures();
    }

    public dispose(): void {
        this.rings.forEach(ring => {
            ring.geometry.dispose();
            ring.material.dispose();
        });
        this.spheres.forEach(sphere => {
            sphere.geometry.dispose();
            sphere.material.dispose();
        });
        this.scene.remove(this.hologramGroup);
    }
}
