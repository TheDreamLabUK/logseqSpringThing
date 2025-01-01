import * as THREE from 'three';
import { Scene } from 'three';
import { Settings } from '../types/settings';
import { Edge } from '../core/types';

export class EdgeManager {
    private edges = new Map<string, THREE.Line>();
    private scene: Scene;
    private settings: Settings;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
    }

    addEdge(edge: Edge): void {
        const key = `${edge.source}-${edge.target}`;
        if (this.edges.has(key)) {
            return;
        }

        const positions = new Float32Array(6); // 2 points * 3 coordinates
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const material = new THREE.LineBasicMaterial({
            color: this.settings.visualization.edges.color,
            transparent: true,
            opacity: 0.6
        });

        const line = new THREE.Line(geometry, material);
        this.edges.set(key, line);
        this.scene.add(line);
    }

    removeEdge(edge: Edge): void {
        const key = `${edge.source}-${edge.target}`;
        const line = this.edges.get(key);
        if (line) {
            this.scene.remove(line);
            if (line.geometry) {
                line.geometry.dispose();
            }
            if (line.material) {
                if (Array.isArray(line.material)) {
                    line.material.forEach(m => m.dispose());
                } else {
                    line.material.dispose();
                }
            }
            this.edges.delete(key);
        }
    }

    updateEdgePosition(edge: Edge, sourcePos: THREE.Vector3, targetPos: THREE.Vector3): void {
        const key = `${edge.source}-${edge.target}`;
        const line = this.edges.get(key);
        if (!line || !line.geometry) return;

        const positions = new Float32Array([
            sourcePos.x, sourcePos.y, sourcePos.z,
            targetPos.x, targetPos.y, targetPos.z
        ]);

        (line.geometry.getAttribute('position') as THREE.BufferAttribute).set(positions);
        line.geometry.getAttribute('position').needsUpdate = true;
        line.geometry.computeBoundingSphere();
    }

    handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        this.edges.forEach(line => {
            if (line.material) {
                if (Array.isArray(line.material)) {
                    line.material.forEach(m => {
                        if (m instanceof THREE.LineBasicMaterial) {
                            m.color.set(settings.visualization.edges.color);
                        }
                    });
                } else if (line.material instanceof THREE.LineBasicMaterial) {
                    line.material.color.set(settings.visualization.edges.color);
                }
            }
        });
    }

    dispose(): void {
        this.edges.forEach(line => {
            this.scene.remove(line);
            if (line.geometry) {
                line.geometry.dispose();
            }
            if (line.material) {
                if (Array.isArray(line.material)) {
                    line.material.forEach(m => m.dispose());
                } else {
                    line.material.dispose();
                }
            }
        });
        this.edges.clear();
    }
}
