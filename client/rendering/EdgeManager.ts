import * as THREE from 'three';
import {
    Scene,
    Vector3,
    Color,
    createBufferAttribute,
    Object3D
} from '../core/threeTypes';
import { MaterialFactory } from './factories/MaterialFactory';
import { EdgeData } from '../core/types';
import { Settings } from '../types/settings';

type EdgeObject = THREE.Line & {
    material: THREE.LineBasicMaterial;
    geometry: THREE.BufferGeometry;
};

export class EdgeManager {
    private edges: Map<string, EdgeObject> = new Map();
    private readonly scene: Scene;
    private readonly settings: Settings;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
    }

    public createEdge(
        id: string,
        source: Vector3,
        target: Vector3,
        color?: Color
    ): void {
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array([
            source.x, source.y, source.z,
            target.x, target.y, target.z
        ]);
        geometry.setAttribute('position', createBufferAttribute(vertices, 3));

        // Create material
        const material = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.6
        });

        // Create line
        const line = new THREE.Line(geometry);
        line.material = material;
        line.userData = { id };

        // Store and add to scene
        this.edges.set(id, line as EdgeObject);
        this.scene.add(line as unknown as Object3D);
    }

    public updateEdge(
        id: string,
        source: Vector3,
        target: Vector3,
        color?: Color
    ): void {
        const edge = this.edges.get(id);
        if (!edge) return;

        // Update geometry
        const vertices = new Float32Array([
            source.x, source.y, source.z,
            target.x, target.y, target.z
        ]);
        edge.geometry.setAttribute('position', createBufferAttribute(vertices, 3));
        edge.geometry.attributes.position.needsUpdate = true;

        // Update material
        if (color) {
            edge.material.color.copy(color);
            edge.material.needsUpdate = true;
        }
    }

    public removeEdge(id: string): void {
        const edge = this.edges.get(id);
        if (!edge) return;

        edge.geometry.dispose();
        edge.material.dispose();

        this.scene.remove(edge as unknown as Object3D);
        this.edges.delete(id);
    }

    public getEdge(id: string): EdgeObject | undefined {
        return this.edges.get(id);
    }

    public clear(): void {
        this.edges.forEach(edge => {
            edge.geometry.dispose();
            edge.material.dispose();
            this.scene.remove(edge as unknown as Object3D);
        });
        this.edges.clear();
    }

    public updateEdgeVisibility(visible: boolean): void {
        this.edges.forEach(edge => {
            edge.visible = visible;
        });
    }

    public updateEdgeOpacity(opacity: number): void {
        this.edges.forEach(edge => {
            edge.material.opacity = opacity;
            edge.material.needsUpdate = true;
        });
    }

    public getEdgesByData(data: Partial<EdgeData>): EdgeObject[] {
        return Array.from(this.edges.values()).filter(edge => {
            const userData = edge.userData as EdgeData;
            return Object.entries(data).every(([key, value]) => {
                return key in userData && userData[key as keyof EdgeData] === value;
            });
        });
    }

    public dispose(): void {
        this.clear();
    }
}
