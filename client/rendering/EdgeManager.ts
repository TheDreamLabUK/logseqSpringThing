import {
    Scene,
    BufferGeometry,
    Line,
    Vector3,
    Material,
    Color,
    Object3D
} from 'three';
import { MaterialFactory } from './factories/MaterialFactory';
import { EdgeData } from '../core/types';
import { Settings } from '../types/settings';

export class EdgeManager {
    private edges: Map<string, Line> = new Map();
    private readonly scene: Scene;
    private readonly materialFactory: MaterialFactory;
    private readonly settings: Settings;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
        this.materialFactory = new MaterialFactory(settings);
    }

    public createEdge(
        id: string,
        source: Vector3,
        target: Vector3,
        color?: Color
    ): void {
        const geometry = new BufferGeometry();
        const vertices = new Float32Array([
            source.x, source.y, source.z,
            target.x, target.y, target.z
        ]);
        geometry.setAttribute('position', { itemSize: 3, array: vertices });

        const material = this.materialFactory.createEdgeMaterial(color);
        const line = new Line(geometry, material);
        line.userData = { id };

        this.edges.set(id, line);
        this.scene.add(line);
    }

    public updateEdge(
        id: string,
        source: Vector3,
        target: Vector3,
        color?: Color
    ): void {
        const edge = this.edges.get(id);
        if (!edge) return;

        const geometry = edge.geometry;
        const vertices = new Float32Array([
            source.x, source.y, source.z,
            target.x, target.y, target.z
        ]);
        geometry.setAttribute('position', { itemSize: 3, array: vertices });
        geometry.attributes.position.needsUpdate = true;

        if (color) {
            const material = edge.material as Material;
            material.color = color;
            material.needsUpdate = true;
        }
    }

    public removeEdge(id: string): void {
        const edge = this.edges.get(id);
        if (!edge) return;

        edge.geometry.dispose();
        if (Array.isArray(edge.material)) {
            edge.material.forEach(material => material.dispose());
        } else {
            edge.material.dispose();
        }

        this.scene.remove(edge);
        this.edges.delete(id);
    }

    public getEdge(id: string): Line | undefined {
        return this.edges.get(id);
    }

    public clear(): void {
        this.edges.forEach(edge => {
            edge.geometry.dispose();
            if (Array.isArray(edge.material)) {
                edge.material.forEach(material => material.dispose());
            } else {
                edge.material.dispose();
            }
            this.scene.remove(edge);
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
            if (Array.isArray(edge.material)) {
                edge.material.forEach(material => {
                    material.opacity = opacity;
                    material.needsUpdate = true;
                });
            } else {
                edge.material.opacity = opacity;
                edge.material.needsUpdate = true;
            }
        });
    }

    public getEdgesByData(data: Partial<EdgeData>): Line[] {
        return Array.from(this.edges.values()).filter(edge => {
            const userData = edge.userData as EdgeData;
            return Object.entries(data).every(([key, value]) => userData[key] === value);
        });
    }

    public dispose(): void {
        this.clear();
    }
}
