import {
    BufferGeometry,
    BufferAttribute,
    Vector3,
    Scene,
    Group,
    Object3D,
    Material,
    Mesh,
    MeshBasicMaterial
} from 'three';
import { Edge } from '../core/types';
import { Settings } from '../types/settings';

export class EdgeManager {
    private scene: Scene;
    private edges: Map<string, Mesh> = new Map();
    private edgeGroup: Group;
    private settings: Settings;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
        this.edgeGroup = new Group();
        
        // Enable both layers by default for desktop mode
        this.edgeGroup.layers.enable(0);
        this.edgeGroup.layers.enable(1);
        
        scene.add(this.edgeGroup);
    }

    private createEdgeGeometry(source: Vector3, target: Vector3): BufferGeometry {
        const geometry = new BufferGeometry();
        
        // Calculate direction and create vertices directly in world space
        const direction = new Vector3().subVectors(target, source);
        const width = this.settings.visualization.edges.baseWidth * 0.1;
        
        // Calculate perpendicular vector for width
        const up = new Vector3(0, 1, 0);
        const right = new Vector3().crossVectors(direction, up).normalize().multiplyScalar(width);
        
        // Create vertices in world space
        const vertices = new Float32Array([
            source.x - right.x, source.y - right.y, source.z - right.z,
            source.x + right.x, source.y + right.y, source.z + right.z,
            target.x + right.x, target.y + right.y, target.z + right.z,
            target.x - right.x, target.y - right.y, target.z - right.z
        ]);
        
        const indices = new Uint16Array([
            0, 1, 2,
            0, 2, 3
        ]);
        
        geometry.setAttribute('position', new BufferAttribute(vertices, 3));
        geometry.setIndex(new BufferAttribute(indices, 1));
        
        return geometry;
    }

    public updateEdges(edges: Edge[]): void {
        // Clear existing edges
        this.edges.forEach(edge => {
            this.edgeGroup.remove(edge);
            edge.geometry.dispose();
            if (edge.material instanceof Material) {
                edge.material.dispose();
            }
        });
        this.edges.clear();

        // Create new edges
        edges.forEach(edge => {
            if (!edge.sourcePosition || !edge.targetPosition) return;

            const source = new Vector3(
                edge.sourcePosition.x,
                edge.sourcePosition.y,
                edge.sourcePosition.z
            );
            const target = new Vector3(
                edge.targetPosition.x,
                edge.targetPosition.y,
                edge.targetPosition.z
            );

            const geometry = this.createEdgeGeometry(source, target);
            
            const material = new MeshBasicMaterial({
                color: this.settings.visualization.edges.color,
                transparent: true,
                opacity: this.settings.visualization.edges.opacity
            });

            const line = new Mesh(geometry, material);
            // Enable both layers for the edge
            line.layers.enable(0);
            line.layers.enable(1);
            
            this.edgeGroup.add(line);
            this.edges.set(edge.id, line);
        });
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        this.edges.forEach(edge => {
            if (edge.material instanceof MeshBasicMaterial) {
                edge.material.color.set(settings.visualization.edges.color);
                edge.material.opacity = settings.visualization.edges.opacity;
                edge.material.needsUpdate = true;
            }
        });
    }

    public setXRMode(enabled: boolean): void {
        if (enabled) {
            // In XR mode, only show on layer 1
            this.edgeGroup.layers.disable(0);
            this.edgeGroup.layers.enable(1);
            this.edgeGroup.traverse((child: Object3D) => {
                child.layers.disable(0);
                child.layers.enable(1);
            });
        } else {
            // In desktop mode, show on both layers
            this.edgeGroup.layers.enable(0);
            this.edgeGroup.layers.enable(1);
            this.edgeGroup.traverse((child: Object3D) => {
                child.layers.enable(0);
                child.layers.enable(1);
            });
        }
    }

    public dispose(): void {
        this.edges.forEach(edge => {
            edge.geometry.dispose();
            if (edge.material instanceof Material) {
                edge.material.dispose();
            }
            this.edgeGroup.remove(edge);
        });
        this.edges.clear();
        this.scene.remove(this.edgeGroup);
    }
}
