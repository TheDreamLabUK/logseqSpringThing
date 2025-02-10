import { 
    BufferGeometry,
    BufferAttribute,
    Vector3,
    Scene,
    Group,
    Object3D,
    Material,
    Mesh,
    MeshBasicMaterial,
    DoubleSide
} from 'three';
import { Edge } from '../core/types';
import { Settings } from '../types/settings';
import { HologramShaderMaterial } from './materials/HologramShaderMaterial';

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

    private createEdgeGeometry(source: Vector3, target: Vector3, isHologram: boolean = false): BufferGeometry {
        const geometry = new BufferGeometry();
        
        // Calculate direction and create vertices directly in world space
        const direction = new Vector3().subVectors(target, source);
        const width = this.settings.visualization.edges.baseWidth * (isHologram ? 0.15 : 0.1);
        
        // Calculate perpendicular vector for width
        const up = new Vector3(0, 1, 0);
        const right = new Vector3().crossVectors(direction, up).normalize().multiplyScalar(width);
        
        // Create vertices for a thin rectangular prism along the edge
        const vertices = new Float32Array([
            // Front face
            source.x - right.x, source.y - right.y, source.z - right.z,
            source.x + right.x, source.y + right.y, source.z + right.z,
            target.x + right.x, target.y + right.y, target.z + right.z,
            target.x - right.x, target.y - right.y, target.z - right.z,
            
            // Back face (slightly offset)
            source.x - right.x, source.y - right.y, source.z - right.z + 0.001,
            source.x + right.x, source.y + right.y, source.z + right.z + 0.001,
            target.x + right.x, target.y + right.y, target.z + right.z + 0.001,
            target.x - right.x, target.y - right.y, target.z - right.z + 0.001
        ]);
        
        // Create indices for both faces
        const indices = new Uint16Array([
            // Front face
            0, 1, 2,
            0, 2, 3,
            // Back face
            4, 6, 5,
            4, 7, 6,
            // Connect front to back
            0, 4, 1,
            1, 4, 5,
            1, 5, 2,
            2, 5, 6,
            2, 6, 3,
            3, 6, 7,
            3, 7, 0,
            0, 7, 4
        ]);
        
        geometry.setAttribute('position', new BufferAttribute(vertices, 3));
        geometry.setIndex(new BufferAttribute(indices, 1));
        
        // Calculate normals for proper lighting
        const normals = new Float32Array(vertices.length);
        for (let i = 0; i < vertices.length; i += 3) {
            // Set all normals to point outward from the edge
            normals[i] = right.x;
            normals[i + 1] = right.y;
            normals[i + 2] = right.z;
        }
        geometry.setAttribute('normal', new BufferAttribute(normals, 3));
        
        return geometry;
    }

    private createEdgeMaterial(isHologram: boolean = false): Material {
        if (isHologram) {
            return new HologramShaderMaterial({
                visualization: {
                    hologram: {
                        opacity: this.settings.visualization.edges.opacity,
                        color: this.settings.visualization.edges.color
                    },
                    edges: {
                        baseWidth: this.settings.visualization.edges.baseWidth
                    }
                }
            });
        }
        
        // Default edge material
        return new MeshBasicMaterial({
            color: this.settings.visualization.edges.color,
            transparent: true,
            opacity: this.settings.visualization.edges.opacity,
            side: DoubleSide
        });
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

            const isHologram = edge.type === 'hologram';
            const geometry = this.createEdgeGeometry(source, target, isHologram);
            const material = this.createEdgeMaterial(isHologram);
            const mesh = new Mesh(geometry, material);

            // Enable both layers for the edge
            mesh.layers.enable(0);
            mesh.layers.enable(1);
            
            this.edgeGroup.add(mesh);
            this.edges.set(edge.id, mesh);
        });
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        this.edges.forEach((edge) => {
            if (edge.material instanceof HologramShaderMaterial) {
                edge.material.uniforms.opacity.value = settings.visualization.edges.opacity;
                edge.material.uniforms.color.value.set(settings.visualization.edges.color);
                edge.material.uniforms.edgeWidth.value = settings.visualization.edges.baseWidth;
                edge.material.needsUpdate = true;
            } else if (edge.material instanceof MeshBasicMaterial) {
                edge.material.color.set(settings.visualization.edges.color);
                edge.material.opacity = settings.visualization.edges.opacity;
                edge.material.transparent = true;
                edge.material.side = DoubleSide;
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
