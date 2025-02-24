import { 
    BufferGeometry,
    BufferAttribute,
    Vector3,
    Scene,
    Group,
    Object3D,
    Material,
    Mesh
} from 'three';
import { Edge } from '../core/types';
import { Settings } from '../types/settings';
import { EdgeShaderMaterial } from './materials/EdgeShaderMaterial';
import { NodeInstanceManager } from './node/instance/NodeInstanceManager';
import { SettingsStore } from '../state/SettingsStore';

export class EdgeManager {
    private scene: Scene;
    private edges: Map<string, Mesh> = new Map();
    private edgeGroup: Group;
    private nodeManager: NodeInstanceManager;
    private edgeData: Map<string, Edge> = new Map();
    private settings: Settings;
    private settingsStore: SettingsStore;
    private updateFrameCount = 0;
    private readonly UPDATE_FREQUENCY = 2; // Update every other frame

    constructor(scene: Scene, settings: Settings, nodeManager: NodeInstanceManager) {
        this.scene = scene;
        this.nodeManager = nodeManager;
        this.settings = settings;
        this.settingsStore = SettingsStore.getInstance();
        this.edgeGroup = new Group();
        
        // Enable both layers by default for desktop mode
        this.edgeGroup.layers.enable(0);
        this.edgeGroup.layers.enable(1);
        
        scene.add(this.edgeGroup);

        // Subscribe to settings changes
        this.settingsStore.subscribe('visualization.edges', (_: string, settings: any) => {
            if (settings && typeof settings === 'object') {
                this.settings = {
                    ...this.settings,
                    visualization: {
                        ...this.settings.visualization,
                        edges: settings
                    }
                };
                this.handleSettingsUpdate(this.settings);
                this.updateAllEdgeGeometries();
            }
        });
    }

    private getEdgeWidth(): number {
        return this.settings.visualization.edges.baseWidth || 0.005; // Default width in meters (5mm)
    }

    private createEdgeGeometry(source: Vector3, target: Vector3): BufferGeometry {
        const geometry = new BufferGeometry();
        const direction = new Vector3().subVectors(target, source).normalize();
        const width = this.getEdgeWidth();

        // Calculate perpendicular vector for width
        const up = new Vector3(0, 1, 0);
        const right = new Vector3().crossVectors(direction, up).normalize().multiplyScalar(width / 2);

        // Create vertices for a thin rectangular prism along the edge
        const vertices = new Float32Array([
            // Front face
            source.x - right.x, source.y - right.y, source.z - right.z,
            source.x + right.x, source.y + right.y, source.z + right.z,
            target.x + right.x, target.y + right.y, target.z + right.z,
            target.x - right.x, target.y - right.y, target.z - right.z,
            
            // Back face (slightly offset)
            source.x - right.x, source.y - right.y, source.z - right.z + width,
            source.x + right.x, source.y + right.y, source.z + right.z + width,
            target.x + right.x, target.y + right.y, target.z + right.z + width,
            target.x - right.x, target.y - right.y, target.z - right.z + width
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

    private createEdgeMaterial(): Material {
        return new EdgeShaderMaterial(this.settings);
    }

    private updateAllEdgeGeometries(): void {
        this.edgeData.forEach((edgeData, edgeId) => {
            const mesh = this.edges.get(edgeId);
            if (!mesh) return;

            const sourcePos = this.nodeManager.getNodePosition(edgeData.source);
            const targetPos = this.nodeManager.getNodePosition(edgeData.target);

            if (sourcePos && targetPos) {
                // Update edge geometry
                const oldGeometry = mesh.geometry;
                mesh.geometry = this.createEdgeGeometry(sourcePos, targetPos);
                oldGeometry.dispose();

                // Update shader material source/target
                if (mesh.material instanceof EdgeShaderMaterial) {
                    mesh.material.setSourceTarget(sourcePos, targetPos);
                }
            }
        });
    }

    public updateEdges(edges: Edge[]): void {
        // Clear existing edges
        this.edgeData.clear();
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
            const material = this.createEdgeMaterial();
            const mesh = new Mesh(geometry, material);

            // Enable both layers for the edge
            mesh.layers.enable(0);
            mesh.layers.enable(1);
            
            this.edgeGroup.add(mesh);
            
            // Set source and target positions for the shader
            if (material instanceof EdgeShaderMaterial) {
                material.setSourceTarget(source, target);
            }
            this.edges.set(edge.id, mesh);
            this.edgeData.set(edge.id, edge);
        });
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        this.edges.forEach((edge) => {
            if (edge.material instanceof EdgeShaderMaterial) {
                // Update the material properties directly
                edge.material.opacity = settings.visualization.edges.opacity;
                
                // Try to update color
                try {
                  // Use the color property directly since we're now extending MeshBasicMaterial
                  edge.material.color.set(settings.visualization.edges.color);
                } catch (error) {
                  console.warn('Could not update edge material color');
                }
                
                // Mark material as needing update
                edge.material.needsUpdate = true;
            }
        });
    }
    
    public update(deltaTime: number): void {
        this.updateFrameCount++;
        if (this.updateFrameCount % this.UPDATE_FREQUENCY !== 0) return;
        
        // Update edge positions based on current node positions
        this.edgeData.forEach((edgeData, edgeId) => {
            const mesh = this.edges.get(edgeId);
            if (!mesh) return;

            const sourcePos = this.nodeManager.getNodePosition(edgeData.source);
            const targetPos = this.nodeManager.getNodePosition(edgeData.target);

            if (sourcePos && targetPos) {
                // Update edge geometry
                const oldGeometry = mesh.geometry;
                mesh.geometry.dispose();
                
                // Create new geometry and update mesh
                mesh.geometry = this.createEdgeGeometry(sourcePos, targetPos);
                
                // Clean up old resources after successful update
                oldGeometry.dispose();

                // Update shader material source/target
                if (mesh.material instanceof EdgeShaderMaterial) {
                    mesh.material.setSourceTarget(sourcePos, targetPos);
                    mesh.material.update(deltaTime * this.UPDATE_FREQUENCY);
                }
            }
            // If positions not found, edge will remain at last known position
            else if (mesh.material instanceof EdgeShaderMaterial) {
                mesh.material.update(deltaTime * this.UPDATE_FREQUENCY);
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
        this.clearEdges();
        this.scene.remove(this.edgeGroup);
    }

    private clearEdges(): void {
        this.edges.forEach(edge => {
            if (edge) {
                // Remove from group first
                this.edgeGroup.remove(edge);
                
                // Dispose of geometry
                if (edge.geometry) {
                    edge.geometry.dispose();
                }
                
                // Dispose of material
                if (edge.material instanceof Material) {
                    edge.material.dispose();
                }
            }
        });
        this.edges.clear();
    }
}
