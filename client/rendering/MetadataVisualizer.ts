import { 
    Color, 
    Matrix4, 
    Mesh, 
    PerspectiveCamera, 
    Scene, 
    Vector3, 
    Material,
    MeshBasicMaterial,
    Quaternion
} from 'three';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { Metadata } from '../types/metadata';
import { Settings } from '../core/types';
import { defaultSettings } from '../state/defaultSettings';

export class MetadataVisualizer {
    private readonly camera: PerspectiveCamera;
    private readonly scene: Scene;
    private readonly geometryFactory: GeometryFactory;
    private readonly materialFactory: MaterialFactory;
    private readonly settings: Settings;
    private nodes: Map<string, Mesh> = new Map();

    constructor(camera: PerspectiveCamera, scene: Scene, settings: Settings = defaultSettings) {
        this.camera = camera;
        this.scene = scene;
        this.settings = settings;
        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
    }

    public createNodeMesh(metadata: Metadata): Mesh {
        const geometry = this.geometryFactory.getNodeGeometry(this.settings.hologram.desktopQuality);
        const material = this.materialFactory.getMetadataMaterial();
        
        const mesh = new Mesh(geometry, material);
        mesh.position.set(
            metadata.position?.x || 0,
            metadata.position?.y || 0,
            metadata.position?.z || 0
        );
        
        this.nodes.set(metadata.id, mesh);
        this.scene.add(mesh);
        
        return mesh;
    }

    public dispose(): void {
        this.nodes.forEach(mesh => {
            this.scene.remove(mesh);
            mesh.geometry.dispose();
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach(m => m.dispose());
            } else {
                mesh.material.dispose();
            }
        });
        this.nodes.clear();
    }

    public updateNodeMetadata(
        mesh: Mesh,
        _age: number,
        linkCount: number,
        material: Material
    ): void {
        // Calculate color based on link count
        if (material instanceof MeshBasicMaterial) {
            // Simple color interpolation based on link count
            const intensity = Math.min(linkCount / 10, 1); // Cap at 10 links
            
            // Convert RGB values to hex
            const red = Math.floor(intensity * 255);
            const green = Math.floor((1 - intensity) * 255);
            const blue = 0;
            const hexColor = (red << 16) | (green << 8) | blue;
            
            // Create and assign color
            const newColor = new Color(hexColor);
            material.color = newColor;
        }

        // Update mesh orientation to face camera
        const meshPosition = mesh.position;
        const cameraPosition = this.camera.position;

        // Calculate direction from mesh to camera
        const direction = new Vector3()
            .subVectors(cameraPosition, meshPosition)
            .normalize();

        // Calculate up vector (world up)
        const up = new Vector3(0, 1, 0);

        // Calculate right vector
        const right = new Vector3()
            .crossVectors(up, direction)
            .normalize();

        // Recalculate up vector to ensure orthogonality
        up.crossVectors(direction, right).normalize();

        // Create rotation matrix
        const rotationMatrix = new Matrix4();
        rotationMatrix.elements = [
            right.x, up.x, direction.x, 0,
            right.y, up.y, direction.y, 0,
            right.z, up.z, direction.z, 0,
            0, 0, 0, 1
        ];

        // Create quaternion from direction
        const quaternion = new Quaternion();
        const m = rotationMatrix.elements;
        const trace = m[0] + m[5] + m[10];

        if (trace > 0) {
            const s = 0.5 / Math.sqrt(trace + 1.0);
            quaternion.w = 0.25 / s;
            quaternion.x = (m[6] - m[9]) * s;
            quaternion.y = (m[8] - m[2]) * s;
            quaternion.z = (m[1] - m[4]) * s;
        } else {
            if (m[0] > m[5] && m[0] > m[10]) {
                const s = 2.0 * Math.sqrt(1.0 + m[0] - m[5] - m[10]);
                quaternion.w = (m[6] - m[9]) / s;
                quaternion.x = 0.25 * s;
                quaternion.y = (m[1] + m[4]) / s;
                quaternion.z = (m[8] + m[2]) / s;
            } else if (m[5] > m[10]) {
                const s = 2.0 * Math.sqrt(1.0 + m[5] - m[0] - m[10]);
                quaternion.w = (m[8] - m[2]) / s;
                quaternion.x = (m[1] + m[4]) / s;
                quaternion.y = 0.25 * s;
                quaternion.z = (m[6] + m[9]) / s;
            } else {
                const s = 2.0 * Math.sqrt(1.0 + m[10] - m[0] - m[5]);
                quaternion.w = (m[1] - m[4]) / s;
                quaternion.x = (m[8] + m[2]) / s;
                quaternion.y = (m[6] + m[9]) / s;
                quaternion.z = 0.25 * s;
            }
        }

        // Apply rotation
        mesh.quaternion.x = quaternion.x;
        mesh.quaternion.y = quaternion.y;
        mesh.quaternion.z = quaternion.z;
        mesh.quaternion.w = quaternion.w;
    }
}
