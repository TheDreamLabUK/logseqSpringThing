import { 
    Matrix4, 
    Mesh, 
    PerspectiveCamera, 
    Scene, 
    Vector3, 
    MeshBasicMaterial,
    Quaternion,
    BufferGeometry
} from 'three';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { NodeData, VisualizationSettings } from '../core/types';

export class MetadataVisualizer {
    private readonly camera: PerspectiveCamera;
    private readonly scene: Scene;
    private readonly geometryFactory: GeometryFactory;
    private nodes: Map<string, Mesh> = new Map();
    private nodeGeometry: BufferGeometry;
    private readonly materialFactory: MaterialFactory;

    constructor(camera: PerspectiveCamera, scene: Scene, settings: VisualizationSettings) {
        this.camera = camera;
        this.scene = scene;
        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        this.nodeGeometry = this.geometryFactory.getNodeGeometry('medium');
    }

    public addNodeMetadata(node: NodeData): void {
        if (this.nodes.has(node.id)) {
            return;
        }

        const material = this.materialFactory.createNodeMaterial('basic', node.color);
        const mesh = new Mesh(this.nodeGeometry, material);
        
        mesh.position.copy(node.position);
        mesh.scale.setScalar(node.size || 1.0);
        
        // Make the metadata mesh always face the camera
        this.updateMetadataOrientation(mesh);
        
        this.nodes.set(node.id, mesh);
        this.scene.add(mesh);
    }

    public removeNodeMetadata(nodeId: string): void {
        const mesh = this.nodes.get(nodeId);
        if (mesh) {
            this.scene.remove(mesh);
            mesh.geometry.dispose();
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach(m => m.dispose());
            } else {
                mesh.material.dispose();
            }
            this.nodes.delete(nodeId);
        }
    }

    public updateMetadataOrientation(mesh: Mesh): void {
        // Calculate direction from mesh to camera
        const direction = new Vector3()
            .subVectors(this.camera.position, mesh.position)
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
        const rotationMatrix = new Matrix4().makeBasis(right, up, direction);
        const quaternion = new Quaternion().setFromRotationMatrix(rotationMatrix);

        // Apply rotation
        mesh.quaternion.copy(quaternion);
    }

    public updateSettings(settings: VisualizationSettings): void {
        this.settings = settings;
        this.nodes.forEach((mesh) => {
            if (mesh.material instanceof MeshBasicMaterial) {
                mesh.material.opacity = settings.nodes.material?.opacity || 1;
                mesh.material.transparent = settings.nodes.material?.transparent || false;
                mesh.material.needsUpdate = true;
            }
        });
    }

    public dispose(): void {
        this.nodes.forEach(mesh => {
            this.scene.remove(mesh);
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach(m => m.dispose());
            } else {
                mesh.material.dispose();
            }
        });
        this.nodes.clear();
        this.nodeGeometry.dispose();
    }
}
