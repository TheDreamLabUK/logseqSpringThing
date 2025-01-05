import { Scene, Line, LineBasicMaterial, BufferGeometry, Float32BufferAttribute, Vector3, Color } from 'three';
import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';

export interface Edge {
    source: string;
    target: string;
    color?: string;
    width?: number;
}

export class EdgeManager {
    private logger = createLogger('EdgeManager');
    private scene: Scene;
    private settings: Settings;
    private edges: Map<string, Line>;

    constructor(scene: Scene, settings: Settings) {
        this.logger.debug('Initializing EdgeManager');
        this.scene = scene;
        this.settings = settings;
        this.edges = new Map();
    }

    initialize(edges: Edge[], nodePositions: Map<string, Vector3>) {
        this.logger.debug('Initializing edges', { count: edges.length });
        
        edges.forEach(edge => {
            const sourcePos = nodePositions.get(edge.source);
            const targetPos = nodePositions.get(edge.target);
            
            if (!sourcePos || !targetPos) {
                this.logger.warn('Missing position for edge', { edge });
                return;
            }

            this.createEdge(edge, sourcePos, targetPos);
        });
    }

    private createEdge(edge: Edge, sourcePos: Vector3, targetPos: Vector3) {
        const geometry = new BufferGeometry();
        const positions = new Float32Array([
            sourcePos.x, sourcePos.y, sourcePos.z,
            targetPos.x, targetPos.y, targetPos.z
        ]);
        
        geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));

        const material = new LineBasicMaterial({
            color: new Color(edge.color || this.settings.visualization.edges.color),
            linewidth: edge.width || this.settings.visualization.edges.defaultWidth,
            transparent: true,
            opacity: this.settings.visualization.edges.opacity
        });

        const line = new Line(geometry, material);
        const key = `${edge.source}-${edge.target}`;
        
        this.edges.set(key, line);
        this.scene.add(line);
    }

    updateEdgeVisibility(edge: Edge, visible: boolean) {
        const key = `${edge.source}-${edge.target}`;
        const line = this.edges.get(key);
        if (line) {
            line.visible = visible;
        }
    }

    dispose() {
        this.logger.debug('Disposing EdgeManager');
        
        this.edges.forEach(line => {
            this.scene.remove(line);
            line.geometry.dispose();
            (line.material as Material).dispose();
        });
        this.edges.clear();
    }

    // Add other necessary methods...
} 