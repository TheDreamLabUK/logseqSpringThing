import * as THREE from 'three';
import { visualizationSettings } from '../services/visualizationSettings.js';
import { LAYERS, LAYER_GROUPS, LayerManager } from './layerManager.js';
import { TextRenderer } from './textRenderer.js';

/**
 * NodeManager handles the efficient rendering and updating of nodes and edges in the graph visualization.
 * It uses THREE.js InstancedMesh for performance and supports both regular updates and binary position updates.
 */
export class NodeManager {
    constructor(scene, camera, settings = {}) {
        // Core references
        this.scene = scene;
        this.camera = camera;
        
        // Data structures
        this.nodeData = new Map();          // Stores node data
        this.labelPool = new Map();         // Reusable label sprites
        this.instanceIds = new Map();       // Maps positions to node IDs
        
        // Rendering structures
        this.nodeInstancedMeshes = null;    // Different LOD meshes for nodes
        this.edgeInstancedMesh = null;      // Single mesh for all edges
        this.instancedContainer = null;      // Container for all instanced meshes
        
        // Binary update optimization
        this.instancePositions = new Float32Array(30000);  // Pre-allocated for position updates
        this.instanceSizes = new Float32Array(10000);      // Pre-allocated for size updates
        this._labelUpdateTimeout = null;                   // For throttling label updates
        
        // Reusable objects for matrix operations
        this.matrix = new THREE.Matrix4();
        this.quaternion = new THREE.Quaternion();
        this.position = new THREE.Vector3();
        this.scale = new THREE.Vector3();
        this.color = new THREE.Color();

        // Text renderer for labels
        this.textRenderer = new TextRenderer();
        
        // Initialize settings
        this.initializeSettings(settings);
        
        // Create container and initialize meshes
        this.instancedContainer = new THREE.Group();
        this.instancedContainer.name = 'instancedContainer';
        this.scene.add(this.instancedContainer);
        
        // Initialize instanced meshes
        this.initInstancedMeshes();
        
        // Set up label rendering
        this.initializeLabelRenderer();
        
        // Bind methods
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        this.updateNodePositions = this.updateNodePositions.bind(this);
        
        // Add event listeners
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
    }

    /**
     * Creates or updates a label for a node
     */
    createNodeLabel(nodeId, node) {
        let label = this.labelPool.get(nodeId);
        
        // Create label text with metadata
        let labelText = node.label || nodeId;
        if (node.metadata && Object.keys(node.metadata).length > 0) {
            labelText += '\n' + Object.entries(node.metadata)
                .map(([key, value]) => `${key}: ${value}`)
                .join('\n');
        }

        // Create or update label sprite
        if (!label) {
            label = this.textRenderer.createTextSprite(labelText, {
                fontSize: this.labelFontSize,
                color: 0xffffff,
                backgroundColor: 0x000000,
                backgroundOpacity: 0.85
            });
            
            // Make label always face camera
            label.material.depthWrite = false;
            label.material.depthTest = false;
            label.renderOrder = 1;
            
            this.labelPool.set(nodeId, label);
            this.scene.add(label);
        } else {
            // Update existing label
            const newSprite = this.textRenderer.createTextSprite(labelText, {
                fontSize: this.labelFontSize,
                color: 0xffffff,
                backgroundColor: 0x000000,
                backgroundOpacity: 0.85
            });
            
            // Update material and texture
            if (label.material) {
                if (label.material.map) label.material.map.dispose();
                label.material.dispose();
            }
            label.material = newSprite.material;
            label.scale.copy(newSprite.scale);
        }

        // Position label above node
        const nodeSize = this.getNodeSize(node.metadata || {});
        const pos = new THREE.Vector3(node.x, node.y, node.z);
        label.position.copy(pos).add(new THREE.Vector3(0, nodeSize * 1.5, 0));
        
        // Make label face camera
        label.quaternion.copy(this.camera.quaternion);
        
        return label;
    }

    /**
     * Updates labels for all nodes
     */
    updateLabels() {
        this.nodeData.forEach((node, nodeId) => {
            this.createNodeLabel(nodeId, node);
        });
    }

    /**
     * Handles binary position updates from WebSocket
     */
    updateNodePositions(positions, isInitialLayout = false) {
        if (!this.nodeInstancedMeshes) return;

        const matrix = this.matrix;
        const position = this.position;
        const quaternion = this.quaternion;
        const updatedNodes = new Set();

        // Update positions in batches
        for (let i = 0; i < positions.length; i++) {
            const nodeId = Array.from(this.nodeData.keys())[i];
            if (!nodeId) continue;

            const pos = positions[i];
            const node = this.nodeData.get(nodeId);
            
            // Update node data
            node.x = pos[0];
            node.y = pos[1];
            node.z = pos[2];
            node.vx = pos[3];
            node.vy = pos[4];
            node.vz = pos[5];

            // Update instance matrix
            position.set(pos[0], pos[1], pos[2]);
            const size = this.getNodeSize(node.metadata || {});
            this.scale.set(size, size, size);
            matrix.compose(position, quaternion, this.scale);

            // Update appropriate LOD mesh
            const distance = this.camera.position.distanceTo(position);
            let targetMesh;
            if (distance < 50) targetMesh = this.nodeInstancedMeshes.high;
            else if (distance < 100) targetMesh = this.nodeInstancedMeshes.medium;
            else targetMesh = this.nodeInstancedMeshes.low;

            targetMesh.setMatrixAt(i, matrix);

            // Track updates
            updatedNodes.add(nodeId);

            // Update instance lookup data
            const posIndex = i * 3;
            this.instancePositions[posIndex] = pos[0];
            this.instancePositions[posIndex + 1] = pos[1];
            this.instancePositions[posIndex + 2] = pos[2];
            this.instanceIds.set(`${[pos[0], pos[1], pos[2]]}`, nodeId);

            // Update label position
            const label = this.labelPool.get(nodeId);
            if (label) {
                label.position.copy(position).add(new THREE.Vector3(0, size * 1.5, 0));
                label.quaternion.copy(this.camera.quaternion);
            }
        }

        // Update instance matrices
        Object.values(this.nodeInstancedMeshes).forEach(mesh => {
            mesh.instanceMatrix.needsUpdate = true;
        });

        // Update edges and labels
        if (updatedNodes.size > 0) {
            this.updateEdgesForNodes(updatedNodes);
            this.throttledLabelUpdate(updatedNodes);
        }

        // Center camera on initial layout
        if (isInitialLayout) {
            this.centerCamera();
        }
    }

    /**
     * Updates edges connected to moved nodes
     * @param {Set<string>} updatedNodes - Set of node IDs that moved
     */
    updateEdgesForNodes(updatedNodes) {
        if (!this.edgeInstancedMesh) return;

        const matrix = this.matrix;
        const position = this.position;
        const quaternion = this.quaternion;
        const scale = this.scale;
        let edgeIndex = 0;

        // Update only affected edges
        this.nodeData.forEach((sourceNode, sourceId) => {
            this.nodeData.forEach((targetNode, targetId) => {
                if (sourceId === targetId) return;
                if (!updatedNodes.has(sourceId) && !updatedNodes.has(targetId)) return;

                const start = new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z);
                const end = new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z);
                const direction = end.clone().sub(start);
                const length = direction.length();

                if (length === 0) return;

                // Update edge transform
                const center = start.clone().add(end).multiplyScalar(0.5);
                position.copy(center);
                direction.normalize();
                quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
                scale.set(0.15, length, 0.15);

                matrix.compose(position, quaternion, scale);
                this.edgeInstancedMesh.setMatrixAt(edgeIndex++, matrix);
            });
        });

        this.edgeInstancedMesh.count = edgeIndex;
        this.edgeInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    /**
     * Throttled label update to improve performance
     * @param {Set<string>} updatedNodes - Set of node IDs that need label updates
     */
    throttledLabelUpdate(updatedNodes) {
        if (this._labelUpdateTimeout) return;

        this._labelUpdateTimeout = setTimeout(() => {
            updatedNodes.forEach(nodeId => {
                const node = this.nodeData.get(nodeId);
                if (!node) return;

                const label = this.labelPool.get(nodeId);
                if (label) {
                    const size = this.getNodeSize(node.metadata || {});
                    label.position.set(node.x, node.y + size * 1.5, node.z);
                    label.visible = true;
                }
            });
            this._labelUpdateTimeout = null;
        }, 100);
    }

    dispose() {
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        window.removeEventListener('xrsessionstart', () => this.handleXRStateChange(true));
        window.removeEventListener('xrsessionend', () => this.handleXRStateChange(false));
        
        if (this.xrController) {
            this.xrController.removeEventListener('select', this.handleXRSelect);
            this.scene.remove(this.xrController);
        }

        Object.values(this.nodeInstancedMeshes).forEach(mesh => {
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
            this.instancedContainer.remove(mesh);
        });

        if (this.edgeInstancedMesh) {
            if (this.edgeInstancedMesh.geometry) this.edgeInstancedMesh.geometry.dispose();
            if (this.edgeInstancedMesh.material) this.edgeInstancedMesh.material.dispose();
            this.instancedContainer.remove(this.edgeInstancedMesh);
        }

        this.labelPool.forEach(label => {
            if (label.material) {
                if (label.material.map) label.material.map.dispose();
                label.material.dispose();
            }
            this.scene.remove(label);
        });

        this.scene.remove(this.instancedContainer);
        
        this.nodeData.clear();
        this.labelPool.clear();
        this.instanceIds.clear();
    }
}
