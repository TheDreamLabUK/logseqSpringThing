import * as THREE from 'three';
import { visualizationSettings } from '../services/visualizationSettings.js';
import { LAYERS, LAYER_GROUPS, LayerManager } from './layerManager.js';

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
     * Initializes settings from server or defaults
     */
    initializeSettings(settings) {
        const nodeSettings = visualizationSettings.getNodeSettings();
        if (!nodeSettings) {
            console.warn('Using default node settings');
        }

        // Node appearance
        this.minNodeSize = settings.minNodeSize || nodeSettings?.minNodeSize || 0.15;
        this.maxNodeSize = settings.maxNodeSize || nodeSettings?.maxNodeSize || 0.4;
        this.nodeColor = new THREE.Color(settings.nodeColor || nodeSettings?.color || 0xffa500);
        
        // Material settings
        this.materialSettings = {
            metalness: nodeSettings?.material?.metalness || 0.3,
            roughness: nodeSettings?.material?.roughness || 0.5,
            clearcoat: nodeSettings?.material?.clearcoat || 0.8,
            opacity: nodeSettings?.material?.opacity || 0.95,
            emissiveMinIntensity: nodeSettings?.material?.emissiveMinIntensity || 0.0,
            emissiveMaxIntensity: nodeSettings?.material?.emissiveMaxIntensity || 0.3
        };

        // Age-based colors
        this.ageColors = {
            NEW: new THREE.Color(nodeSettings?.colorNew || 0x00ff88),
            RECENT: new THREE.Color(nodeSettings?.colorRecent || 0x4444ff),
            MEDIUM: new THREE.Color(nodeSettings?.colorMedium || 0xffaa00),
            OLD: new THREE.Color(nodeSettings?.colorOld || 0xff4444)
        };
        
        // Edge appearance
        const edgeSettings = visualizationSettings.getEdgeSettings();
        this.edgeColor = new THREE.Color(settings.edgeColor || edgeSettings?.color || 0xffffff);
        this.edgeOpacity = settings.edgeOpacity || edgeSettings?.opacity || 0.4;
        
        // Label settings
        this.labelFontSize = settings.labelFontSize || 32;
        this.maxAge = settings.maxAge || 30; // days
    }

    /**
     * Initializes instanced meshes for efficient rendering
     */
    initInstancedMeshes() {
        try {
            // Create geometries for different LOD levels
            const highDetailGeometry = new THREE.SphereGeometry(1, 32, 32);
            const mediumDetailGeometry = new THREE.SphereGeometry(1, 16, 16);
            const lowDetailGeometry = new THREE.SphereGeometry(1, 8, 8);

            // Create base material
            const nodeMaterial = new THREE.MeshPhysicalMaterial({
                metalness: this.materialSettings.metalness,
                roughness: this.materialSettings.roughness,
                transparent: true,
                opacity: this.materialSettings.opacity,
                clearcoat: this.materialSettings.clearcoat,
                clearcoatRoughness: 0.1,
                emissive: this.nodeColor,
                emissiveIntensity: this.materialSettings.emissiveMinIntensity
            });

            // Initialize instanced meshes
            const maxInstances = 10000;
            this.nodeInstancedMeshes = {
                high: new THREE.InstancedMesh(highDetailGeometry, nodeMaterial.clone(), maxInstances),
                medium: new THREE.InstancedMesh(mediumDetailGeometry, nodeMaterial.clone(), maxInstances),
                low: new THREE.InstancedMesh(lowDetailGeometry, nodeMaterial.clone(), maxInstances)
            };

            // Set up edge instanced mesh
            const edgeGeometry = new THREE.CylinderGeometry(0.15, 0.15, 1, 8);
            edgeGeometry.rotateX(Math.PI / 2);
            
            const edgeMaterial = new THREE.MeshBasicMaterial({
                color: this.edgeColor,
                transparent: true,
                opacity: this.edgeOpacity,
                depthWrite: false
            });

            this.edgeInstancedMesh = new THREE.InstancedMesh(
                edgeGeometry,
                edgeMaterial,
                maxInstances * 2
            );

            // Add meshes to container
            Object.values(this.nodeInstancedMeshes).forEach(mesh => {
                mesh.count = 0;
                LayerManager.setLayerGroup(mesh, 'BLOOM');
                this.instancedContainer.add(mesh);
            });

            LayerManager.setLayerGroup(this.edgeInstancedMesh, 'EDGE');
            this.instancedContainer.add(this.edgeInstancedMesh);

        } catch (error) {
            console.error('Error initializing instanced meshes:', error);
            throw error;
        }
    }

    /**
     * Initializes the label renderer
     */
    initializeLabelRenderer() {
        this.labelCanvas = document.createElement('canvas');
        this.labelContext = this.labelCanvas.getContext('2d', {
            alpha: true,
            desynchronized: true,
            willReadFrequently: false
        });
    }

    /**
     * Handles binary position updates from WebSocket
     * @param {Float32Array} positions - Array of [x,y,z,vx,vy,vz] values
     * @param {boolean} isInitialLayout - Whether this is the initial layout
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
