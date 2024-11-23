// public/js/threeJS/threeGraph.js

import * as THREE from 'three';
import { visualizationSettings } from '../services/visualizationSettings.js';

/**
 * ForceGraph class manages the creation and updating of nodes and edges in the Three.js scene.
 */
export class ForceGraph {
    /**
     * Creates a new ForceGraph instance.
     * @param {THREE.Scene} scene - The Three.js scene.
     */
    constructor(scene) {
        this.scene = scene;

        // Data structures
        this.nodes = [];
        this.links = [];

        // Meshes
        this.nodeMeshes = new Map();
        this.linkMeshes = new Map();

        // Object pools with pre-allocation
        this.nodeMeshPool = [];
        this.linkMeshPool = [];
        this.geometryPool = new Map(); // Pool for reusing geometries
        this.materialPool = new Map(); // Pool for reusing materials

        // Level of Detail
        this.lod = new THREE.LOD();
        this.scene.add(this.lod);

        // Shared geometry for instancing
        this.sharedNodeGeometry = null;
        this.sharedEdgeGeometry = null;

        // Get settings
        const nodeSettings = visualizationSettings.getNodeSettings();
        const edgeSettings = visualizationSettings.getEdgeSettings();
        
        // Store settings
        this.nodeColors = {
            NEW: new THREE.Color(nodeSettings.colorNew),
            RECENT: new THREE.Color(nodeSettings.colorRecent),
            MEDIUM: new THREE.Color(nodeSettings.colorMedium),
            OLD: new THREE.Color(nodeSettings.colorOld),
            CORE: new THREE.Color(nodeSettings.colorCore),
            SECONDARY: new THREE.Color(nodeSettings.colorSecondary),
            DEFAULT: new THREE.Color(nodeSettings.colorDefault)
        };
        this.edgeColor = new THREE.Color(edgeSettings.color);
        this.edgeOpacity = edgeSettings.opacity;
        this.minNodeSize = nodeSettings.minNodeSize;  // In meters (0.1m = 10cm)
        this.maxNodeSize = nodeSettings.maxNodeSize;  // In meters (0.3m = 30cm)
        this.materialSettings = nodeSettings.material;

        // Initialize shared resources
        this.initSharedResources();
    }

    /**
     * Initialize shared geometries and materials
     */
    initSharedResources() {
        // Create shared node geometry with different LOD levels
        const highDetail = new THREE.SphereGeometry(1, 32, 32);
        const mediumDetail = new THREE.SphereGeometry(1, 16, 16);
        const lowDetail = new THREE.SphereGeometry(1, 8, 8);

        this.geometryPool.set('node-high', highDetail);
        this.geometryPool.set('node-medium', mediumDetail);
        this.geometryPool.set('node-low', lowDetail);

        // Create shared edge geometry
        const edgeGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(6);
        edgeGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.geometryPool.set('edge', edgeGeometry);
    }

    /**
     * Get or create a geometry from the pool
     * @param {string} type - The type of geometry
     * @param {number} size - The size for node geometries
     * @returns {THREE.BufferGeometry}
     */
    getGeometry(type, size = 1) {
        const key = `${type}-${size}`;
        if (this.geometryPool.has(key)) {
            return this.geometryPool.get(key);
        }

        let geometry;
        switch (type) {
            case 'node-high':
                geometry = new THREE.SphereGeometry(size, 32, 32);
                break;
            case 'node-medium':
                geometry = new THREE.SphereGeometry(size, 16, 16);
                break;
            case 'node-low':
                geometry = new THREE.SphereGeometry(size, 8, 8);
                break;
            case 'edge':
                geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
                break;
        }

        this.geometryPool.set(key, geometry);
        return geometry;
    }

    /**
     * Get or create a material from the pool
     * @param {string} type - The type of material
     * @param {object} params - Material parameters
     * @returns {THREE.Material}
     */
    getMaterial(type, params) {
        const key = `${type}-${JSON.stringify(params)}`;
        if (this.materialPool.has(key)) {
            return this.materialPool.get(key);
        }

        let material;
        switch (type) {
            case 'node':
                material = new THREE.MeshPhysicalMaterial({
                    color: params.color,
                    emissive: params.color,
                    emissiveIntensity: params.emissiveIntensity,
                    metalness: this.materialSettings.metalness,
                    roughness: this.materialSettings.roughness,
                    transparent: true,
                    opacity: this.materialSettings.opacity,
                    envMapIntensity: 1.0,
                    clearcoat: this.materialSettings.clearcoat,
                    clearcoatRoughness: this.materialSettings.clearcoatRoughness
                });
                break;
            case 'edge':
                material = new THREE.LineBasicMaterial({
                    color: params.color,
                    opacity: params.opacity,
                    transparent: true,
                    linewidth: params.linewidth || 1
                });
                break;
        }

        this.materialPool.set(key, material);
        return material;
    }

    // Previous methods remain the same until updateNodes...

    /**
     * Updates nodes in the scene based on the graph data.
     */
    updateNodes() {
        const newNodeIds = new Set(this.nodes.map((node) => node.id));

        // Remove nodes that no longer exist
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!newNodeIds.has(nodeId)) {
                this.lod.removeLevel(mesh);
                this.nodeMeshes.delete(nodeId);
                
                // Return mesh to pool
                if (mesh.material) {
                    mesh.material.dispose();
                }
                this.nodeMeshPool.push(mesh);
            }
        });

        // Add or update nodes
        this.nodes.forEach((node) => {
            const nodeSize = this.getNodeSize(node);
            const nodeColor = this.getNodeColor(node);
            const distance = node.metadata?.distance || 0;

            if (this.nodeMeshes.has(node.id)) {
                const mesh = this.nodeMeshes.get(node.id);
                mesh.position.set(node.x, node.y, node.z);
                
                // Update material if needed
                const material = this.getMaterial('node', {
                    color: nodeColor,
                    emissiveIntensity: this.calculateEmissiveIntensity(node)
                });
                
                if (mesh.material !== material) {
                    if (mesh.material) mesh.material.dispose();
                    mesh.material = material;
                }

                // Update geometry if size changed
                if (mesh.geometry.parameters.radius !== nodeSize) {
                    const geometry = this.getGeometry('node-high', nodeSize);
                    mesh.geometry = geometry;
                }
            } else {
                // Create LOD levels
                const highDetail = new THREE.Mesh(
                    this.getGeometry('node-high', nodeSize),
                    this.getMaterial('node', {
                        color: nodeColor,
                        emissiveIntensity: this.calculateEmissiveIntensity(node)
                    })
                );
                
                const mediumDetail = new THREE.Mesh(
                    this.getGeometry('node-medium', nodeSize),
                    highDetail.material
                );
                
                const lowDetail = new THREE.Mesh(
                    this.getGeometry('node-low', nodeSize),
                    highDetail.material
                );

                // Create LOD object
                const nodeLOD = new THREE.LOD();
                nodeLOD.addLevel(highDetail, 0);
                nodeLOD.addLevel(mediumDetail, 10);
                nodeLOD.addLevel(lowDetail, 20);
                nodeLOD.position.set(node.x, node.y, node.z);
                
                this.lod.addLevel(nodeLOD, distance);
                this.nodeMeshes.set(node.id, nodeLOD);
            }
        });
    }

    /**
     * Updates edges in the scene based on the graph data.
     */
    updateLinks() {
        const newLinkKeys = new Set(this.links.map((link) => `${link.source}-${link.target}`));

        // Remove edges that no longer exist
        this.linkMeshes.forEach((line, linkKey) => {
            if (!newLinkKeys.has(linkKey)) {
                this.scene.remove(line);
                if (line.material) line.material.dispose();
                this.linkMeshes.delete(linkKey);
                this.linkMeshPool.push(line);
            }
        });

        // Add or update edges
        this.links.forEach((link) => {
            const linkKey = `${link.source}-${link.target}`;
            const weight = link.weight || 1;
            const normalizedWeight = Math.min(weight / 10, 1);

            const sourceMesh = this.nodeMeshes.get(link.source);
            const targetMesh = this.nodeMeshes.get(link.target);
            
            if (!sourceMesh || !targetMesh) return;

            if (this.linkMeshes.has(linkKey)) {
                const line = this.linkMeshes.get(linkKey);
                const positions = line.geometry.attributes.position.array;
                positions[0] = sourceMesh.position.x;
                positions[1] = sourceMesh.position.y;
                positions[2] = sourceMesh.position.z;
                positions[3] = targetMesh.position.x;
                positions[4] = targetMesh.position.y;
                positions[5] = targetMesh.position.z;
                line.geometry.attributes.position.needsUpdate = true;
                
                // Update material if needed
                const material = this.getMaterial('edge', {
                    color: this.edgeColor,
                    opacity: this.edgeOpacity * normalizedWeight,
                    linewidth: Math.max(1, Math.min(weight, 5))
                });
                
                if (line.material !== material) {
                    if (line.material) line.material.dispose();
                    line.material = material;
                }
            } else {
                // Create new edge
                const geometry = this.getGeometry('edge');
                const material = this.getMaterial('edge', {
                    color: this.edgeColor,
                    opacity: this.edgeOpacity * normalizedWeight,
                    linewidth: Math.max(1, Math.min(weight, 5))
                });

                let line;
                if (this.linkMeshPool.length > 0) {
                    line = this.linkMeshPool.pop();
                    line.geometry = geometry;
                    line.material = material;
                } else {
                    line = new THREE.Line(geometry, material);
                }

                const positions = line.geometry.attributes.position.array;
                positions[0] = sourceMesh.position.x;
                positions[1] = sourceMesh.position.y;
                positions[2] = sourceMesh.position.z;
                positions[3] = targetMesh.position.x;
                positions[4] = targetMesh.position.y;
                positions[5] = targetMesh.position.z;
                line.geometry.attributes.position.needsUpdate = true;

                this.scene.add(line);
                this.linkMeshes.set(linkKey, line);
            }
        });
    }

    /**
     * Calculate emissive intensity based on node age
     * @param {object} node - The node object
     * @returns {number} - The emissive intensity
     */
    calculateEmissiveIntensity(node) {
        const lastModified = node.metadata?.github_last_modified || 
                           node.metadata?.last_modified || 
                           new Date().toISOString();
        const now = Date.now();
        const ageInDays = (now - new Date(lastModified).getTime()) / (24 * 60 * 60 * 1000);
        
        // Normalize age to 0-1 range and invert (newer = brighter)
        const normalizedAge = Math.min(ageInDays / 30, 1);
        return this.materialSettings.emissiveMaxIntensity - 
            (normalizedAge * (this.materialSettings.emissiveMaxIntensity - this.materialSettings.emissiveMinIntensity));
    }

    /**
     * Dispose of all resources
     */
    dispose() {
        // Dispose of node resources
        this.nodeMeshes.forEach(mesh => {
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            }
        });

        // Dispose of edge resources
        this.linkMeshes.forEach(line => {
            if (line.geometry) line.geometry.dispose();
            if (line.material) line.material.dispose();
        });

        // Dispose of pooled resources
        this.nodeMeshPool.forEach(mesh => {
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            }
        });

        this.linkMeshPool.forEach(line => {
            if (line.geometry) line.geometry.dispose();
            if (line.material) line.material.dispose();
        });

        // Dispose of shared resources
        this.geometryPool.forEach(geometry => geometry.dispose());
        this.materialPool.forEach(material => material.dispose());

        // Clear all collections
        this.nodeMeshes.clear();
        this.linkMeshes.clear();
        this.nodeMeshPool.length = 0;
        this.linkMeshPool.length = 0;
        this.geometryPool.clear();
        this.materialPool.clear();

        // Remove LOD from scene
        if (this.lod.parent) {
            this.lod.parent.remove(this.lod);
        }
    }
}
