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

        // Instanced meshes for performance
        this.nodeInstancedMesh = null;
        this.nodeCount = 0;

        // Object pools
        this.nodeMeshPool = [];
        this.linkMeshPool = [];

        // Level of Detail
        this.lod = new THREE.LOD();
        this.scene.add(this.lod);

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
    }

    /**
     * Calculates node size in meters based on metadata.
     * @param {object} node - The node object with metadata.
     * @returns {number} - The node size in meters.
     */
    getNodeSize(node) {
        if (node.metadata?.node_size) {
            const size = parseFloat(node.metadata.node_size);
            // Normalize size between minNodeSize (0.1m) and maxNodeSize (0.3m)
            return this.minNodeSize + (size * (this.maxNodeSize - this.minNodeSize));
        }
        return this.minNodeSize; // Default to minimum size (10cm)
    }

    /**
     * Calculates node color based on age and type.
     * @param {object} node - The node object with metadata.
     * @returns {THREE.Color} - The color of the node.
     */
    getNodeColor(node) {
        // First check node type
        if (node.type === 'core') return this.nodeColors.CORE;
        if (node.type === 'secondary') return this.nodeColors.SECONDARY;

        // Then check age if type is not special
        const lastModified = node.metadata?.github_last_modified || 
                           node.metadata?.last_modified || 
                           new Date().toISOString();
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const dayInMs = 24 * 60 * 60 * 1000;
        
        if (age < 3 * dayInMs) return this.nodeColors.NEW;        // Less than 3 days old
        if (age < 7 * dayInMs) return this.nodeColors.RECENT;     // Less than 7 days old
        if (age < 30 * dayInMs) return this.nodeColors.MEDIUM;    // Less than 30 days old
        return this.nodeColors.OLD;                               // 30 days or older
    }

    /**
     * Creates a material for a node.
     * @param {THREE.Color} color - The base color for the node.
     * @param {object} node - The node object with metadata.
     * @returns {THREE.MeshPhysicalMaterial} - The material for the node.
     */
    createNodeMaterial(color, node) {
        const lastModified = node.metadata?.github_last_modified || 
                           node.metadata?.last_modified || 
                           new Date().toISOString();
        const now = Date.now();
        const ageInDays = (now - new Date(lastModified).getTime()) / (24 * 60 * 60 * 1000);
        
        // Normalize age to 0-1 range and invert (newer = brighter)
        const normalizedAge = Math.min(ageInDays / 30, 1);
        const emissiveIntensity = this.materialSettings.emissiveMaxIntensity - 
            (normalizedAge * (this.materialSettings.emissiveMaxIntensity - this.materialSettings.emissiveMinIntensity));

        return new THREE.MeshPhysicalMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: emissiveIntensity,
            metalness: this.materialSettings.metalness,
            roughness: this.materialSettings.roughness,
            transparent: true,
            opacity: this.materialSettings.opacity,
            envMapIntensity: 1.0,
            clearcoat: this.materialSettings.clearcoat,
            clearcoatRoughness: this.materialSettings.clearcoatRoughness
        });
    }

    /**
     * Updates the graph with new data.
     * @param {object} graphData - The graph data containing nodes and edges.
     */
    updateGraph(graphData) {
        this.nodes = graphData.nodes;
        this.links = graphData.edges;
        this.renderGraph();
    }

    /**
     * Renders the graph by creating and updating nodes and edges.
     */
    renderGraph() {
        this.updateNodes();
        this.updateLinks();
    }

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
                this.nodeMeshPool.push(mesh); // Return to pool
            }
        });

        // Add or update nodes
        this.nodes.forEach((node) => {
            const nodeSize = this.getNodeSize(node);
            const nodeColor = this.getNodeColor(node);

            if (this.nodeMeshes.has(node.id)) {
                const mesh = this.nodeMeshes.get(node.id);
                mesh.position.set(node.x, node.y, node.z);
                
                // Update geometry and material if needed
                if (mesh.geometry.parameters.radius !== nodeSize) {
                    mesh.geometry.dispose();
                    mesh.geometry = new THREE.SphereGeometry(nodeSize, 32, 32);
                }
                mesh.material.dispose();
                mesh.material = this.createNodeMaterial(nodeColor, node);
            } else {
                // Get mesh from pool or create new one
                let mesh;
                if (this.nodeMeshPool.length > 0) {
                    mesh = this.nodeMeshPool.pop();
                    mesh.geometry.dispose();
                    mesh.material.dispose();
                    mesh.geometry = new THREE.SphereGeometry(nodeSize, 32, 32);
                    mesh.material = this.createNodeMaterial(nodeColor, node);
                } else {
                    const geometry = new THREE.SphereGeometry(nodeSize, 32, 32);
                    const material = this.createNodeMaterial(nodeColor, node);
                    mesh = new THREE.Mesh(geometry, material);
                }

                mesh.position.set(node.x, node.y, node.z);
                mesh.userData = { id: node.id, name: node.label };
                this.lod.addLevel(mesh, 0);

                this.nodeMeshes.set(node.id, mesh);
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
                this.linkMeshes.delete(linkKey);
                this.linkMeshPool.push(line);
            }
        });

        // Add or update edges
        this.links.forEach((link) => {
            const linkKey = `${link.source}-${link.target}`;
            const weight = link.weight || 1;
            const normalizedWeight = Math.min(weight / 10, 1);

            if (this.linkMeshes.has(linkKey)) {
                const line = this.linkMeshes.get(linkKey);
                const sourceMesh = this.nodeMeshes.get(link.source);
                const targetMesh = this.nodeMeshes.get(link.target);
                if (sourceMesh && targetMesh) {
                    const positions = line.geometry.attributes.position.array;
                    positions[0] = sourceMesh.position.x;
                    positions[1] = sourceMesh.position.y;
                    positions[2] = sourceMesh.position.z;
                    positions[3] = targetMesh.position.x;
                    positions[4] = targetMesh.position.y;
                    positions[5] = targetMesh.position.z;
                    line.geometry.attributes.position.needsUpdate = true;
                    
                    // Update edge appearance
                    line.material.opacity = this.edgeOpacity * normalizedWeight;
                }
            } else {
                // Get line from pool or create new one
                let line;
                if (this.linkMeshPool.length > 0) {
                    line = this.linkMeshPool.pop();
                } else {
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array(6);
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    const material = new THREE.LineBasicMaterial({ 
                        color: this.edgeColor, 
                        opacity: this.edgeOpacity * normalizedWeight, 
                        transparent: true,
                        linewidth: Math.max(1, Math.min(weight, 5))
                    });
                    line = new THREE.Line(geometry, material);
                }

                const sourceMesh = this.nodeMeshes.get(link.source);
                const targetMesh = this.nodeMeshes.get(link.target);
                if (sourceMesh && targetMesh) {
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
            }
        });
    }
}
