// public/js/threeJS/threeGraph.js

import * as THREE from 'three';
import { visualizationSettings } from '../services/visualizationSettings.js';

/**
 * Enhanced ForceGraph class with instanced mesh rendering
 */
export class ForceGraph {
    constructor(scene) {
        this.scene = scene;

        // Data structures
        this.nodes = [];
        this.links = [];
        this.nodeInstances = new Map();
        this.linkInstances = new Map();

        // Instanced meshes
        this.nodeInstancedMesh = null;
        this.linkInstancedMesh = null;

        // Temporary objects for matrix calculations
        this.tempMatrix = new THREE.Matrix4();
        this.tempColor = new THREE.Color();
        this.tempVector = new THREE.Vector3();
        this.tempQuaternion = new THREE.Quaternion();
        this.tempScale = new THREE.Vector3();

        // Level of Detail
        this.lod = new THREE.LOD();
        this.scene.add(this.lod);

        // Settings
        const nodeSettings = visualizationSettings.getNodeSettings();
        const edgeSettings = visualizationSettings.getEdgeSettings();
        
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
        this.minNodeSize = nodeSettings.minNodeSize;
        this.maxNodeSize = nodeSettings.maxNodeSize;
        this.materialSettings = nodeSettings.material;

        // Initialize instanced meshes
        this.initInstancedMeshes();
    }

    /**
     * Initialize instanced meshes for nodes and links
     */
    initInstancedMeshes() {
        // Create node geometry with different LOD levels
        const highDetailGeometry = new THREE.SphereGeometry(1, 32, 32);
        const mediumDetailGeometry = new THREE.SphereGeometry(1, 16, 16);
        const lowDetailGeometry = new THREE.SphereGeometry(1, 8, 8);

        // Create node material
        const nodeMaterial = new THREE.MeshPhysicalMaterial({
            metalness: this.materialSettings.metalness,
            roughness: this.materialSettings.roughness,
            transparent: true,
            opacity: this.materialSettings.opacity,
            envMapIntensity: 1.0,
            clearcoat: this.materialSettings.clearcoat,
            clearcoatRoughness: this.materialSettings.clearcoatRoughness
        });

        // Create instanced meshes for each LOD level
        const maxInstances = 10000; // Adjust based on expected graph size
        this.nodeInstancedMeshes = {
            high: new THREE.InstancedMesh(highDetailGeometry, nodeMaterial.clone(), maxInstances),
            medium: new THREE.InstancedMesh(mediumDetailGeometry, nodeMaterial.clone(), maxInstances),
            low: new THREE.InstancedMesh(lowDetailGeometry, nodeMaterial.clone(), maxInstances)
        };

        // Add LOD levels
        this.lod.addLevel(this.nodeInstancedMeshes.high, 0);
        this.lod.addLevel(this.nodeInstancedMeshes.medium, 10);
        this.lod.addLevel(this.nodeInstancedMeshes.low, 20);

        // Create link geometry
        const linkGeometry = new THREE.CylinderGeometry(0.01, 0.01, 1, 8, 1);
        linkGeometry.rotateX(Math.PI / 2); // Align with Z-axis

        // Create link material
        const linkMaterial = new THREE.MeshBasicMaterial({
            color: this.edgeColor,
            transparent: true,
            opacity: this.edgeOpacity,
            depthWrite: false
        });

        // Create instanced mesh for links
        this.linkInstancedMesh = new THREE.InstancedMesh(
            linkGeometry,
            linkMaterial,
            maxInstances * 2 // Links typically more numerous than nodes
        );

        this.scene.add(this.linkInstancedMesh);

        // Initialize instance counts
        this.nodeInstanceCount = 0;
        this.linkInstanceCount = 0;
    }

    /**
     * Calculate node size based on metadata
     * @param {object} node - Node object
     * @returns {number} Node size
     */
    getNodeSize(node) {
        const baseSize = (node.metadata?.size || 1) * this.minNodeSize;
        const weight = node.metadata?.weight || 1;
        return Math.min(baseSize * Math.sqrt(weight), this.maxNodeSize);
    }

    /**
     * Get node color based on metadata
     * @param {object} node - Node object
     * @returns {THREE.Color} Node color
     */
    getNodeColor(node) {
        const type = node.metadata?.type || 'DEFAULT';
        return this.nodeColors[type] || this.nodeColors.DEFAULT;
    }

    /**
     * Calculate emissive intensity based on node age
     * @param {object} node - Node object
     * @returns {number} Emissive intensity
     */
    calculateEmissiveIntensity(node) {
        const lastModified = node.metadata?.github_last_modified || 
                           node.metadata?.last_modified || 
                           new Date().toISOString();
        const now = Date.now();
        const ageInDays = (now - new Date(lastModified).getTime()) / (24 * 60 * 60 * 1000);
        
        const normalizedAge = Math.min(ageInDays / 30, 1);
        return this.materialSettings.emissiveMaxIntensity - 
            (normalizedAge * (this.materialSettings.emissiveMaxIntensity - this.materialSettings.emissiveMinIntensity));
    }

    /**
     * Update node instances
     */
    updateNodes() {
        // Reset instance count
        this.nodeInstanceCount = 0;

        // Update node instances
        this.nodes.forEach((node, index) => {
            const size = this.getNodeSize(node);
            const color = this.getNodeColor(node);
            const emissiveIntensity = this.calculateEmissiveIntensity(node);

            // Set transform matrix
            this.tempMatrix.compose(
                new THREE.Vector3(node.x, node.y, node.z),
                this.tempQuaternion,
                new THREE.Vector3(size, size, size)
            );

            // Update instances for each LOD level
            Object.values(this.nodeInstancedMeshes).forEach(instancedMesh => {
                instancedMesh.setMatrixAt(index, this.tempMatrix);
                instancedMesh.setColorAt(index, color);
                instancedMesh.material.emissiveIntensity = emissiveIntensity;
            });

            this.nodeInstances.set(node.id, index);
            this.nodeInstanceCount = Math.max(this.nodeInstanceCount, index + 1);
        });

        // Update instance meshes
        Object.values(this.nodeInstancedMeshes).forEach(instancedMesh => {
            instancedMesh.count = this.nodeInstanceCount;
            instancedMesh.instanceMatrix.needsUpdate = true;
            if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
        });
    }

    /**
     * Update link instances
     */
    updateLinks() {
        // Reset instance count
        this.linkInstanceCount = 0;

        // Update link instances
        this.links.forEach((link, index) => {
            const sourceIndex = this.nodeInstances.get(link.source);
            const targetIndex = this.nodeInstances.get(link.target);

            if (sourceIndex === undefined || targetIndex === undefined) return;

            const sourcePos = new THREE.Vector3(
                this.nodes[sourceIndex].x,
                this.nodes[sourceIndex].y,
                this.nodes[sourceIndex].z
            );
            const targetPos = new THREE.Vector3(
                this.nodes[targetIndex].x,
                this.nodes[targetIndex].y,
                this.nodes[targetIndex].z
            );

            // Calculate link transform
            const distance = sourcePos.distanceTo(targetPos);
            this.tempVector.subVectors(targetPos, sourcePos);
            this.tempQuaternion.setFromUnitVectors(
                new THREE.Vector3(0, 0, 1),
                this.tempVector.normalize()
            );

            this.tempMatrix.compose(
                sourcePos.lerp(targetPos, 0.5), // Position at midpoint
                this.tempQuaternion,
                new THREE.Vector3(1, 1, distance)
            );

            // Update link instance
            this.linkInstancedMesh.setMatrixAt(index, this.tempMatrix);
            
            const weight = link.weight || 1;
            const normalizedWeight = Math.min(weight / 10, 1);
            this.tempColor.copy(this.edgeColor).multiplyScalar(normalizedWeight);
            this.linkInstancedMesh.setColorAt(index, this.tempColor);

            this.linkInstances.set(`${link.source}-${link.target}`, index);
            this.linkInstanceCount = Math.max(this.linkInstanceCount, index + 1);
        });

        // Update link instance mesh
        this.linkInstancedMesh.count = this.linkInstanceCount;
        this.linkInstancedMesh.instanceMatrix.needsUpdate = true;
        if (this.linkInstancedMesh.instanceColor) this.linkInstancedMesh.instanceColor.needsUpdate = true;
    }

    /**
     * Update graph visualization
     */
    update() {
        this.updateNodes();
        this.updateLinks();
    }

    /**
     * Clean up resources
     */
    dispose() {
        // Dispose of node resources
        Object.values(this.nodeInstancedMeshes).forEach(instancedMesh => {
            instancedMesh.geometry.dispose();
            instancedMesh.material.dispose();
        });

        // Dispose of link resources
        if (this.linkInstancedMesh) {
            this.linkInstancedMesh.geometry.dispose();
            this.linkInstancedMesh.material.dispose();
        }

        // Remove from scene
        this.scene.remove(this.lod);
        this.scene.remove(this.linkInstancedMesh);

        // Clear collections
        this.nodeInstances.clear();
        this.linkInstances.clear();
        this.nodes = [];
        this.links = [];
    }
}
