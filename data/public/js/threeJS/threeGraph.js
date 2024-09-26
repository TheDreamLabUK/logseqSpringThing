// public/js/threeJS/threeGraph.js

import * as THREE from 'three';

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
            if (this.nodeMeshes.has(node.id)) {
                const mesh = this.nodeMeshes.get(node.id);
                mesh.position.set(node.x, node.y, node.z);
                // Optionally update node properties like color or size
            } else {
                // Get mesh from pool or create new one
                let mesh;
                if (this.nodeMeshPool.length > 0) {
                    mesh = this.nodeMeshPool.pop();
                } else {
                    // Create a new node mesh
                    const geometry = new THREE.SphereGeometry(2, 16, 16);
                    const material = new THREE.MeshStandardMaterial({ color: this.getNodeColor(node) });
                    mesh = new THREE.Mesh(geometry, material);
                }

                mesh.position.set(node.x, node.y, node.z);
                mesh.userData = { id: node.id, name: node.label };
                this.lod.addLevel(mesh, 0); // Add to LOD

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
                this.linkMeshPool.push(line); // Return to pool
            }
        });

        // Add or update edges
        this.links.forEach((link) => {
            const linkKey = `${link.source}-${link.target}`;
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
                }
            } else {
                // Get line from pool or create new one
                let line;
                if (this.linkMeshPool.length > 0) {
                    line = this.linkMeshPool.pop();
                } else {
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array(6); // 2 points * 3 coordinates
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    const material = new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.5, transparent: true });
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

    /**
     * Determines the color of a node based on its properties.
     * @param {object} node - The node object.
     * @returns {THREE.Color} - The color of the node.
     */
    getNodeColor(node) {
        // Example: Color nodes based on a 'type' property
        if (node.type === 'core') {
            return new THREE.Color(0xffa500); // Orange for core nodes
        } else if (node.type === 'secondary') {
            return new THREE.Color(0x00ffff); // Cyan for secondary nodes
        } else {
            return new THREE.Color(0x00ff00); // Green for default nodes
        }
    }
}
