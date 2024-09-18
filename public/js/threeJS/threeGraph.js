import * as THREE from 'three';

export class ForceGraph {
  constructor(scene) {
    this.scene = scene;
    this.nodes = [];
    this.links = [];
    this.nodeObjects = new Map();
    this.linkObjects = new Map();
  }

  updateGraph(data) {
    this.nodes = data.nodes;
    this.links = data.links;
    this.renderGraph();
  }

  renderGraph() {
    this.updateNodes();
    this.updateLinks();
  }

  updateNodes() {
    const newNodeIds = new Set(this.nodes.map(n => n.id));

    // Remove nodes that no longer exist
    for (const [id, object] of this.nodeObjects) {
      if (!newNodeIds.has(id)) {
        this.scene.remove(object);
        this.nodeObjects.delete(id);
      }
    }

    // Update existing nodes and add new ones
    this.nodes.forEach(node => {
      if (this.nodeObjects.has(node.id)) {
        const nodeObject = this.nodeObjects.get(node.id);
        nodeObject.position.set(node.x, node.y, node.z);
      } else {
        const geometry = new THREE.SphereGeometry(5);
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(node.x, node.y, node.z);
        this.scene.add(sphere);
        this.nodeObjects.set(node.id, sphere);
      }
    });
  }

  updateLinks() {
    const newLinkIds = new Set(this.links.map(l => `${l.source}-${l.target}`));

    // Remove links that no longer exist
    for (const [id, object] of this.linkObjects) {
      if (!newLinkIds.has(id)) {
        this.scene.remove(object);
        this.linkObjects.delete(id);
      }
    }

    // Update existing links and add new ones
    this.links.forEach(link => {
      const linkId = `${link.source}-${link.target}`;
      const sourceNode = this.nodes.find(n => n.id === link.source);
      const targetNode = this.nodes.find(n => n.id === link.target);

      if (this.linkObjects.has(linkId)) {
        const linkObject = this.linkObjects.get(linkId);
        linkObject.geometry.setFromPoints([
          new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z),
          new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z)
        ]);
      } else {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z),
          new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z)
        ]);
        const material = new THREE.LineBasicMaterial({ color: 0xffffff });
        const line = new THREE.Line(geometry, material);
        this.scene.add(line);
        this.linkObjects.set(linkId, line);
      }
    });
  }
}
