import * as THREE from 'three';
import { ForceGraph3D } from '3d-force-graph';

export class ForceGraph {
  constructor(scene) {
    this.scene = scene;
    this.nodes = [];
    this.links = [];
    this.graph = ForceGraph3D()(document.createElement('div'))
      .graphData({ nodes: this.nodes, links: this.links })
      .onEngineTick(() => this.renderGraph());
  }

  addNode(node) {
    this.nodes.push(node);
    this.updateGraphData();
  }

  addLink(link) {
    this.links.push(link);
    this.updateGraphData();
  }

  updateNodePosition(node, position) {
    Object.assign(node, position);
  }

  updateGraphData() {
    this.graph.graphData({ nodes: this.nodes, links: this.links });
  }

  renderGraph() {
    // This method is called on each engine tick
    // Update Three.js objects based on force-graph positions
    this.nodes.forEach(node => {
      if (node.threeObject) {
        node.threeObject.position.set(node.x, node.y, node.z);
      } else {
        const geometry = new THREE.SphereGeometry(5);
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(node.x, node.y, node.z);
        this.scene.add(sphere);
        node.threeObject = sphere;
      }
    });

    this.links.forEach(link => {
      if (link.threeObject) {
        const start = this.nodes.find(n => n.id === link.source);
        const end = this.nodes.find(n => n.id === link.target);
        link.threeObject.geometry.setFromPoints([start, end]);
      } else {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          this.nodes.find(n => n.id === link.source),
          this.nodes.find(n => n.id === link.target)
        ]);
        const material = new THREE.LineBasicMaterial({ color: 0xffffff });
        const line = new THREE.Line(geometry, material);
        this.scene.add(line);
        link.threeObject = line;
      }
    });
  }
}
