import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export class Visualization {
  constructor() {
    this.initThreeJS();
  }

  initThreeJS() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.z = 100;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    window.addEventListener('resize', this.onWindowResize.bind(this), false);
  }

  createNodeObjects(nodes) {
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const nodesGroup = new THREE.Group();

    nodes.forEach(node => {
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(node.x, node.y, node.z);
      nodesGroup.add(sphere);
    });

    this.scene.add(nodesGroup);
  }

  createEdgeObjects(edges) {
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });
    const edgesGroup = new THREE.Group();

    edges.forEach(edge => {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(edge.source.x, edge.source.y, edge.source.z),
        new THREE.Vector3(edge.target.x, edge.target.y, edge.target.z),
      ]);
      const line = new THREE.Line(geometry, material);
      edgesGroup.add(line);
    });

    this.scene.add(edgesGroup);
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
