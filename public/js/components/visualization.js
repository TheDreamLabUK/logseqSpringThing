import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';

export class Visualization {
  constructor() {
    this.initThreeJS();
    this.addLights();
    this.addBackground();
    this.initPostProcessing();
  }

  initThreeJS() {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x000000, 0.02);

    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 0, 50);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(
      this.camera,
      this.renderer.domElement
    );
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    window.addEventListener(
      'resize',
      this.onWindowResize.bind(this),
      false
    );
  }

  addLights() {
    // Ambient Light
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
    this.scene.add(ambientLight);

    // Point Light
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(50, 50, 50);
    this.scene.add(pointLight);

    // Directional Light
    const directionalLight = new THREE.DirectionalLight(
      0xffffff,
      0.5
    );
    directionalLight.position.set(-50, -50, -50);
    this.scene.add(directionalLight);
  }

  addBackground() {
    const loader = new THREE.CubeTextureLoader();
    const texture = loader.load([
      'path/to/px.jpg',
      'path/to/nx.jpg',
      'path/to/py.jpg',
      'path/to/ny.jpg',
      'path/to/pz.jpg',
      'path/to/nz.jpg',
    ]);
    this.scene.background = texture;
  }

  initPostProcessing() {
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      1.5,
      0.4,
      0.85
    );
    bloomPass.threshold = 0;
    bloomPass.strength = 1.5;
    bloomPass.radius = 0;
    this.composer.addPass(bloomPass);
  }

  createNodeObjects(nodes) {
    const geometry = new THREE.SphereGeometry(1, 32, 32);

    const material = new THREE.MeshStandardMaterial({
      color: 0xff0000,
      metalness: 0.7,
      roughness: 0.2,
      emissive: 0x111111,
      envMap: this.scene.background,
    });

    this.nodesGroup = new THREE.Group();

    nodes.forEach((node) => {
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(node.x || 0, node.y || 0, node.z || 0);
      this.nodesGroup.add(sphere);
    });

    this.scene.add(this.nodesGroup);
  }

  createEdgeObjects(edges) {
    const material = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.5,
    });

    const edgesGroup = new THREE.Group();

    edges.forEach((edge) => {
      const geometry = new THREE.BufferGeometry();
      const vertices = new Float32Array([
        edge.source.x || 0,
        edge.source.y || 0,
        edge.source.z || 0,
        edge.target.x || 0,
        edge.target.y || 0,
        edge.target.z || 0,
      ]);
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(vertices, 3)
      );

      const line = new THREE.Line(geometry, material);
      edgesGroup.add(line);
    });

    this.scene.add(edgesGroup);
  }

  onWindowResize() {
    this.camera.aspect =
      window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(
      window.innerWidth,
      window.innerHeight
    );
    this.composer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.controls.update();

    // Rotate the nodes group for some animation
    if (this.nodesGroup) {
      this.nodesGroup.rotation.y += 0.001;
    }

    // Use composer for post-processing
    this.composer.render();
  }
}
