import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory';

export class WebXRVisualization {
  constructor(scene, renderer, camera, websocketService) {
    this.scene = scene;
    this.renderer = renderer;
    this.camera = camera;
    this.websocketService = websocketService;
    this.controller1 = null;
    this.controller2 = null;
    this.controllerGrip1 = null;
    this.controllerGrip2 = null;
    this.nodes = new Map();

    this.initWebSocket();
  }

  initWebSocket() {
    this.websocketService.on('nodePositions', this.updateNodePositions.bind(this));
    this.websocketService.on('graphUpdate', this.updateGraph.bind(this));
  }

  initVR() {
    this.renderer.xr.enabled = true;
    document.body.appendChild(VRButton.createButton(this.renderer));
  }

  initVRControllers() {
    const controllerModelFactory = new XRControllerModelFactory();

    this.controller1 = this.renderer.xr.getController(0);
    this.controller1.addEventListener('selectstart', this.onSelectStart.bind(this));
    this.controller1.addEventListener('selectend', this.onSelectEnd.bind(this));
    this.scene.add(this.controller1);

    this.controller2 = this.renderer.xr.getController(1);
    this.controller2.addEventListener('selectstart', this.onSelectStart.bind(this));
    this.controller2.addEventListener('selectend', this.onSelectEnd.bind(this));
    this.scene.add(this.controller2);

    this.controllerGrip1 = this.renderer.xr.getControllerGrip(0);
    this.controllerGrip1.add(controllerModelFactory.createControllerModel(this.controllerGrip1));
    this.scene.add(this.controllerGrip1);

    this.controllerGrip2 = this.renderer.xr.getControllerGrip(1);
    this.controllerGrip2.add(controllerModelFactory.createControllerModel(this.controllerGrip2));
    this.scene.add(this.controllerGrip2);

    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -1)
    ]);

    const line = new THREE.Line(geometry);
    line.name = 'line';
    line.scale.z = 5;

    this.controller1.add(line.clone());
    this.controller2.add(line.clone());
  }

  onSelectStart(event) {
    const controller = event.target;
    const intersections = this.getIntersections(controller);

    if (intersections.length > 0) {
      const intersection = intersections[0];
      this.selectNode(intersection.object);
    }
  }

  onSelectEnd() {
    this.deselectNode();
  }

  getIntersections(controller) {
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.identity().extractRotation(controller.matrixWorld);

    const raycaster = new THREE.Raycaster();
    const rayOrigin = new THREE.Vector3();
    const rayDirection = new THREE.Vector3(0, 0, -1);

    rayOrigin.copy(controller.position);
    rayDirection.applyMatrix4(tempMatrix);

    raycaster.set(rayOrigin, rayDirection);

    return raycaster.intersectObjects(Array.from(this.nodes.values()));
  }

  selectNode(node) {
    // Implement node selection logic here
    console.log('Node selected:', node);
  }

  deselectNode() {
    // Implement node deselection logic here
    console.log('Node deselected');
  }

  updateNodePositions(positions) {
    positions.forEach(({ id, position }) => {
      const node = this.nodes.get(id);
      if (node) {
        node.position.set(position.x, position.y, position.z);
      }
    });
  }

  updateGraph(graphData) {
    // Remove old nodes
    this.nodes.forEach(node => this.scene.remove(node));
    this.nodes.clear();

    // Add new nodes
    graphData.nodes.forEach(nodeData => {
      const geometry = new THREE.SphereGeometry(0.1, 32, 32);
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const nodeMesh = new THREE.Mesh(geometry, material);
      nodeMesh.position.set(nodeData.x, nodeData.y, nodeData.z);
      this.scene.add(nodeMesh);
      this.nodes.set(nodeData.id, nodeMesh);
    });

    // Update edges (if needed)
    // ...
  }

  animate() {
    this.renderer.setAnimationLoop(this.render.bind(this));
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
