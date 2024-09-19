// public/js/components/webXRVisualization.js

import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';

/**
 * WebXRVisualization handles immersive VR/AR interactions within the Three.js scene.
 */
export class WebXRVisualization {
  /**
   * Creates a new WebXRVisualization instance.
   * @param {THREE.Scene} scene - The Three.js scene.
   * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
   * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
   * @param {WebsocketService} websocketService - The WebSocket service instance.
   */
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

    // Initialize WebXR components
    this.initVR();
    this.initControllers();
    this.initWebSocketListeners();
  }

  /**
   * Initializes the VR button and enables WebXR.
   */
  initVR() {
    // Append the VR button to the document for entering VR mode
    document.body.appendChild(VRButton.createButton(this.renderer));
  }

  /**
   * Initializes VR controllers and their models.
   */
  initControllers() {
    // Create a controller model factory for adding controller visuals
    const controllerModelFactory = new XRControllerModelFactory();

    // Initialize Controller 1
    this.controller1 = this.renderer.xr.getController(0);
    this.controller1.addEventListener('selectstart', this.onSelectStart.bind(this));
    this.controller1.addEventListener('selectend', this.onSelectEnd.bind(this));
    this.scene.add(this.controller1);

    // Initialize Controller 2
    this.controller2 = this.renderer.xr.getController(1);
    this.controller2.addEventListener('selectstart', this.onSelectStart.bind(this));
    this.controller2.addEventListener('selectend', this.onSelectEnd.bind(this));
    this.scene.add(this.controller2);

    // Create grips for controllers to visualize their models
    this.controllerGrip1 = this.renderer.xr.getControllerGrip(0);
    this.controllerGrip1.add(controllerModelFactory.createControllerModel(this.controllerGrip1));
    this.scene.add(this.controllerGrip1);

    this.controllerGrip2 = this.renderer.xr.getControllerGrip(1);
    this.controllerGrip2.add(controllerModelFactory.createControllerModel(this.controllerGrip2));
    this.scene.add(this.controllerGrip2);

    // Add laser lines to controllers for interaction visualization
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

  /**
   * Initializes WebSocket listeners for node interactions in VR.
   */
  initWebSocketListeners() {
    // Listen for node position updates via WebSocket
    this.websocketService.on('nodePositions', this.updateNodePositions.bind(this));

    // Listen for graph updates via WebSocket
    this.websocketService.on('graphUpdate', this.updateGraph.bind(this));
  }

  /**
   * Event handler for when a select (e.g., trigger press) starts on a controller.
   * @param {Event} event - The event object.
   */
  onSelectStart(event) {
    const controller = event.target;
    const intersections = this.getIntersections(controller);

    if (intersections.length > 0) {
      const intersection = intersections[0];
      this.selectNode(intersection.object);
    }
  }

  /**
   * Event handler for when a select (e.g., trigger release) ends on a controller.
   */
  onSelectEnd() {
    this.deselectNode();
  }

  /**
   * Raycasts to find intersected objects with the controller's laser.
   * @param {THREE.Object3D} controller - The controller object.
   * @returns {Array} An array of intersected objects.
   */
  getIntersections(controller) {
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.identity().extractRotation(controller.matrixWorld);

    const raycaster = new THREE.Raycaster();
    const rayOrigin = new THREE.Vector3();
    const rayDirection = new THREE.Vector3(0, 0, -1);

    rayOrigin.copy(controller.position);
    rayDirection.applyMatrix4(tempMatrix);

    raycaster.set(rayOrigin, rayDirection);

    return raycaster.intersectObjects(Array.from(this.scene.children).filter(child => child instanceof THREE.Mesh));
  }

  /**
   * Handles the selection of a node.
   * @param {THREE.Mesh} node - The node mesh that was selected.
   */
  selectNode(node) {
    console.log('Node selected:', node.userData.id);
    // Implement node selection logic, such as highlighting or displaying info
    node.material.emissive.set(0x333333); // Example: Change emissive color to indicate selection
  }

  /**
   * Handles the deselection of a node.
   */
  deselectNode() {
    console.log('Node deselected');
    // Implement node deselection logic, such as removing highlights or hiding info
    // Example: Reset emissive color
    // Ensure that you track the selected node to reset its color
  }

  /**
   * Updates node positions based on data received from the server.
   * @param {Array} positions - Array of objects containing node IDs and their new positions.
   */
  updateNodePositions(positions) {
    positions.forEach(({ id, position }) => {
      const node = this.nodes.get(id);
      if (node) {
        node.position.set(position.x, position.y, position.z);
      }
    });
  }

  /**
   * Updates the entire graph visualization based on updated graph data.
   * @param {object} graphData - The updated graph data containing nodes and edges.
   */
  updateGraph(graphData) {
    // Similar to the Visualization class, update nodes and edges here if needed
    // This method can be expanded based on specific requirements
  }

  /**
   * Renders the scene using the Three.js renderer.
   */
  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
