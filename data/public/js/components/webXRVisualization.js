import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.module.js';
import { VRButton } from 'https://cdn.jsdelivr.net/npm/three@0.132.2/examples/jsm/webxr/VRButton.js';

export class WebXRVisualization {
    constructor(graphDataManager) {
        this.graphDataManager = graphDataManager;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.nodes = new Map();
        this.edges = new Map();
    }

    initialize() {
        this.initScene();
        this.initCamera();
        this.initRenderer();
        this.initVR();
        this.initLights();
        this.animate();
    }

    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
    }

    initCamera() {
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.z = 5;
    }

    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.xr.enabled = true;
        document.getElementById('scene-container').appendChild(this.renderer.domElement);
    }

    initVR() {
        document.body.appendChild(VRButton.createButton(this.renderer));
    }

    initLights() {
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
    }

    updateVisualization() {
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData) return;

        this.updateNodes(graphData.nodes);
        this.updateEdges(graphData.edges);
    }

    updateNodes(nodes) {
        nodes.forEach(node => {
            if (!this.nodes.has(node.id)) {
                const geometry = new THREE.SphereGeometry(0.1, 32, 32);
                const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
                const mesh = new THREE.Mesh(geometry, material);
                this.scene.add(mesh);
                this.nodes.set(node.id, mesh);
            }
            const nodeMesh = this.nodes.get(node.id);
            nodeMesh.position.set(node.x, node.y, node.z);
        });
    }

    updateEdges(edges) {
        edges.forEach(edge => {
            const edgeId = `${edge.source}-${edge.target}`;
            if (!this.edges.has(edgeId)) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(edge.source.x, edge.source.y, edge.source.z),
                    new THREE.Vector3(edge.target.x, edge.target.y, edge.target.z)
                ]);
                const material = new THREE.LineBasicMaterial({ color: 0xffffff });
                const line = new THREE.Line(geometry, material);
                this.scene.add(line);
                this.edges.set(edgeId, line);
            } else {
                const line = this.edges.get(edgeId);
                const positions = line.geometry.attributes.position.array;
                positions[0] = edge.source.x;
                positions[1] = edge.source.y;
                positions[2] = edge.source.z;
                positions[3] = edge.target.x;
                positions[4] = edge.target.y;
                positions[5] = edge.target.z;
                line.geometry.attributes.position.needsUpdate = true;
            }
        });
    }

    animate() {
        this.renderer.setAnimationLoop(() => {
            this.renderer.render(this.scene, this.camera);
        });
    }
}
