import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';

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
        this.initThreeJS();
        this.initWebXR();
        this.animate();
    }

    initThreeJS() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('scene-container').appendChild(this.renderer.domElement);

        // Add some basic lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        this.scene.add(directionalLight);

        this.camera.position.z = 5;
    }

    initWebXR() {
        // Check if WebXR is supported
        if ('xr' in navigator) {
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                if (supported) {
                    const enterVRButton = document.createElement('button');
                    enterVRButton.textContent = 'Enter VR';
                    enterVRButton.onclick = this.enterVR.bind(this);
                    document.body.appendChild(enterVRButton);
                }
            });
        }
    }

    enterVR() {
        navigator.xr.requestSession('immersive-vr').then((session) => {
            this.renderer.xr.setSession(session);
            this.renderer.xr.enabled = true;
        });
    }

    updateVisualization() {
        if (!this.graphDataManager.graphData) return;

        // Clear existing nodes and edges
        this.nodes.forEach(node => this.scene.remove(node));
        this.edges.forEach(edge => this.scene.remove(edge));
        this.nodes.clear();
        this.edges.clear();

        // Create new nodes
        this.graphDataManager.graphData.nodes.forEach(nodeData => {
            const geometry = new THREE.SphereGeometry(0.1, 32, 32);
            const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
            const nodeMesh = new THREE.Mesh(geometry, material);
            nodeMesh.position.set(
                Math.random() * 10 - 5,
                Math.random() * 10 - 5,
                Math.random() * 10 - 5
            );
            this.scene.add(nodeMesh);
            this.nodes.set(nodeData.id, nodeMesh);
        });

        // Create new edges
        this.graphDataManager.graphData.edges.forEach(edgeData => {
            const sourceNode = this.nodes.get(edgeData.source);
            const targetNode = this.nodes.get(edgeData.target);
            if (sourceNode && targetNode) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    sourceNode.position,
                    targetNode.position
                ]);
                const material = new THREE.LineBasicMaterial({ color: 0xffffff });
                const edgeLine = new THREE.Line(geometry, material);
                this.scene.add(edgeLine);
                this.edges.set(edgeData.id, edgeLine);
            }
        });
    }

    animate() {
        this.renderer.setAnimationLoop(() => {
            this.renderer.render(this.scene, this.camera);
        });
    }
}
