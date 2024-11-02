// Keep all existing imports...
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';

// Constants for Spacemouse sensitivity
const TRANSLATION_SPEED = 0.01;
const ROTATION_SPEED = 0.01;

// Bloom configuration
const BLOOM_LAYER = 1;
const NORMAL_LAYER = 0;
const bloomParams = {
    exposure: 1,
    bloomStrength: 1.5,
    bloomThreshold: 0.2,
    bloomRadius: 0.4
};

// Node appearance configuration
const NODE_COLORS = {
    NEW: new THREE.Color(0x0088ff),      // Blue for new files
    MEDIUM: new THREE.Color(0x00ff88),    // Green for medium-age files
    OLD: new THREE.Color(0xff8800)        // Orange for old files
};

const NODE_SHAPES = {
    FEW_LINKS: 'sphere',      // 0-5 links
    MEDIUM_LINKS: 'box',      // 6-15 links
    MANY_LINKS: 'octahedron'  // 16+ links
};

/**
 * Class representing a WebXR visualization environment.
 */
export class WebXRVisualization {
    constructor(graphDataManager) {
        console.log('WebXRVisualization constructor called');
        this.graphDataManager = graphDataManager;

        // Initialize the scene, camera, and renderer
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.camera.position.set(0, 0, 500);
        this.camera.layers.enable(BLOOM_LAYER);
        this.camera.layers.enable(NORMAL_LAYER);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.toneMappingExposure = bloomParams.exposure;
        this.renderer.setClearColor(0x000000);

        this.controls = null;
        this.bloomComposer = null;
        this.finalComposer = null;

        // Maps to hold node and edge meshes and their labels
        this.nodeMeshes = new Map();
        this.edgeMeshes = new Map();
        this.nodeLabels = new Map();

        this.hologramGroup = new THREE.Group();
        this.animationFrameId = null;

        this.selectedNode = null;

        // Force-directed layout parameters
        this.forceDirectedIterations = 100;
        this.forceDirectedRepulsion = 1.0;
        this.forceDirectedAttraction = 0.01;

        // Lighting references
        this.ambientLight = null;
        this.directionalLight = null;

        // Initialize settings
        this.initializeSettings();

        // Add event listener for graph data updates
        window.addEventListener('graphDataUpdated', (event) => {
            console.log('Received graphDataUpdated event:', event.detail);
            this.updateVisualization();
        });

        console.log('WebXRVisualization constructor completed');
    }

    initializeSettings() {
        console.log('Initializing settings');
        this.nodeColor = 0x1A0B31;
        this.edgeColor = 0x4444ff;
        this.hologramColor = 0xFFD700;
        this.nodeSizeScalingFactor = 1;
        this.hologramScale = 1;
        this.hologramOpacity = 0.1;
        this.edgeOpacity = 0.6;
        this.labelFontSize = 48;
        this.fogDensity = 0.002;
        this.minNodeSize = 1;
        this.maxNodeSize = 5;
        this.bloomStrength = bloomParams.bloomStrength;
        this.bloomRadius = bloomParams.bloomRadius;
        this.bloomThreshold = bloomParams.bloomThreshold;
        
        // Lighting settings
        this.ambientLightIntensity = 1.5;
        this.directionalLightIntensity = 1.0;
        this.directionalLightColor = 0xffffff;
        this.ambientLightColor = 0x404040;
        
        console.log('Settings initialized');
    }

    updateVisualFeatures(control, value) {
        console.log(`Updating visual feature: ${control} = ${value}`);
        let needsUpdate = false;
        let layoutChanged = false;

        switch (control) {
            // Lighting controls
            case 'ambientLightIntensity':
                this.ambientLightIntensity = value;
                if (this.ambientLight) {
                    this.ambientLight.intensity = value;
                }
                needsUpdate = true;
                break;

            case 'directionalLightIntensity':
                this.directionalLightIntensity = value;
                if (this.directionalLight) {
                    this.directionalLight.intensity = value;
                }
                needsUpdate = true;
                break;

            case 'ambientLightColor':
                this.ambientLightColor = value;
                if (this.ambientLight) {
                    this.ambientLight.color.setHex(value);
                }
                needsUpdate = true;
                break;

            case 'directionalLightColor':
                this.directionalLightColor = value;
                if (this.directionalLight) {
                    this.directionalLight.color.setHex(value);
                }
                needsUpdate = true;
                break;

            // Node appearance
            case 'nodeColor':
                this.nodeColor = value;
                this.nodeMeshes.forEach(mesh => {
                    mesh.material.color.setHex(value);
                    mesh.material.emissive.setHex(value);
                });
                needsUpdate = true;
                break;

            // Edge appearance
            case 'edgeColor':
                this.edgeColor = value;
                this.edgeMeshes.forEach(line => {
                    line.material.color.setHex(value);
                });
                needsUpdate = true;
                break;

            case 'edgeOpacity':
                this.edgeOpacity = value;
                this.edgeMeshes.forEach(line => {
                    line.material.opacity = value;
                });
                needsUpdate = true;
                break;

            // Hologram appearance
            case 'hologramColor':
                this.hologramColor = value;
                this.hologramGroup.children.forEach(child => {
                    child.material.color.setHex(value);
                    child.material.emissive.setHex(value);
                });
                needsUpdate = true;
                break;

            case 'hologramScale':
                this.hologramScale = value;
                this.hologramGroup.scale.setScalar(value);
                needsUpdate = true;
                break;

            case 'hologramOpacity':
                this.hologramOpacity = value;
                this.hologramGroup.children.forEach(child => {
                    child.material.opacity = value;
                });
                needsUpdate = true;
                break;

            // Node sizing
            case 'nodeSizeScalingFactor':
                this.nodeSizeScalingFactor = value;
                const graphData = this.graphDataManager.getGraphData();
                if (graphData) {
                    this.updateNodes(graphData.nodes);
                }
                needsUpdate = true;
                break;

            // Bloom effects
            case 'bloomStrength':
                this.bloomStrength = value;
                if (this.bloomComposer) {
                    this.bloomComposer.passes.forEach(pass => {
                        if (pass instanceof UnrealBloomPass) {
                            pass.strength = value;
                        }
                    });
                }
                needsUpdate = true;
                break;

            case 'bloomRadius':
                this.bloomRadius = value;
                if (this.bloomComposer) {
                    this.bloomComposer.passes.forEach(pass => {
                        if (pass instanceof UnrealBloomPass) {
                            pass.radius = value;
                        }
                    });
                }
                needsUpdate = true;
                break;

            case 'bloomThreshold':
                this.bloomThreshold = value;
                if (this.bloomComposer) {
                    this.bloomComposer.passes.forEach(pass => {
                        if (pass instanceof UnrealBloomPass) {
                            pass.threshold = value;
                        }
                    });
                }
                needsUpdate = true;
                break;

            // Force-directed layout controls
            case 'forceDirectedIterations':
                this.forceDirectedIterations = value;
                layoutChanged = true;
                break;

            case 'forceDirectedRepulsion':
                this.forceDirectedRepulsion = value;
                layoutChanged = true;
                break;

            case 'forceDirectedAttraction':
                this.forceDirectedAttraction = value;
                layoutChanged = true;
                break;

            // Additional controls
            case 'labelFontSize':
                this.labelFontSize = value;
                this.updateNodeLabels();
                needsUpdate = true;
                break;

            case 'fogDensity':
                this.fogDensity = value;
                if (this.scene.fog) {
                    this.scene.fog.density = value;
                }
                needsUpdate = true;
                break;
        }

        if (layoutChanged) {
            const graphData = this.graphDataManager.getGraphData();
            if (graphData) {
                this.applyForceDirectedLayout(graphData);
                this.updateNodes(graphData.nodes);
                this.updateEdges(graphData.edges);
            }
        }

        if (needsUpdate) {
            // Ensure composers are updated
            if (this.bloomComposer) this.bloomComposer.render();
            if (this.finalComposer) this.finalComposer.render();
        }
    }

    initThreeJS() {
        console.log('Initializing Three.js');
        const container = document.getElementById('scene-container');
        if (container) {
            container.appendChild(this.renderer.domElement);
        } else {
            console.error("Could not find 'scene-container' element");
            return;
        }

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        this.scene.fog = new THREE.FogExp2(0x000000, this.fogDensity);
        this.addLights();
        this.initPostProcessing();
        this.createHologramStructure();

        window.addEventListener('resize', this.onWindowResize.bind(this), false);

        this.animate();
        console.log('Three.js initialization completed');
    }

    initPostProcessing() {
        console.log('Initializing post-processing');
        
        // Create render targets
        const renderTarget = new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                encoding: THREE.sRGBEncoding
            }
        );

        // Setup bloom composer
        this.bloomComposer = new EffectComposer(this.renderer, renderTarget);
        this.bloomComposer.renderToScreen = false;
        
        const renderScene = new RenderPass(this.scene, this.camera);
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            this.bloomStrength,
            this.bloomRadius,
            this.bloomThreshold
        );

        this.bloomComposer.addPass(renderScene);
        this.bloomComposer.addPass(bloomPass);

        // Setup final composer
        this.finalComposer = new EffectComposer(this.renderer);
        this.finalComposer.addPass(renderScene);

        // Add custom shader pass to combine bloom with scene
        const finalPass = new ShaderPass(
            new THREE.ShaderMaterial({
                uniforms: {
                    baseTexture: { value: null },
                    bloomTexture: { value: this.bloomComposer.renderTarget2.texture }
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D baseTexture;
                    uniform sampler2D bloomTexture;
                    varying vec2 vUv;
                    void main() {
                        vec4 baseColor = texture2D(baseTexture, vUv);
                        vec4 bloomColor = texture2D(bloomTexture, vUv);
                        gl_FragColor = baseColor + vec4(1.0) * bloomColor;
                    }
                `,
                defines: {}
            }),
            "baseTexture"
        );
        finalPass.needsSwap = true;
        this.finalComposer.addPass(finalPass);

        console.log('Post-processing initialized');
    }

    addLights() {
        console.log('Adding lights to the scene');

        // Remove existing lights
        if (this.ambientLight) {
            this.scene.remove(this.ambientLight);
        }
        if (this.directionalLight) {
            this.scene.remove(this.directionalLight);
        }

        // Create and configure ambient light
        this.ambientLight = new THREE.AmbientLight(this.ambientLightColor, this.ambientLightIntensity);
        this.scene.add(this.ambientLight);

        // Create and configure directional light
        this.directionalLight = new THREE.DirectionalLight(this.directionalLightColor, this.directionalLightIntensity);
        this.directionalLight.position.set(50, 50, 50);
        this.scene.add(this.directionalLight);

        console.log('Lights added to the scene');
    }

    applyForceDirectedLayout(graphData) {
        console.log('Applying force-directed layout');
        const nodes = graphData.nodes;
        const edges = graphData.edges;

        // Initialize node velocities
        nodes.forEach(node => {
            node.vx = 0;
            node.vy = 0;
            node.vz = 0;
        });

        for (let iteration = 0; iteration < this.forceDirectedIterations; iteration++) {
            // Calculate repulsion between all nodes
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[j].x - nodes[i].x;
                    const dy = nodes[j].y - nodes[i].y;
                    const dz = nodes[j].z - nodes[i].z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                    const force = this.forceDirectedRepulsion / (distance * distance);

                    // Apply force with damping
                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    const fz = (dz / distance) * force;

                    nodes[i].vx -= fx;
                    nodes[i].vy -= fy;
                    nodes[i].vz -= fz;
                    nodes[j].vx += fx;
                    nodes[j].vy += fy;
                    nodes[j].vz += fz;
                }
            }

            // Calculate attraction along edges
            edges.forEach(edge => {
                const source = nodes.find(node => node.id === edge.source);
                const target = nodes.find(node => node.id === edge.target_node);
                if (source && target) {
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dz = target.z - source.z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                    const force = this.forceDirectedAttraction * distance;

                    // Apply force with damping
                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    const fz = (dz / distance) * force;

                    source.vx += fx;
                    source.vy += fy;
                    source.vz += fz;
                    target.vx -= fx;
                    target.vy -= fy;
                    target.vz -= fz;
                }
            });

            // Update positions with velocity damping
            const damping = 0.9;
            nodes.forEach(node => {
                node.x += node.vx * damping;
                node.y += node.vy * damping;
                node.z += node.vz * damping;
                node.vx *= damping;
                node.vy *= damping;
                node.vz *= damping;
            });
        }

        console.log('Force-directed layout applied');
    }

    calculateNodeSize(fileSize) {
        // More nuanced size calculation using log scale
        const minSize = this.minNodeSize;
        const maxSize = this.maxNodeSize;
        const logMin = Math.log(1);
        const logMax = Math.log(1e9); // 1GB as max reference
        const logSize = Math.log(fileSize + 1);
        
        // Normalize and scale
        const normalizedSize = (logSize - logMin) / (logMax - logMin);
        return minSize + (maxSize - minSize) * normalizedSize;
    }

    calculateNodeColor(lastModified) {
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const dayInMs = 24 * 60 * 60 * 1000;
        
        // Color based on age ranges
        if (age < 7 * dayInMs) {
            return NODE_COLORS.NEW;
        } else if (age < 30 * dayInMs) {
            return NODE_COLORS.MEDIUM;
        } else {
            return NODE_COLORS.OLD;
        }
    }

    createNodeGeometry(size, hyperlinkCount) {
        // Choose geometry based on hyperlink count
        if (hyperlinkCount < 6) {
            return new THREE.SphereGeometry(size, 32, 32);
        } else if (hyperlinkCount < 16) {
            return new THREE.BoxGeometry(size, size, size);
        } else {
            return new THREE.OctahedronGeometry(size);
        }
    }

    updateNodeGeometry(mesh, size, hyperlinkCount) {
        const newGeometry = this.createNodeGeometry(size, hyperlinkCount);
        mesh.geometry.dispose();
        mesh.geometry = newGeometry;
    }

    createNodeLabel(text, fileSize, lastModified, hyperlinkCount) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = `${this.labelFontSize}px Arial`;
        
        // Calculate text dimensions
        const nameMetrics = context.measureText(text);
        const infoText = `${this.formatFileSize(fileSize)} | ${this.formatAge(lastModified)} | ${hyperlinkCount} links`;
        const infoMetrics = context.measureText(infoText);
        
        const textWidth = Math.max(nameMetrics.width, infoMetrics.width);
        canvas.width = textWidth + 20;
        canvas.height = this.labelFontSize * 2 + 30;

        // Draw background
        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);

        // Draw name
        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 10, this.labelFontSize);
        
        // Draw info
        context.font = `${this.labelFontSize / 2}px Arial`;
        context.fillStyle = 'lightgray';
        context.fillText(infoText, 10, this.labelFontSize + 20);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(canvas.width / 10, canvas.height / 10, 1);
        sprite.layers.set(NORMAL_LAYER);

        return sprite;
    }

    updateNodeLabel(label, text, fileSize, lastModified, hyperlinkCount) {
        const canvas = label.material.map.image;
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Update label content
        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.font = `${this.labelFontSize}px Arial`;
        context.fillStyle = 'white';
        context.fillText(text, 10, this.labelFontSize);
        
        const infoText = `${this.formatFileSize(fileSize)} | ${this.formatAge(lastModified)} | ${hyperlinkCount} links`;
        context.font = `${this.labelFontSize / 2}px Arial`;
        context.fillStyle = 'lightgray';
        context.fillText(infoText, 10, this.labelFontSize + 20);

        label.material.map.needsUpdate = true;
    }

    formatFileSize(size) {
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let i = 0;
        while (size >= 1024 && i < units.length - 1) {
            size /= 1024;
            i++;
        }
        return `${size.toFixed(2)} ${units[i]}`;
    }

    formatAge(lastModified) {
        const now = Date.now();
        const age = now - new Date(lastModified).getTime();
        const days = Math.floor(age / (24 * 60 * 60 * 1000));
        
        if (days < 1) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days}d ago`;
        if (days < 30) return `${Math.floor(days / 7)}w ago`;
        if (days < 365) return `${Math.floor(days / 30)}m ago`;
        return `${Math.floor(days / 365)}y ago`;
    }

    updateNodes(nodes) {
        console.log(`Updating nodes: ${nodes.length}`);
        const existingNodeIds = new Set(nodes.map(node => node.id));

        // Remove non-existent nodes
        this.nodeMeshes.forEach((mesh, nodeId) => {
            if (!existingNodeIds.has(nodeId)) {
                this.scene.remove(mesh);
                this.nodeMeshes.delete(nodeId);
                const label = this.nodeLabels.get(nodeId);
                if (label) {
                    this.scene.remove(label);
                    this.nodeLabels.delete(nodeId);
                }
            }
        });

        // Update or create nodes
        nodes.forEach(node => {
            if (!node.id || typeof node.x !== 'number' || typeof node.y !== 'number' || typeof node.z !== 'number') {
                console.warn('Invalid node data:', node);
                return;
            }

            const metadata = node.metadata || {};
            const fileSize = parseInt(metadata.file_size) || 1;
            const lastModified = metadata.last_modified || new Date().toISOString();
            const hyperlinkCount = parseInt(metadata.hyperlink_count) || 0;

            const size = this.calculateNodeSize(fileSize);
            const color = this.calculateNodeColor(lastModified);

            let mesh = this.nodeMeshes.get(node.id);

            if (!mesh) {
                // Create new node
                const geometry = this.createNodeGeometry(size, hyperlinkCount);
                const material = new THREE.MeshStandardMaterial({
                    color: color,
                    metalness: 0.5,
                    roughness: 0.5,
                    emissive: color,
                    emissiveIntensity: 0.2
                });

                mesh = new THREE.Mesh(geometry, material);
                mesh.layers.enable(BLOOM_LAYER);
                this.scene.add(mesh);
                this.nodeMeshes.set(node.id, mesh);

                // Create label
                const label = this.createNodeLabel(node.label || node.id, fileSize, lastModified, hyperlinkCount);
                label.layers.set(NORMAL_LAYER);
                this.scene.add(label);
                this.nodeLabels.set(node.id, label);
            } else {
                // Update existing node
                this.updateNodeGeometry(mesh, size, hyperlinkCount);
                mesh.material.color = color;
                mesh.material.emissive = color;
            }

            // Update positions
            mesh.position.set(node.x, node.y, node.z);
            const label = this.nodeLabels.get(node.id);
            if (label) {
                label.position.set(node.x, node.y + size + 2, node.z);
                this.updateNodeLabel(label, node.label || node.id, fileSize, lastModified, hyperlinkCount);
            }
        });
    }

    updateEdges(edges) {
        console.log(`Updating edges: ${edges.length}`);
        const existingEdgeKeys = new Set(edges.map(edge => `${edge.source}-${edge.target_node}`));

        // Remove non-existent edges
        this.edgeMeshes.forEach((line, edgeKey) => {
            if (!existingEdgeKeys.has(edgeKey)) {
                this.scene.remove(line);
                this.edgeMeshes.delete(edgeKey);
            }
        });

        // Update or create edges
        edges.forEach(edge => {
            if (!edge.source || !edge.target_node) {
                console.warn('Invalid edge data:', edge);
                return;
            }

            const edgeKey = `${edge.source}-${edge.target_node}`;
            let line = this.edgeMeshes.get(edgeKey);
            const sourceMesh = this.nodeMeshes.get(edge.source);
            const targetMesh = this.nodeMeshes.get(edge.target_node);

            if (!line && sourceMesh && targetMesh) {
                // Create new edge
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(6);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                const material = new THREE.LineBasicMaterial({
                    color: this.edgeColor,
                    transparent: true,
                    opacity: this.edgeOpacity,
                    linewidth: 2
                });

                line = new THREE.Line(geometry, material);
                line.layers.set(NORMAL_LAYER);
                this.scene.add(line);
                this.edgeMeshes.set(edgeKey, line);
            }

            if (line && sourceMesh && targetMesh) {
                // Update edge positions
                const positions = line.geometry.attributes.position.array;
                positions[0] = sourceMesh.position.x;
                positions[1] = sourceMesh.position.y;
                positions[2] = sourceMesh.position.z;
                positions[3] = targetMesh.position.x;
                positions[4] = targetMesh.position.y;
                positions[5] = targetMesh.position.z;
                line.geometry.attributes.position.needsUpdate = true;
            }
        });
    }

    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        this.controls.update();

        // Animate hologram
        this.hologramGroup.children.forEach(child => {
            child.rotation.x += child.userData.rotationSpeed;
            child.rotation.y += child.userData.rotationSpeed;
        });

        // Update label orientations
        this.nodeLabels.forEach(label => {
            label.lookAt(this.camera.position);
        });

        // Render with bloom effect
        this.camera.layers.set(BLOOM_LAYER);
        this.bloomComposer.render();
        
        this.camera.layers.set(NORMAL_LAYER);
        this.finalComposer.render();
    }

    onWindowResize() {
        console.log('Window resized');
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        
        // Update composers
        this.bloomComposer.setSize(window.innerWidth, window.innerHeight);
        this.finalComposer.setSize(window.innerWidth, window.innerHeight);
    }

    dispose() {
        console.log('Disposing WebXRVisualization');
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        // Dispose of all geometries and materials
        this.scene.traverse(object => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => {
                        if (material.map) material.map.dispose();
                        material.dispose();
                    });
                } else {
                    if (object.material.map) object.material.map.dispose();
                    object.material.dispose();
                }
            }
        });

        // Dispose of render targets
        if (this.bloomComposer) {
            this.bloomComposer.renderTarget1.dispose();
            this.bloomComposer.renderTarget2.dispose();
        }
        if (this.finalComposer) {
            this.finalComposer.renderTarget1.dispose();
            this.finalComposer.renderTarget2.dispose();
        }

        this.renderer.dispose();
        if (this.controls) {
            this.controls.dispose();
        }

        console.log('WebXRVisualization disposed');
    }

    handleSpacemouseInput(x, y, z, rx, ry, rz) {
        if (!this.camera || !this.controls) {
            console.warn('Camera or controls not initialized for Spacemouse input');
            return;
        }

        // Apply translation
        this.camera.position.x += x * TRANSLATION_SPEED;
        this.camera.position.y += y * TRANSLATION_SPEED;
        this.camera.position.z += z * TRANSLATION_SPEED;
        
        // Apply rotation
        this.camera.rotation.x += rx * ROTATION_SPEED;
        this.camera.rotation.y += ry * ROTATION_SPEED;
        this.camera.rotation.z += rz * ROTATION_SPEED;

        this.controls.update();
    }

    debugLabels() {
        console.log('Debugging labels');
        console.log('Total labels:', this.nodeLabels.size);

        this.camera.updateMatrixWorld();
        this.camera.updateProjectionMatrix();

        const frustum = new THREE.Frustum();
        const cameraViewProjectionMatrix = new THREE.Matrix4();
        cameraViewProjectionMatrix.multiplyMatrices(
            this.camera.projectionMatrix,
            this.camera.matrixWorldInverse
        );
        frustum.setFromProjectionMatrix(cameraViewProjectionMatrix);

        this.nodeLabels.forEach((label, nodeId) => {
            console.log(`Label for node ${nodeId}:`, {
                position: label.position.toArray(),
                visible: label.visible,
                inFrustum: frustum.containsPoint(label.position),
                material: {
                    color: label.material.color.getHex(),
                    opacity: label.material.opacity,
                    transparent: label.material.transparent,
                    visible: label.material.visible
                },
                geometry: {
                    type: label.geometry.type,
                    parameters: label.geometry.parameters
                }
            });
        });
    }

    createHologramStructure() {
        // Clear existing hologram structure
        while (this.hologramGroup.children.length > 0) {
            const child = this.hologramGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.hologramGroup.remove(child);
        }

        // Create new hologram structure
        const hologramGeometry = new THREE.TorusGeometry(100, 3, 16, 100);
        const hologramMaterial = new THREE.MeshStandardMaterial({
            color: this.hologramColor,
            emissive: this.hologramColor,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: this.hologramOpacity,
            metalness: 0.8,
            roughness: 0.2
        });

        // Create multiple rings with different orientations
        for (let i = 0; i < 3; i++) {
            const ring = new THREE.Mesh(hologramGeometry, hologramMaterial);
            ring.rotation.x = Math.PI / 2 * i;
            ring.rotation.y = Math.PI / 4 * i;
            ring.userData.rotationSpeed = 0.002 * (i + 1);
            this.hologramGroup.add(ring);
        }

        this.scene.add(this.hologramGroup);
    }

    updateVisualization() {
        console.log('Updating visualization');
        const graphData = this.graphDataManager.getGraphData();
        if (!graphData) {
            console.warn('No graph data available');
            return;
        }

        this.applyForceDirectedLayout(graphData);
        this.updateNodes(graphData.nodes);
        this.updateEdges(graphData.edges);
    }
}
