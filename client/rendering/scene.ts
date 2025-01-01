import {
    Scene as ThreeScene,
    PerspectiveCamera,
    WebGLRenderer,
    AmbientLight,
    DirectionalLight,
    GridHelper,
    Color,
    Object3D,
    Vector3,
    Camera
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { Settings } from '../types/settings';

export class SceneManager {
    private scene: ThreeScene;
    private camera: PerspectiveCamera;
    private renderer: WebGLRenderer;
    private composer: EffectComposer;
    private controls: OrbitControls;
    private renderCallbacks: Array<() => void> = [];
    private isDisposed = false;
    private grid: GridHelper | null = null;
    constructor(container: HTMLElement, settings: Settings) {
        this.scene = new ThreeScene();
        this.camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(this.renderer.domElement);

        this.composer = new EffectComposer(this.renderer);
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
            new Vector3(window.innerWidth, window.innerHeight),
            1.5,
            0.4,
            0.85
        );
        this.composer.addPass(bloomPass);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        this.setupScene(settings);
        this.setupEventListeners();
        this.animate();
    }

    private setupScene(settings: Settings): void {
        // Set up camera
        this.camera.position.z = 5;
        this.camera.position.y = 2;
        this.camera.position.x = 2;
        this.camera.lookAt(0, 0, 0);

        // Add lights
        const ambientLight = new AmbientLight(0x404040);
        this.scene.add(ambientLight);

        const directionalLight = new DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Add grid if enabled
        if (settings.render.showGrid) {
            this.grid = new GridHelper(10, 10);
            this.scene.add(this.grid);
        }

        // Set background color
        const backgroundColor = new Color(settings.render.backgroundColor);
        this.scene.background = backgroundColor;
        this.renderer.setClearColor(backgroundColor);
    }

    private setupEventListeners(): void {
        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    private removeEventListeners(): void {
        window.removeEventListener('resize', this.onWindowResize.bind(this));
    }

    private onWindowResize(): void {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
        this.composer.setSize(width, height);
    }

    public addRenderCallback(callback: () => void): void {
        this.renderCallbacks.push(callback);
    }

    public removeRenderCallback(callback: () => void): void {
        const index = this.renderCallbacks.indexOf(callback);
        if (index !== -1) {
            this.renderCallbacks.splice(index, 1);
        }
    }

    public addObject(object: Object3D): void {
        this.scene.add(object);
    }

    public removeObject(object: Object3D): void {
        this.scene.remove(object);
    }

    private animate = (): void => {
        if (this.isDisposed) return;

        requestAnimationFrame(this.animate);

        this.controls.update();

        // Execute render callbacks
        for (const callback of this.renderCallbacks) {
            callback();
        }

        this.composer.render();
    };

    public dispose(): void {
        this.isDisposed = true;
        this.removeEventListeners();

        // Dispose of Three.js objects
        this.scene.traverse((object: Object3D) => {
            if (object instanceof Object3D) {
                if (object.geometry) {
                    object.geometry.dispose();
                }
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(material => material.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            }
        });

        this.renderer.dispose();
        this.composer.dispose();
    }

    public updateSettings(settings: Settings): void {
        // Update background color
        const backgroundColor = new Color(settings.render.backgroundColor);
        this.scene.background = backgroundColor;
        this.renderer.setClearColor(backgroundColor);

        // Update grid visibility
        if (settings.render.showGrid) {
            if (!this.grid) {
                this.grid = new GridHelper(10, 10);
                this.scene.add(this.grid);
            }
        } else if (this.grid) {
            this.scene.remove(this.grid);
            this.grid = null;
        }

        // Update controls
        this.controls.autoRotate = settings.controls.autoRotate;
        this.controls.rotateSpeed = settings.controls.rotateSpeed;
        this.controls.zoomSpeed = settings.controls.zoomSpeed;
        this.controls.panSpeed = settings.controls.panSpeed;
    }

    public getCamera(): Camera {
        return this.camera;
    }

    public getScene(): ThreeScene {
        return this.scene;
    }

    public getRenderer(): WebGLRenderer {
        return this.renderer;
    }
}
