import { ref, onBeforeUnmount } from 'vue';
import THREE, { OrbitControls } from '../utils/three';
import type { Scene, PerspectiveCamera, WebGLRenderer, Object3D, Material, Texture } from 'three';

interface ThreeResources {
  scene: Scene;
  camera: PerspectiveCamera;
  renderer: WebGLRenderer;
  controls?: OrbitControls;
  environment?: {
    ground: THREE.Mesh;
    gridHelper: THREE.GridHelper;
    dispose: () => void;
  };
}

interface MaterialWithMap extends Material {
  map?: Texture | null;
}

export function useThreeScene() {
  const resources = ref<ThreeResources | null>(null);

  const handleContextLost = (event: Event) => {
    event.preventDefault();
    console.warn('WebGL context lost. Attempting to restore...');
  };

  const handleContextRestored = () => {
    console.log('WebGL context restored');
    window.dispatchEvent(new Event('webglcontextrestored'));
  };

  const setupLighting = (scene: Scene) => {
    // Main directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    
    // Optimize shadow map settings
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 500;
    directionalLight.shadow.bias = -0.0001;
    
    scene.add(directionalLight);

    // Fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);

    // Ambient light for overall scene brightness
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
  };

  const initScene = async () => {
    try {
      // Create scene
      const scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x000000, 0.002);

      // Create camera
      const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );
      camera.position.set(0, 1.6, 3);

      // Create renderer
      const canvas = document.createElement('canvas');
      const contextAttributes = {
        alpha: true,
        antialias: true,
        powerPreference: "high-performance" as WebGLPowerPreference,
        failIfMajorPerformanceCaveat: false,
        preserveDrawingBuffer: false
      };

      // Try WebGL2 first
      let gl = canvas.getContext('webgl2', contextAttributes);
      let isWebGL2 = !!gl;

      if (!gl) {
        console.warn('WebGL2 not available, falling back to WebGL1');
        gl = canvas.getContext('webgl', contextAttributes) ||
             canvas.getContext('experimental-webgl', contextAttributes);
        isWebGL2 = false;
      }

      if (!gl) {
        throw new Error('WebGL not supported');
      }

      const renderer = new THREE.WebGLRenderer({
        canvas,
        context: gl as WebGLRenderingContext | WebGL2RenderingContext,
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
        preserveDrawingBuffer: false
      });

      // Configure renderer based on WebGL version
      if (isWebGL2) {
        console.log('Using WebGL2 renderer');
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      } else {
        console.log('Using WebGL1 renderer');
        renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
        renderer.shadowMap.type = THREE.PCFShadowMap;
      }

      // Common renderer settings
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.xr.enabled = true;

      // Store WebGL version
      (renderer as any).capabilities.isWebGL2 = isWebGL2;

      // Add context loss handling
      renderer.domElement.addEventListener('webglcontextlost', handleContextLost, false);
      renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored, false);

      // Set up lighting
      setupLighting(scene);

      // Append renderer to DOM
      const container = document.getElementById('scene-container');
      if (container) {
        container.appendChild(renderer.domElement);
      } else {
        document.body.appendChild(renderer.domElement);
      }

      // Create orbit controls
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.maxPolarAngle = Math.PI * 0.95;
      controls.minDistance = 1;
      controls.maxDistance = 50;
      controls.enablePan = true;
      controls.panSpeed = 0.5;
      controls.rotateSpeed = 0.5;
      controls.zoomSpeed = 0.5;

      // Disable controls in XR
      renderer.xr.addEventListener('sessionstart', () => {
        controls.enabled = false;
      });
      
      renderer.xr.addEventListener('sessionend', () => {
        controls.enabled = true;
      });

      // Create basic environment
      const environment = createBasicEnvironment(scene);

      // Store resources
      resources.value = {
        scene,
        camera,
        renderer,
        controls,
        environment
      };

      // Handle window resize
      window.addEventListener('resize', handleResize);

      return resources.value;

    } catch (error) {
      console.error('Error initializing Three.js scene:', error);
      throw error;
    }
  };

  const createBasicEnvironment = (scene: Scene) => {
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x808080,
      roughness: 0.8,
      metalness: 0.2,
      transparent: true,
      opacity: 0.8
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    const gridHelper = new THREE.GridHelper(100, 100);
    (gridHelper.material as THREE.Material).transparent = true;
    (gridHelper.material as THREE.Material).opacity = 0.2;
    scene.add(gridHelper);

    return {
      ground,
      gridHelper,
      dispose: () => {
        groundGeometry.dispose();
        groundMaterial.dispose();
        if (gridHelper.material) {
          (gridHelper.material as THREE.Material).dispose();
        }
        if (gridHelper.geometry) {
          gridHelper.geometry.dispose();
        }
        scene.remove(ground);
        scene.remove(gridHelper);
      }
    };
  };

  const handleResize = () => {
    if (!resources.value) return;

    const { camera, renderer } = resources.value;
    if (!renderer.xr.isPresenting) {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
  };

  const dispose = () => {
    if (!resources.value) return;

    const { scene, renderer, controls, environment } = resources.value;

    // Dispose of scene objects
    scene.traverse((object: Object3D) => {
      if ((object as THREE.Mesh).geometry) {
        (object as THREE.Mesh).geometry.dispose();
      }
      
      if ((object as THREE.Mesh).material) {
        const material = (object as THREE.Mesh).material;
        if (Array.isArray(material)) {
          material.forEach(mat => {
            const matWithMap = mat as MaterialWithMap;
            if (matWithMap.map) matWithMap.map.dispose();
            mat.dispose();
          });
        } else {
          const matWithMap = material as MaterialWithMap;
          if (matWithMap.map) matWithMap.map.dispose();
          material.dispose();
        }
      }
    });

    // Dispose of environment
    if (environment) {
      environment.dispose();
    }

    // Dispose of renderer
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();

    // Remove context loss listeners
    renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
    renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored);

    // Dispose of controls
    if (controls) {
      controls.dispose();
    }

    // Remove resize listener
    window.removeEventListener('resize', handleResize);

    resources.value = null;
  };

  onBeforeUnmount(() => {
    dispose();
  });

  return {
    resources,
    initScene,
    dispose
  };
}
