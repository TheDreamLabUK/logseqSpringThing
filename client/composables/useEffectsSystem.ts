import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { SSAOPass } from 'three/examples/jsm/postprocessing/SSAOPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/examples/jsm/shaders/FXAAShader.js';
import { useSettingsStore } from '@stores/settings';
import { usePlatform } from './usePlatform';
import { PASS_OUTPUT } from '../utils/threeUtils';

// Configure color management for modern Three.js
THREE.ColorManagement.enabled = true;

interface ExtendedUnrealBloomPass extends UnrealBloomPass {
  selectedObjects?: THREE.Object3D[];
}

export function useEffectsSystem(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera
) {
  const settingsStore = useSettingsStore();
  const { getPlatformInfo } = usePlatform();

  // Effect composer and passes
  const composer = ref<EffectComposer | null>(null);
  const bloomPass = ref<ExtendedUnrealBloomPass | null>(null);
  const ssaoPass = ref<SSAOPass | null>(null);
  const fxaaPass = ref<ShaderPass | null>(null);

  // Settings
  const bloomSettings = computed(() => settingsStore.getBloomSettings);
  const platformInfo = computed(() => getPlatformInfo());

  // Resolution
  const resolution = computed(() => {
    const pixelRatio = renderer.getPixelRatio();
    const size = new THREE.Vector2();
    renderer.getSize(size);
    return new THREE.Vector2(
      size.width * pixelRatio,
      size.height * pixelRatio
    );
  });

  // Initialize effect composer
  const initializeComposer = () => {
    // Create render target with appropriate color space
    const renderTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight,
      {
        colorSpace: THREE.SRGBColorSpace,
        samples: (renderer as any).capabilities.isWebGL2 ? 4 : 0
      }
    );

    // Create new composer
    composer.value = new EffectComposer(renderer as any, renderTarget as any);

    // Add render pass
    const renderPass = new RenderPass(scene as any, camera as any);
    (composer.value as any).addPass(renderPass);

    // Initialize bloom if enabled
    if (bloomSettings.value.enabled) {
      const bloom = new UnrealBloomPass(
        resolution.value,
        bloomSettings.value.strength,
        bloomSettings.value.radius,
        bloomSettings.value.threshold
      ) as ExtendedUnrealBloomPass;
      bloomPass.value = bloom;
      (composer.value as any).addPass(bloom);
    }

    // Initialize SSAO for browser platform
    if (platformInfo.value.isBrowser) {
      const ssao = new SSAOPass(scene as any, camera as any);
      ssao.output = PASS_OUTPUT.Default;
      ssaoPass.value = ssao;
      (composer.value as any).addPass(ssao);
    }

    // Initialize FXAA
    const fxaa = new ShaderPass(FXAAShader);
    const uniforms = fxaa.material.uniforms;
    if (uniforms['resolution']) {
      uniforms['resolution'].value.x = 1 / resolution.value.x;
      uniforms['resolution'].value.y = 1 / resolution.value.y;
    }
    fxaaPass.value = fxaa;
    (composer.value as any).addPass(fxaa);
  };

  // Update effect settings
  const updateBloomSettings = () => {
    if (!bloomPass.value) return;

    const settings = bloomSettings.value;
    bloomPass.value.strength = settings.strength;
    bloomPass.value.radius = settings.radius;
    bloomPass.value.threshold = settings.threshold;

    // Update selective bloom settings
    const selectedObjects: THREE.Object3D[] = [];

    // Add objects based on bloom settings
    scene.traverse((object) => {
      if (object.userData.bloomLayer) {
        if (settings.node_bloom_strength > 0 && object.userData.type === 'node') {
          selectedObjects.push(object);
        }
        if (settings.edge_bloom_strength > 0 && object.userData.type === 'edge') {
          selectedObjects.push(object);
        }
        if (settings.environment_bloom_strength > 0 && object.userData.type === 'environment') {
          selectedObjects.push(object);
        }
      }
    });

    // Set selected objects for bloom
    bloomPass.value.selectedObjects = selectedObjects;
  };

  // Handle resize
  const handleResize = () => {
    if (!composer.value || !fxaaPass.value) return;

    const size = new THREE.Vector2();
    renderer.getSize(size);
    composer.value.setSize(size.width, size.height);

    // Update FXAA resolution
    const pixelRatio = renderer.getPixelRatio();
    const uniforms = fxaaPass.value.material.uniforms;
    if (uniforms['resolution']) {
      uniforms['resolution'].value.x = 1 / (size.width * pixelRatio);
      uniforms['resolution'].value.y = 1 / (size.height * pixelRatio);
    }
  };

  // Render function
  const render = () => {
    if (composer.value) {
      composer.value.render();
    }
  };

  // Watch for settings changes
  watch(() => bloomSettings.value, () => {
    updateBloomSettings();
  }, { deep: true });

  // Lifecycle
  onMounted(() => {
    initializeComposer();
    window.addEventListener('resize', handleResize);
  });

  onBeforeUnmount(() => {
    window.removeEventListener('resize', handleResize);
    
    // Dispose of resources
    if (composer.value) {
      composer.value.passes.forEach(pass => {
        if ('dispose' in pass && typeof pass.dispose === 'function') {
          pass.dispose();
        }
      });
    }

    // Clear references
    bloomPass.value = null;
    ssaoPass.value = null;
    fxaaPass.value = null;
    composer.value = null;
  });

  return {
    composer,
    bloomPass,
    ssaoPass,
    fxaaPass,
    render,
    handleResize,
    updateBloomSettings
  };
}
