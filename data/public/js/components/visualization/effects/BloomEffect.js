import * as THREE from 'three';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { LAYERS } from '../layerManager.js';

export class BloomEffect {
    constructor(renderer, scene, camera) {
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.composers = new Map();
        this.renderTargets = new Map();
    }

    createRenderTarget() {
        return new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                colorSpace: THREE.SRGBColorSpace,
                stencilBuffer: false,
                depthBuffer: true
            }
        );
    }

    init(settings) {
        const layers = [
            {
                layer: LAYERS.BLOOM,
                settings: {
                    strength: settings.nodeBloomStrength * 1.2,
                    radius: settings.nodeBloomRadius,
                    threshold: settings.nodeBloomThreshold * 0.8
                }
            },
            {
                layer: LAYERS.HOLOGRAM,
                settings: {
                    strength: settings.environmentBloomStrength * 1.5,
                    radius: settings.environmentBloomRadius * 1.2,
                    threshold: settings.environmentBloomThreshold * 0.7
                }
            },
            {
                layer: LAYERS.EDGE,
                settings: {
                    strength: settings.edgeBloomStrength * 1.3,
                    radius: settings.edgeBloomRadius,
                    threshold: settings.edgeBloomThreshold * 0.9
                }
            }
        ];

        layers.forEach(({ layer, settings }) => {
            const renderTarget = this.createRenderTarget();
            this.renderTargets.set(layer, renderTarget);
            
            const composer = new EffectComposer(this.renderer, renderTarget);
            composer.renderToScreen = false;
            
            const renderPass = new RenderPass(this.scene, this.camera);
            renderPass.clear = true;
            
            const bloomPass = new UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                settings.strength,
                settings.radius,
                settings.threshold
            );
            
            composer.addPass(renderPass);
            composer.addPass(bloomPass);
            
            this.composers.set(layer, composer);
        });
    }

    render(currentCamera) {
        this.composers.forEach((composer, layer) => {
            const originalLayerMask = currentCamera.layers.mask;
            currentCamera.layers.set(layer);
            composer.render();
            currentCamera.layers.mask = originalLayerMask;
        });
    }

    resize(width, height) {
        this.renderTargets.forEach(target => target.setSize(width, height));
        this.composers.forEach(composer => composer.setSize(width, height));
    }

    dispose() {
        this.renderTargets.forEach(target => target.dispose());
        this.composers.forEach(composer => composer.dispose());
    }

    getRenderTargets() {
        return this.renderTargets;
    }
}
