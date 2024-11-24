import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { LAYERS } from '../layerManager.js';

export class CompositionEffect {
    constructor(renderer) {
        this.renderer = renderer;
        this.composer = null;
    }

    init(bloomRenderTargets) {
        const renderTarget = new THREE.WebGLRenderTarget(
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

        this.composer = new EffectComposer(this.renderer, renderTarget);
        
        const shader = {
            uniforms: {
                baseTexture: { value: null },
                bloomTexture0: { value: bloomRenderTargets.get(LAYERS.BLOOM).texture },
                bloomTexture1: { value: bloomRenderTargets.get(LAYERS.HOLOGRAM).texture },
                bloomTexture2: { value: bloomRenderTargets.get(LAYERS.EDGE).texture }
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
                uniform sampler2D bloomTexture0;
                uniform sampler2D bloomTexture1;
                uniform sampler2D bloomTexture2;
                varying vec2 vUv;

                void main() {
                    vec4 baseColor = texture2D(baseTexture, vUv);
                    vec3 bloomColor0 = texture2D(bloomTexture0, vUv).rgb;
                    vec3 bloomColor1 = texture2D(bloomTexture1, vUv).rgb;
                    vec3 bloomColor2 = texture2D(bloomTexture2, vUv).rgb;
                    
                    vec3 color = baseColor.rgb + bloomColor0 + bloomColor1 + bloomColor2;
                    
                    gl_FragColor = vec4(color, baseColor.a);
                }
            `
        };
        
        const finalPass = new ShaderPass(new THREE.ShaderMaterial(shader));
        finalPass.renderToScreen = true;
        this.composer.addPass(finalPass);
    }

    render(baseTexture) {
        const finalPass = this.composer.passes[0];
        finalPass.uniforms.baseTexture.value = baseTexture;
        this.composer.render();
    }

    resize(width, height) {
        this.composer.setSize(width, height);
    }

    dispose() {
        this.composer.dispose();
    }
}
