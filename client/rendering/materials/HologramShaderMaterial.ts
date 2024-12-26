import { ShaderMaterial, UniformsUtils, Vector2 } from 'three';

export class HologramShaderMaterial extends ShaderMaterial {
    constructor(parameters: any) {
        super({
            uniforms: UniformsUtils.merge([
                {
                    time: { value: 0 },
                    color: { value: null },
                    texture: { value: null },
                    resolution: { value: new Vector2(window.innerWidth, window.innerHeight) }
                },
            ]),
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                uniform sampler2D texture;
                uniform vec2 resolution;
                varying vec2 vUv;

                void main() {
                    vec2 uv = vUv;
                    float distortion = sin(time * 2.0 + uv.x * 10.0) * 0.02;
                    uv.y += distortion;
                    vec4 texColor = texture2D(texture, uv);
                    float alpha = 0.5 + 0.5 * sin(time * 2.0 + uv.x * 10.0);
                    gl_FragColor = vec4(color, alpha) * texColor;
                }
            `,
        });
        this.setValues(parameters);
    }
}
