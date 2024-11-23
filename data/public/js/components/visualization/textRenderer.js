import * as THREE from 'three';

// SDF font atlas generation
function generateSDFData(text, fontSize, padding) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to power of 2 for better texture performance
    const size = Math.pow(2, Math.ceil(Math.log2(fontSize * 2 + padding * 2)));
    canvas.width = size;
    canvas.height = size;
    
    // Setup font
    ctx.font = `${fontSize}px Arial`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    
    // Draw text
    ctx.fillStyle = 'white';
    ctx.fillText(text, size/2, size/2);
    
    // Generate SDF
    const imageData = ctx.getImageData(0, 0, size, size);
    const sdf = new Float32Array(size * size);
    
    // Calculate SDF values
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const idx = (y * size + x) * 4;
            const alpha = imageData.data[idx + 3] / 255;
            
            // Calculate distance field
            let minDist = Number.MAX_VALUE;
            const maxSearchDist = fontSize / 2;
            
            for (let sy = -maxSearchDist; sy <= maxSearchDist; sy++) {
                for (let sx = -maxSearchDist; sx <= maxSearchDist; sx++) {
                    const sampX = x + sx;
                    const sampY = y + sy;
                    
                    if (sampX >= 0 && sampX < size && sampY >= 0 && sampY < size) {
                        const sampIdx = (sampY * size + sampX) * 4;
                        const sampAlpha = imageData.data[sampIdx + 3] / 255;
                        
                        if (sampAlpha !== alpha) {
                            const dist = Math.sqrt(sx*sx + sy*sy);
                            minDist = Math.min(minDist, dist);
                        }
                    }
                }
            }
            
            // Normalize and store SDF value
            sdf[y * size + x] = alpha === 1 ? minDist / maxSearchDist : -minDist / maxSearchDist;
        }
    }
    
    return {
        data: sdf,
        size: size,
        texture: new THREE.DataTexture(
            sdf,
            size,
            size,
            THREE.RedFormat,
            THREE.FloatType
        )
    };
}

export class TextRenderer {
    constructor() {
        // SDF shader for high-quality text rendering
        this.material = new THREE.ShaderMaterial({
            uniforms: {
                sdfTexture: { value: null },
                color: { value: new THREE.Color(0xffffff) },
                smoothing: { value: 0.25 },
                threshold: { value: 0.5 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D sdfTexture;
                uniform vec3 color;
                uniform float smoothing;
                uniform float threshold;
                varying vec2 vUv;
                
                void main() {
                    float sdf = texture2D(sdfTexture, vUv).r;
                    float alpha = smoothstep(threshold - smoothing, threshold + smoothing, sdf);
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            depthWrite: false,
            side: THREE.DoubleSide
        });
    }
    
    createTextSprite(text, options = {}) {
        const {
            fontSize = 32,
            padding = 8,
            color = 0xffffff,
            backgroundColor = 0x000000,
            backgroundOpacity = 0.85
        } = options;
        
        // Generate SDF data
        const sdfData = generateSDFData(text, fontSize, padding);
        
        // Create geometry
        const geometry = new THREE.PlaneGeometry(1, 1);
        
        // Update material with new texture
        const material = this.material.clone();
        material.uniforms.sdfTexture.value = sdfData.texture;
        material.uniforms.color.value = new THREE.Color(color);
        
        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        
        // Add background plane if needed
        if (backgroundOpacity > 0) {
            const bgGeometry = new THREE.PlaneGeometry(1.1, 1.1);
            const bgMaterial = new THREE.MeshBasicMaterial({
                color: backgroundColor,
                transparent: true,
                opacity: backgroundOpacity,
                depthWrite: false
            });
            const background = new THREE.Mesh(bgGeometry, bgMaterial);
            background.position.z = -0.001;
            mesh.add(background);
        }
        
        // Scale mesh based on texture size
        const scale = fontSize / sdfData.size;
        mesh.scale.set(sdfData.size * scale, sdfData.size * scale, 1);
        
        return mesh;
    }
    
    dispose() {
        this.material.dispose();
    }
}
