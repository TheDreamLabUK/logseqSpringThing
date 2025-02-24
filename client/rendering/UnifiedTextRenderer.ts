import {
    Scene,
    Camera,
    Group,
    Texture,
    ShaderMaterial,
    BufferGeometry,
    NearestFilter,
    ClampToEdgeWrapping,
    InstancedBufferAttribute,
    PlaneGeometry,
    Mesh,
    Vector3,
    Color,
    NormalBlending,
    MeshBasicMaterial
} from 'three';
import { createLogger } from '../core/logger';
import { LabelSettings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
import { SDFFontAtlasGenerator } from './SDFFontAtlasGenerator';
import '../types/three-ext.d';

const logger = createLogger('UnifiedTextRenderer');

// Vertex shader for SDF text rendering with instancing
const vertexShader = `
    uniform vec3 cameraPosition;
    attribute vec4 textureCoord;
    attribute vec3 instancePosition;
    attribute vec4 instanceColor;
    attribute float instanceScale;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    
    void main() {
        vUv = uv;
        vColor = instanceColor;
        vScale = instanceScale;

        // Scale the position first
        vec3 scale = vec3(instanceScale);
        vec3 vertexPosition = position * scale;
        vertexPosition += instancePosition;  // Add instance position after scaling
        
        #ifdef BILLBOARD_VERTICAL
            // Calculate billboard rotation based on camera position
            vec4 worldPosition = modelMatrix * vec4(vertexPosition, 1.0);
            float angle = atan(cameraPosition.x - worldPosition.x, cameraPosition.z - worldPosition.z);
            mat4 billboardMatrix = mat4(
                cos(angle), 0.0, sin(angle), 0.0,
                0.0, 1.0, 0.0, 0.0,
                -sin(angle), 0.0, cos(angle), 0.0,
                0.0, 0.0, 0.0, 1.0
            );
            vertexPosition = (billboardMatrix * vec4(vertexPosition, 1.0)).xyz;
        #endif
        
        vec4 mvPosition = modelViewMatrix * vec4(vertexPosition, 1.0);
        gl_Position = projectionMatrix * mvPosition;
    }
`;

// Fragment shader for SDF text rendering
const fragmentShader = `
    uniform sampler2D fontAtlas;
    uniform float sdfThreshold;
    uniform float sdfSpread;
    uniform vec3 outlineColor;
    uniform float outlineWidth;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    
    float median(float r, float g, float b) {
        return max(min(r, g), min(max(r, g), b));
    }
    
    void main() {
        vec3 sample = texture2D(fontAtlas, vUv).rgb;
        float sigDist = median(sample.r, sample.g, sample.b);
        
        float alpha = smoothstep(sdfThreshold - sdfSpread, sdfThreshold + sdfSpread, sigDist);
        float outline = smoothstep(sdfThreshold - outlineWidth - sdfSpread, 
                                 sdfThreshold - outlineWidth + sdfSpread, sigDist);
        
        vec4 color = mix(vec4(outlineColor, outline), vColor, alpha);
        gl_FragColor = color;
    }
`;

interface LabelInstance {
    id: string;
    text: string;
    position: Vector3;
    scale: number;
    color: Color;
    visible: boolean;
}

export class UnifiedTextRenderer {
    private scene: Scene;
    private camera: Camera;
    private group: Group;
    private material: ShaderMaterial | MeshBasicMaterial;
    private geometry: BufferGeometry;
    private mesh: Mesh;
    private fontAtlas: Texture | null;
    private labels: Map<string, LabelInstance>;
    private settings: LabelSettings;
    private maxInstances: number;
    private currentInstanceCount: number;
    private logger = createLogger('UnifiedTextRenderer');
    private fontAtlasGenerator: SDFFontAtlasGenerator;
    
    constructor(camera: Camera, scene: Scene, settings: LabelSettings) {
        this.scene = scene;
        this.camera = camera;
        this.settings = settings;
        logger.info('UnifiedTextRenderer settings:', {
            enableLabels: this.settings.enableLabels,
            desktopFontSize: this.settings.desktopFontSize,
            textColor: this.settings.textColor,
            billboardMode: this.settings.billboardMode
        });

        this.labels = new Map();
        this.maxInstances = 1000;
        this.currentInstanceCount = 0;
        this.fontAtlas = null;
        
        this.group = new Group();
        this.scene.add(this.group);
        
        this.fontAtlasGenerator = new SDFFontAtlasGenerator(1024, 4, 8);

        this.logger.info('Initializing material with settings:', {
            billboardMode: settings.billboardMode,
            sdfThreshold: 0.5,
            sdfSpread: 0.1,
            outlineColor: settings.textOutlineColor,
            outlineWidth: settings.textOutlineWidth,
            depthTest: true
        });
        
        // Initialize the material with error handling
        try {
            this.material = new ShaderMaterial({
                vertexShader,
                fragmentShader,
                uniforms: {
                    fontAtlas: { value: null },
                    sdfThreshold: { value: 0.5 },
                    sdfSpread: { value: 0.1 },
                    cameraPosition: { value: this.camera.position },
                    outlineColor: { value: new Color(settings.textOutlineColor) },
                    outlineWidth: { value: settings.textOutlineWidth }
                },
                transparent: true,
                depthTest: true,
                depthWrite: false,
                blending: NormalBlending
            });

            // Force shader compilation
            this.material.needsUpdate = true;
        } catch (error) {
            logger.error('Failed to initialize text shader:', {
                error,
                message: error instanceof Error ? error.message : String(error),
                stack: error instanceof Error ? error.stack : undefined
            });
            // Fallback to basic material
            this.material = new MeshBasicMaterial({ 
                color: new Color(this.settings.textColor) });
        }
        
        this.geometry = this.createInstancedGeometry();
        
        // Debug log instance buffer setup
        this.logger.info('Created instanced geometry:', {
            maxInstances: this.maxInstances,
            instancePosition: this.geometry.getAttribute('instancePosition')?.count,
            instanceColor: this.geometry.getAttribute('instanceColor')?.count,
            instanceScale: this.geometry.getAttribute('instanceScale')?.count
        });
        
        this.mesh = new Mesh(this.geometry, this.material);
        this.group.add(this.mesh);
        
        this.setXRMode(platformManager.isXRMode);
        platformManager.on('xrmodechange', (enabled: boolean) => {
            this.setXRMode(enabled);
        });
        
        // Initialize font atlas
        this.initializeFontAtlas();
    }
    
    private async initializeFontAtlas(): Promise<void> {
        try {
            this.logger.info('Starting font atlas generation with params:', {
                fontFamily: 'Arial',
                fontSize: 32,
                textureSize: (this.fontAtlasGenerator as any)['atlasSize'],
                padding: (this.fontAtlasGenerator as any)['padding'],
                spread: (this.fontAtlasGenerator as any)['spread']
            });

            const { texture } = await this.fontAtlasGenerator.generateAtlas(
                'Arial',
                32 // Base font size for SDF
            );
            
            // Configure texture parameters
            texture.minFilter = NearestFilter;
            texture.magFilter = NearestFilter;
            texture.wrapS = ClampToEdgeWrapping;
            texture.wrapT = ClampToEdgeWrapping;
            
            this.fontAtlas = texture;
            if (this.material instanceof ShaderMaterial && this.material.uniforms) {
                this.material.uniforms.fontAtlas.value = texture;
            }
            
            this.logger.info('Font atlas generated successfully:', {
                textureWidth: (texture as any).image?.width,
                textureHeight: (texture as any).image?.height,
                format: (texture as any).format,
                mipmaps: (texture as any).mipmaps?.length || 0
            });
            
            // Update all existing labels
            this.labels.forEach((label, id) => {
                this.updateLabel(id, label.text, label.position, label.color);
            });
        } catch (error) {
            logger.error('Failed to initialize font atlas:', {
                error,
                message: error instanceof Error ? error.message : String(error),
                stack: error instanceof Error ? error.stack : undefined
            });
        }
    }
    
    private createInstancedGeometry(): BufferGeometry {
        const baseGeometry = new PlaneGeometry(1, 1);
        const instancedGeometry = new BufferGeometry();
        
        // Copy attributes from base geometry
        const position = baseGeometry.getAttribute('position');
        const uv = baseGeometry.getAttribute('uv');
        instancedGeometry.setAttribute('position', position);
        instancedGeometry.setAttribute('uv', uv);
        
        const instancePositions = new Float32Array(this.maxInstances * 3);
        const instanceColors = new Float32Array(this.maxInstances * 4);
        const instanceScales = new Float32Array(this.maxInstances);
        
        // Set up instanced attributes
        instancedGeometry.setAttribute(
            'instancePosition',
            new InstancedBufferAttribute(instancePositions, 3)
        );
        instancedGeometry.setAttribute(
            'instanceColor',
            new InstancedBufferAttribute(instanceColors, 4)
        );
        instancedGeometry.setAttribute(
            'instanceScale',
            new InstancedBufferAttribute(instanceScales, 1)
        );
                
        return instancedGeometry;
    }
    
    public updateLabel(id: string, text: string, position: Vector3, color?: Color): void {
        this.logger.debug('Updating label:', {
            id,
            text,
            position,
            color: color ? [(color as any).r, (color as any).g, (color as any).b] : undefined,
            hasAtlas: !!this.fontAtlas
        });
        
        let label = this.labels.get(id);
        
        if (!label) {
            if (this.currentInstanceCount >= this.maxInstances) {
                this.logger.warn(`Maximum instance count (${this.maxInstances}) reached, cannot add more labels`);
                return;
            }
            
            label = {
                id,
                text,
                position: position.clone(),
                scale: 1.0,
                color: color || new Color(this.settings.textColor),
                visible: true
            };
            
            this.logger.debug('Created new label instance:', {
                id,
                instanceIndex: this.currentInstanceCount,
                position,
                color: color ? [(color as any).r, (color as any).g, (color as any).b] : undefined
            });
            
            this.labels.set(id, label);
            this.currentInstanceCount++;
        } else {
            label.text = text;
            label.position.copy(position);
            if (color) label.color = color;
        }
        
        this.updateInstanceAttributes();
    }
    
    private updateInstanceAttributes(): void {
        const positions = (this.geometry.getAttribute('instancePosition') as InstancedBufferAttribute).array as Float32Array;
        const colors = (this.geometry.getAttribute('instanceColor') as InstancedBufferAttribute).array as Float32Array;
        const scales = (this.geometry.getAttribute('instanceScale') as InstancedBufferAttribute).array as Float32Array;

        // Debug log instance updates
        this.logger.debug('Updating instance attributes:', {
            currentInstanceCount: this.currentInstanceCount,
            labelsCount: this.labels.size,
            positionsLength: positions.length,
            colorsLength: colors.length
        });
        
        let index = 0;
        this.labels.forEach(label => {
            if (label.visible) {
                positions[index * 3] = label.position.x;
                positions[index * 3 + 1] = label.position.y;
                positions[index * 3 + 2] = label.position.z;
                
                const colorArray = label.color.toArray();
                colors.set(colorArray, index * 4);
                colors[index * 4 + 3] = 1.0;
                
                scales[index] = label.scale;
                index++;
            }
        });
        
        // Set instance count on the mesh
        (this.mesh as any).instanceCount = this.currentInstanceCount;
        
        // Debug log final state
        this.logger.debug('Instance attributes updated:', {
            instanceCount: (this.mesh as any).instanceCount,
            visibleLabels: index
        });
        
        (this.geometry.getAttribute('instancePosition') as InstancedBufferAttribute).needsUpdate = true;
        (this.geometry.getAttribute('instanceColor') as InstancedBufferAttribute).needsUpdate = true;
        (this.geometry.getAttribute('instanceScale') as InstancedBufferAttribute).needsUpdate = true;
    }
    
    public removeLabel(id: string): void {
        if (this.labels.delete(id)) {
            this.currentInstanceCount--;
            this.updateInstanceAttributes();
        }
    }
    
    public setXRMode(enabled: boolean): void {
        if (enabled) {
            this.group.layers.disable(0);
            this.group.layers.enable(1);
        } else {
            this.group.layers.enable(0);
            this.group.layers.enable(1);
        }
    }
    
    public update(): void {
        this.camera.updateMatrixWorld();
        
        // Update camera position uniform for shader material
        if (this.material instanceof ShaderMaterial && this.material.uniforms) {
            this.material.uniforms.cameraPosition.value.copy(this.camera.position);
        }
        
        // Log state periodically (every ~100 frames)
        if (Math.random() < 0.01) {
            const debugInfo: any = {
                materialType: this.material instanceof ShaderMaterial ? 'ShaderMaterial' : 'MeshBasicMaterial',
                meshInstanceCount: (this.mesh as any).instanceCount,
                totalLabels: this.labels.size,
                visibleLabels: Array.from(this.labels.values()).filter(l => l.visible).length
            };
            
            // Add shader-specific info if available
            if (this.material instanceof ShaderMaterial && this.material.uniforms) {
                debugInfo.materialUniforms = {
                    hasTexture: !!this.material.uniforms.fontAtlas.value
                };
            }
            this.logger.debug('Renderer state:', debugInfo);
        }
    }

    public dispose(): void {
        this.geometry.dispose();
        if (this.material) {
            this.material.dispose();
        }
        if (this.fontAtlas) {
            this.fontAtlas.dispose();
        }
        if (this.group && this.group.parent) {
            this.group.parent.remove(this.group);
        }
    }
}
