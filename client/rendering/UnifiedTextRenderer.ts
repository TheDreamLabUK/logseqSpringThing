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
    MeshBasicMaterial,
    BufferAttribute
} from 'three';
import { createLogger } from '../core/logger';
import { LabelSettings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
import { SDFFontAtlasGenerator } from './SDFFontAtlasGenerator';
import '../types/three-ext.d';

const logger = createLogger('UnifiedTextRenderer');

// Vertex shader for SDF text rendering with improved billboarding
const vertexShader = `
    // Three.js automatically provides cameraPosition uniform
    
    attribute vec3 position;
    attribute vec2 uv;
    attribute vec3 instancePosition;
    attribute vec4 instanceColor;
    attribute float instanceScale;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    varying float vViewDistance;
    
    void main() {
        vUv = uv;
        vColor = instanceColor;
        vScale = instanceScale;

        // Scale the position first
        vec3 scale = vec3(instanceScale);
        vec3 vertexPosition = position * scale;
        
        // Billboard calculation
        vec3 up = vec3(0.0, 1.0, 0.0);
        vec3 forward = normalize(cameraPosition - instancePosition);
        vec3 right = normalize(cross(up, forward));
        up = normalize(cross(forward, right));
        
        mat4 billboardMatrix = mat4(
            vec4(right, 0.0),
            vec4(up, 0.0),
            vec4(forward, 0.0),
            vec4(0.0, 0.0, 0.0, 1.0)
        );
        
        vertexPosition = (billboardMatrix * vec4(vertexPosition, 1.0)).xyz;
        vertexPosition += instancePosition;
        
        vec4 mvPosition = modelViewMatrix * vec4(vertexPosition, 1.0);
        vViewDistance = -mvPosition.z;  // Distance from camera
        gl_Position = projectionMatrix * mvPosition;
    }
`;

// Fragment shader for SDF text rendering with improved quality
const fragmentShader = `
    precision highp float;
    
    uniform sampler2D fontAtlas;
    uniform float sdfThreshold;
    uniform float sdfSpread;
    uniform vec3 outlineColor;
    uniform float outlineWidth;
    uniform float fadeStart;
    uniform float fadeEnd;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    varying float vViewDistance;
    
    float median(float r, float g, float b) {
        return max(min(r, g), min(max(r, g), b));
    }
    
    void main() {
        vec3 fontSample = texture2D(fontAtlas, vUv).rgb;
        float sigDist = median(fontSample.r, fontSample.g, fontSample.b);
        
        // Dynamic threshold based on distance
        float distanceScale = smoothstep(fadeEnd, fadeStart, vViewDistance);
        float dynamicThreshold = sdfThreshold * (1.0 + (1.0 - distanceScale) * 0.1);
        float dynamicSpread = sdfSpread * (1.0 + (1.0 - distanceScale) * 0.2);
        
        // Improved antialiasing
        float alpha = smoothstep(dynamicThreshold - dynamicSpread, 
                               dynamicThreshold + dynamicSpread, 
                               sigDist);
                               
        float outline = smoothstep(dynamicThreshold - outlineWidth - dynamicSpread,
                                 dynamicThreshold - outlineWidth + dynamicSpread,
                                 sigDist);
        
        // Apply distance-based fade
        alpha *= distanceScale;
        outline *= distanceScale;
        
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
        this.maxInstances = 2000;
        this.currentInstanceCount = 0;
        this.fontAtlas = null;
        
        this.group = new Group();
        this.scene.add(this.group);
        
        this.fontAtlasGenerator = new SDFFontAtlasGenerator(2048, 8, 16);

        this.logger.info('Initializing material with settings:', {
            billboardMode: settings.billboardMode,
            sdfThreshold: 0.45,
            sdfSpread: 0.15,
            outlineColor: settings.textOutlineColor,
            outlineWidth: 0.2,
            fadeStart: 10.0,
            fadeEnd: 100.0,
            depthTest: true
        });
        
        // Initialize the material with error handling
        try {
            this.material = new ShaderMaterial({
                vertexShader,
                fragmentShader,
                // Using WebGL1 compatibility by default
                uniforms: {
                    fontAtlas: { value: null },
                    sdfThreshold: { value: 0.45 },
                    sdfSpread: { value: 0.15 },
                    cameraPosition: { value: this.camera.position },
                    outlineColor: { value: new Color(settings.textOutlineColor) },
                    outlineWidth: { value: 0.2 },
                    fadeStart: { value: 10.0 },
                    fadeEnd: { value: 100.0 }
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
                color: new Color(this.settings.textColor),
                transparent: true
            });
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
        const normal = baseGeometry.getAttribute('normal');
        
        instancedGeometry.setAttribute('position', position);
        instancedGeometry.setAttribute('uv', uv);
        if (normal) instancedGeometry.setAttribute('normal', normal);
        
        // Set up instanced attributes with proper sizes
        const instancePositions = new Float32Array(this.maxInstances * 3); // vec3
        const instanceColors = new Float32Array(this.maxInstances * 4);    // vec4
        const instanceScales = new Float32Array(this.maxInstances);        // float
        
        // Initialize instance attributes with proper itemSize
        instancedGeometry.setAttribute(
            'instancePosition',
            new InstancedBufferAttribute(instancePositions, 3, false)
        );
        instancedGeometry.setAttribute(
            'instanceColor',
            new InstancedBufferAttribute(instanceColors, 4, false)
        );
        instancedGeometry.setAttribute(
            'instanceScale',
            new InstancedBufferAttribute(instanceScales, 1, false)
        );
        
        // Copy index if present
        const index = (baseGeometry as any).index;
        if (index instanceof BufferAttribute) {
            instancedGeometry.setIndex(index);
        }
        
        // Clean up base geometry
        baseGeometry.dispose();
                
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
        if (!this.camera || !this.material) return;
        
        // Update only visible labels
        this.labels.forEach((label, id) => {
            if (this.isLabelVisible(label)) {
                this.updateLabel(id, label.text, label.position, label.color);
            }
        });
        
        // Update camera uniforms
        if (this.material instanceof ShaderMaterial) {
            this.material.uniforms.cameraPosition.value.copy(this.camera.position);
        }
    }

    private isLabelVisible(label: LabelInstance): boolean {
        if (!label.visible) return false;
        
        // Use distance-based culling with the camera's far plane
        const distanceToCamera = label.position.distanceTo(this.camera.position);
        const margin = 5.0;  // Units in world space
        
        // Check if label is within camera's view distance (with margin)
        return distanceToCamera <= (this.camera as any).far + margin;
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
