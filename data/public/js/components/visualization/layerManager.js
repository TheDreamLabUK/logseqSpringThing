import * as THREE from 'three';

// Layer constants for rendering pipeline
export const LAYERS = {
    NORMAL_LAYER: 0,  // Base layer for regular rendering
    BLOOM: 1,        // Layer for node bloom effects
    HOLOGRAM: 2,     // Layer for hologram effects
    EDGE: 3,         // Layer for edge bloom effects
    LABEL: 4         // Layer for labels
};

// Layer groups with specific rendering requirements
export const LAYER_GROUPS = {
    // Base scene elements (no bloom)
    BASE: [LAYERS.NORMAL_LAYER],
    
    // Elements that should have bloom
    BLOOM: [LAYERS.NORMAL_LAYER, LAYERS.BLOOM],
    
    // Hologram elements with enhanced bloom
    HOLOGRAM: [LAYERS.NORMAL_LAYER, LAYERS.HOLOGRAM],
    
    // Edge elements with subtle bloom
    EDGE: [LAYERS.NORMAL_LAYER, LAYERS.EDGE],
    
    // Label elements (should be visible in all layers)
    LABEL: [
        LAYERS.NORMAL_LAYER,
        LAYERS.BLOOM,
        LAYERS.HOLOGRAM,
        LAYERS.EDGE,
        LAYERS.LABEL
    ]
};

// Material presets for different layer groups
const MATERIAL_PRESETS = {
    BLOOM: {
        emissiveIntensity: 1.0,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false
    },
    HOLOGRAM: {
        emissiveIntensity: 1.5,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false
    },
    EDGE: {
        emissiveIntensity: 0.8,
        transparent: true,
        opacity: 0.6,
        blending: THREE.NormalBlending,
        depthWrite: true,
        toneMapped: false
    }
};

// Enhanced LayerManager with better type checking and error handling
export const LayerManager = {
    // Enable multiple layers for an object
    enableLayers(object, layers) {
        if (!object || !object.layers) {
            console.error('Invalid object provided to enableLayers');
            return;
        }

        if (!Array.isArray(layers)) {
            layers = [layers];
        }

        // Reset layers before enabling new ones
        object.layers.mask = 0;
        
        layers.forEach(layer => {
            if (typeof layer === 'number' && layer >= 0) {
                object.layers.enable(layer);
            } else {
                console.warn(`Invalid layer value: ${layer}`);
            }
        });
    },

    // Set object to specific layer group with material optimization
    setLayerGroup(object, groupName) {
        if (!object || !object.layers) {
            console.error('Invalid object provided to setLayerGroup');
            return;
        }

        const layers = LAYER_GROUPS[groupName];
        if (!layers) {
            console.warn(`Unknown layer group: ${groupName}`);
            return;
        }
        
        // Reset layers
        object.layers.mask = 0;
        
        // Enable all layers in group
        layers.forEach(layer => object.layers.enable(layer));

        // Apply material presets if object has material
        if (object.material && MATERIAL_PRESETS[groupName]) {
            // Clone material to avoid affecting other objects
            if (!object.material._isCloned) {
                object.material = object.material.clone();
                object.material._isCloned = true;
            }
            
            // Apply preset properties
            Object.assign(object.material, MATERIAL_PRESETS[groupName]);
            
            // Special handling for emissive color
            if (object.material.color && object.material.emissive) {
                object.material.emissive.copy(object.material.color);
            }
        }
    },

    // Check if object is in layer with proper type checking
    isInLayer(object, layer) {
        if (!object || !object.layers || typeof layer !== 'number') {
            return false;
        }
        return object.layers.test(new THREE.Layers().set(layer));
    },

    // Get all objects in a specific layer with filtering options
    getObjectsInLayer(scene, layer, options = {}) {
        if (!scene || typeof layer !== 'number') {
            console.error('Invalid parameters provided to getObjectsInLayer');
            return [];
        }

        const objects = [];
        const {
            includeInvisible = false,
            includeHelpers = false
        } = options;

        scene.traverse(object => {
            if (this.isInLayer(object, layer)) {
                if (!includeInvisible && !object.visible) return;
                if (!includeHelpers && object.isHelper) return;
                objects.push(object);
            }
        });
        return objects;
    },

    // Get objects that should receive bloom
    getBloomObjects(scene) {
        return this.getObjectsInLayer(scene, LAYERS.BLOOM).concat(
            this.getObjectsInLayer(scene, LAYERS.HOLOGRAM),
            this.getObjectsInLayer(scene, LAYERS.EDGE)
        );
    },

    // Optimize material for bloom rendering
    optimizeForBloom(object, intensity = 1.0) {
        if (!object || !object.material) return;

        // Clone material to avoid affecting other objects
        if (!object.material._isCloned) {
            object.material = object.material.clone();
            object.material._isCloned = true;
        }

        // Apply bloom-specific optimizations
        object.material.toneMapped = false;
        object.material.transparent = true;
        object.material.blending = THREE.AdditiveBlending;
        object.material.depthWrite = false;
        
        if (object.material.emissive) {
            object.material.emissive.copy(object.material.color || new THREE.Color(1, 1, 1));
            object.material.emissiveIntensity = intensity;
        }
    },

    // Reset object to base layer
    resetToBaseLayer(object) {
        if (!object || !object.layers) return;
        
        object.layers.set(LAYERS.NORMAL_LAYER);
        
        if (object.material && object.material._isCloned) {
            object.material.dispose();
            object.material = new THREE.MeshStandardMaterial({
                color: object.material.color,
                transparent: false,
                toneMapped: true,
                emissiveIntensity: 0
            });
            object.material._isCloned = false;
        }
    },

    // Check if object should receive bloom
    shouldReceiveBloom(object) {
        return this.isInLayer(object, LAYERS.BLOOM) ||
               this.isInLayer(object, LAYERS.HOLOGRAM) ||
               this.isInLayer(object, LAYERS.EDGE);
    },

    // Create a bloom-optimized material
    createBloomMaterial(color, intensity = 1.0) {
        return new THREE.MeshPhysicalMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: intensity,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            toneMapped: false
        });
    }
};
