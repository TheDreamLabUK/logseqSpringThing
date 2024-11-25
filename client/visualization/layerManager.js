import * as THREE from 'three';

// Layer constants for rendering pipeline
export const LAYERS = {
    NORMAL_LAYER: 0,  // Base layer for regular rendering
    BLOOM: 1,        // Layer for node bloom effects
    HOLOGRAM: 2,     // Layer for hologram effects
    EDGE: 3,         // Layer for edge bloom effects
    LABEL: 4         // Layer for labels
};

// Simplified layer groups - everything visible on normal layer
export const LAYER_GROUPS = {
    // Base scene elements
    BASE: [LAYERS.NORMAL_LAYER],
    
    // Nodes
    BLOOM: [LAYERS.NORMAL_LAYER],
    
    // Hologram elements
    HOLOGRAM: [LAYERS.NORMAL_LAYER],
    
    // Edge elements
    EDGE: [LAYERS.NORMAL_LAYER],
    
    // Label elements
    LABEL: [LAYERS.NORMAL_LAYER]
};

// Simplified material presets for basic rendering
const MATERIAL_PRESETS = {
    BLOOM: {
        transparent: true,
        opacity: 1.0,
        blending: THREE.NormalBlending,
        depthWrite: true,
        toneMapped: true
    },
    HOLOGRAM: {
        transparent: true,
        opacity: 0.8,
        blending: THREE.NormalBlending,
        depthWrite: true,
        toneMapped: true
    },
    EDGE: {
        transparent: true,
        opacity: 0.8,
        blending: THREE.NormalBlending,
        depthWrite: true,
        toneMapped: true
    }
};

// Enhanced LayerManager with simplified rendering
export const LayerManager = {
    // Enable multiple layers for an object
    enableLayers(object, layers) {
        if (!object || !object.layers) {
            console.error('Invalid object provided to enableLayers');
            return;
        }

        // Always enable normal layer
        object.layers.set(LAYERS.NORMAL_LAYER);
    },

    // Set object to specific layer group with basic material settings
    setLayerGroup(object, groupName) {
        if (!object || !object.layers) {
            console.error('Invalid object provided to setLayerGroup');
            return;
        }

        // Always set to normal layer for visibility
        object.layers.set(LAYERS.NORMAL_LAYER);

        // Apply basic material presets if object has material
        if (object.material && MATERIAL_PRESETS[groupName]) {
            // Clone material to avoid affecting other objects
            if (!object.material._isCloned) {
                object.material = object.material.clone();
                object.material._isCloned = true;
            }
            
            // Apply basic preset properties
            Object.assign(object.material, MATERIAL_PRESETS[groupName]);
            
            // Ensure material is visible and properly rendered
            object.material.needsUpdate = true;
        }
    },

    // Check if object is in layer
    isInLayer(object, layer) {
        if (!object || !object.layers || typeof layer !== 'number') {
            return false;
        }
        return object.layers.test(new THREE.Layers().set(layer));
    },

    // Get all objects in a specific layer
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

    // Reset object to base layer with standard material
    resetToBaseLayer(object) {
        if (!object || !object.layers) return;
        
        object.layers.set(LAYERS.NORMAL_LAYER);
        
        if (object.material && object.material._isCloned) {
            object.material.dispose();
            object.material = new THREE.MeshStandardMaterial({
                color: object.material.color,
                transparent: true,
                opacity: 1.0,
                toneMapped: true
            });
            object.material._isCloned = false;
        }
    },

    // Create a standard material
    createStandardMaterial(color) {
        return new THREE.MeshStandardMaterial({
            color: color,
            transparent: true,
            opacity: 1.0,
            toneMapped: true,
            depthWrite: true,
            blending: THREE.NormalBlending
        });
    }
};
