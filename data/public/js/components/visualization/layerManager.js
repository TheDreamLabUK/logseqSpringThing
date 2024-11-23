// Layer constants for rendering pipeline
export const LAYERS = {
    NORMAL_LAYER: 0,  // Base layer for regular rendering
    BLOOM: 1,        // Layer for node bloom effects
    HOLOGRAM: 2,     // Layer for hologram effects
    EDGE: 3,         // Layer for edge bloom effects
    LABEL: 4         // Layer for labels
};

// Layer groups for optimization
export const LAYER_GROUPS = {
    // Base scene elements (no bloom)
    BASE: [LAYERS.NORMAL_LAYER],
    
    // Elements that should have bloom
    BLOOM: [LAYERS.NORMAL_LAYER, LAYERS.BLOOM],
    
    // Hologram elements
    HOLOGRAM: [LAYERS.NORMAL_LAYER, LAYERS.HOLOGRAM],
    
    // Edge elements
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

// Helper functions for layer management
export const LayerManager = {
    // Enable multiple layers for an object
    enableLayers(object, layers) {
        if (!Array.isArray(layers)) {
            layers = [layers];
        }
        layers.forEach(layer => object.layers.enable(layer));
    },

    // Set object to specific layer group
    setLayerGroup(object, groupName) {
        const layers = LAYER_GROUPS[groupName];
        if (!layers) {
            console.warn(`Unknown layer group: ${groupName}`);
            return;
        }
        
        // Reset layers
        object.layers.mask = 0;
        
        // Enable all layers in group
        layers.forEach(layer => object.layers.enable(layer));
    },

    // Check if object is in layer
    isInLayer(object, layer) {
        return object.layers.test(new THREE.Layers().set(layer));
    },

    // Get all objects in a specific layer
    getObjectsInLayer(scene, layer) {
        const objects = [];
        scene.traverse(object => {
            if (this.isInLayer(object, layer)) {
                objects.push(object);
            }
        });
        return objects;
    }
};
