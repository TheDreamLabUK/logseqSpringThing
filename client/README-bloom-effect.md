# Bloom Effect Implementation Guide

## Overview

The bloom effect in this project is implemented using Three.js's `UnrealBloomPass` and a layer-based approach to selectively apply bloom to specific objects. This document explains how the bloom effect works, the improvements made, and best practices for using it.

## Implementation Details

### Layer-Based Approach

The bloom effect uses a layer-based approach to selectively apply the effect:

- Layer 0: Default layer for all objects
- Layer 1: Bloom layer for objects that should have the bloom effect

Objects that should have the bloom effect need to have Layer 1 enabled:

```typescript
// Enable bloom layer on an object
object.layers.enable(1);
```

### Bloom Settings

The bloom effect has several configurable settings:

- `enabled`: Whether the bloom effect is enabled
- `strength`: Overall strength of the bloom effect (0-5)
- `radius`: Radius of the bloom effect (0-5)
- `threshold`: Brightness threshold for the bloom effect (0-1)
- `edgeBloomStrength`: Strength of the bloom effect on edges
- `nodeBloomStrength`: Strength of the bloom effect on nodes
- `environmentBloomStrength`: Strength of the bloom effect on the environment

### Performance Considerations

The bloom effect can be performance-intensive. The implementation includes several optimizations:

1. **Adaptive Strength**: When FPS drops below 20, the bloom strength is reduced to maintain performance
2. **Fallback Rendering**: If there's an error with the bloom effect, it falls back to standard rendering
3. **Extreme Low FPS Handling**: At extremely low FPS (<15), bloom is temporarily disabled

## Recent Improvements

The following improvements have been made to the bloom effect:

1. **Consistent Rendering**: Removed conditional rendering based on frame budget to prevent bloom from flashing on/off
2. **Graceful Degradation**: Instead of completely disabling bloom at low FPS, it now gradually reduces intensity
3. **Error Handling**: Added better error handling to prevent rendering failures
4. **Settings Update**: Improved how settings are updated to prevent visual glitches
5. **Layer Management**: Better management of bloom layers with explicit constants

## Best Practices

### Adding Bloom to Objects

To add bloom to an object:

```typescript
// Create your mesh
const mesh = new Mesh(geometry, material);

// Enable bloom layer
mesh.layers.enable(1);

// Add to scene
scene.add(mesh);
```

### Adjusting Bloom Intensity

Different object types should use different bloom intensities:

- **Edges**: Use `edgeBloomStrength` (recommended: 1.0-2.0)
- **Nodes**: Use `nodeBloomStrength` (recommended: 2.0-3.0)
- **Environment**: Use `environmentBloomStrength` (recommended: 0.5-1.5)

### Performance Optimization

For better performance:

1. Be selective about which objects get the bloom effect
2. Use lower bloom radius values (1.0-2.0) for better performance
3. Consider disabling bloom in XR mode or on low-end devices

## Troubleshooting

### Bloom Effect Flashing On/Off

If the bloom effect is flashing on and off:

1. Check if FPS is consistently low (below 15)
2. Verify that objects have the correct layer enabled
3. Reduce the overall bloom strength and radius

### No Bloom Effect

If objects aren't showing the bloom effect:

1. Verify the object has Layer 1 enabled
2. Check if bloom is enabled in settings
3. Ensure the object's material has appropriate properties (emissive materials work best with bloom)

### Performance Issues

If bloom is causing performance issues:

1. Reduce the number of objects with bloom
2. Lower the bloom radius and strength
3. Increase the bloom threshold to only affect brighter parts of the scene