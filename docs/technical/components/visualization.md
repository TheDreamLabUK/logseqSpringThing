# Visualization System

## Core Components

### Edge Rendering
- Dynamic width calculation based on settings
- Node radius-aware endpoint positioning
- Efficient geometry updates

### Text Rendering
- Distance-based scaling
- Font loading with retry mechanism
- Billboard behavior for readability
- Outline support for visibility

### Scale Management
- Base scale from settings
- AR mode compatibility
- Performance-optimized scaling

## Implementation Details

### Edge Rendering
```typescript
const edgeWidth = Math.max(widthRange[0], Math.min(widthRange[1], baseWidth)) 
    * settings.visualization.edges.scaleFactor;
```

### Text Visibility
- Font loading verification
- Configurable retry mechanism
- Cached font instances
- Distance-based scaling

## Performance Considerations

- GPU-optimized geometry updates
- Batched rendering operations
- Memory-efficient data structures
- AR-specific optimizations

## Configuration

```yaml
visualization:
  edges:
    scaleFactor: 2.0
    widthRange: [0.1, 5.0]
  text:
    fontSize: 16
    outlineWidth: 2
    outlineColor: "#000000"
```