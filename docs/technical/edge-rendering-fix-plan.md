# Edge Rendering Fix Plan

## Current Issues
1. Edges don't reach node surfaces
2. Edge widths are too thin
3. Edge positions don't update properly with node movement

## Root Causes
1. Edge endpoints don't account for node radii when calculating positions
2. Hardcoded edge width scaling (0.1) makes edges too thin
3. Edge position calculation needs to account for node sizes

## Implementation Plan

### 1. Update Edge Width Calculation
In EdgeManager.ts, modify the edge width calculation to:
- Remove hardcoded 0.1 scale factor
- Use the edge.scaleFactor from settings (default 2.0)
- Properly scale width based on settings ranges

```typescript
// Before
const edgeWidth = Math.max(widthRange[0], Math.min(widthRange[1], baseWidth)) * 0.1;

// After
const edgeWidth = Math.max(widthRange[0], Math.min(widthRange[1], baseWidth)) * settings.visualization.edges.scaleFactor;
```

### 2. Account for Node Sizes
Modify edge endpoint calculations to account for node radii:

```typescript
// Calculate node radii based on settings
const sourceRadius = settings.visualization.nodes.baseSize * 0.5;
const targetRadius = settings.visualization.nodes.baseSize * 0.5;

// Adjust start and end positions by node radii
const direction = endPos.clone().sub(startPos).normalize();
const adjustedStart = startPos.clone().add(direction.clone().multiplyScalar(sourceRadius));
const adjustedEnd = endPos.clone().sub(direction.clone().multiplyScalar(targetRadius));

// Use adjusted positions for edge placement
const position = adjustedStart.clone().add(adjustedEnd).multiplyScalar(0.5);
const length = adjustedStart.distanceTo(adjustedEnd);
```

### 3. Improve Edge Updates
Ensure edges update properly when nodes move:
- Subscribe to node position updates in VisualizationController
- Trigger edge updates when node positions change
- Use proper matrix updates for instanced mesh

## Testing Plan
1. Verify edges connect to node surfaces
2. Check edge widths match settings
3. Confirm edges update with node movement
4. Test with different node sizes
5. Validate performance with many edges

## Expected Results
- Edges should visually connect to node surfaces
- Edge widths should be proportional to settings
- Edges should maintain connections as nodes move
- Performance should remain smooth with many edges