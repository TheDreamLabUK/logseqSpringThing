# Scale Factors Architecture

## Core Principles

1. AR Space Matching
   - Only xrSessionManager.ts should handle room-scale adjustments
   - All other scaling should be relative to the room scale when in AR mode
   - Room scale is the source of truth for real-world space matching

2. Visualization Hierarchy
   - Node scaling is intentionally designed for visual importance
   - Hologram instance scaling is designed for scene composition
   - These visual scales should not affect spatial relationships

## Permitted Scale Usage

### AR/XR Layer (Space Matching)
- Location: xrSessionManager.ts
- Purpose: Match virtual space to real world space
- Implementation: Uses roomScale setting
- Scope: Applies to entire AR group/camera rig

### Node Visualization Layer
- Location: EnhancedNodeManager.ts
- Purpose: Visual representation of node importance
- Implementation: Base size + importance-based scaling
- Scope: Individual node instances

### Scene Elements Layer
- Location: HologramManager.ts
- Purpose: Visual composition of holographic elements
- Implementation: Instance-based scaling for geometric elements
- Scope: Individual hologram instances

## Restricted Areas

The following areas should NOT implement their own scaling:

1. Edge Rendering
   - Should inherit from node scaling
   - No independent scale factors

2. Physics/Graph Layout
   - Should operate in normalized space
   - Scaling should only be applied at visualization layer


## Implementation Guidelines

1. Scale Inheritance
   - All child objects should inherit parent scale
   - No competing scale factors in hierarchy

2. AR Mode Considerations
   - All visual scaling must respect room scale in AR mode
   - Visual effects should not interfere with spatial matching

3. Configuration
   - Scale factors should be centralized in settings
   - Clear separation between space-matching and visual scales

## Validation Points

When reviewing scale factor usage:

1. Check if scale factor is necessary
   - Is it for space matching? → Should be in xrSessionManager.ts
   - Is it for visualization? → Should be in appropriate visualization layer
   - Neither? → Should probably be removed

2. Verify scale inheritance
   - Is it respecting parent scales?
   - Is it conflicting with other scale factors?

3. Test AR compatibility
   - Does it interfere with space matching?
   - Does it maintain proper relationships in AR mode?

## Next Steps

1. Audit Existing Code
   - Remove any scale factors outside permitted areas
   - Ensure proper scale inheritance
   - Validate AR mode compatibility

2. Documentation
   - Update code comments to clarify scale factor usage
   - Document scale inheritance in component relationships

3. Testing
   - Add tests for scale inheritance
   - Verify AR space matching with different configurations