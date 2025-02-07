# Scale Factors Review

## Overview
This document outlines the usage of scale factors across the codebase and recommendations for maintaining AR space consistency.

## Current Scale Factor Usage

### AR/XR (Acceptable)
- `xrSessionManager.ts`: Uses roomScale for AR mode
  - Applied to arGroup and cameraRig
  - Controlled by settings
- `config/mod.rs`: room_scale setting (default: 0.1)

### Visualization (Needs Review)
1. MetadataVisualizer.ts
   - Uses scale for metadata importance visualization
   - Could affect AR space perception

2. EnhancedNodeManager.ts
   - Base scale from settings
   - Velocity-based scaling
   - Node importance scaling
   - Recommendation: Consider removing dynamic scaling in AR mode

3. HologramManager.ts
   - Multiple geometric scales:
     - buckminster_scale
     - geodesic_scale
     - triangle_sphere_scale
   - Recommendation: Ensure these are relative to room_scale in AR mode

### Physics/Graph
- graph_service.rs: Uses scale for velocity normalization
- Recommendation: Verify this doesn't affect spatial relationships in AR

## Recommendations

1. AR Mode Consistency
   - All scales should be relative to room_scale in AR mode
   - Consider adding an AR mode check before applying non-essential scaling

2. Configuration Consolidation
   - Review necessity of multiple scale settings
   - Consider consolidating visualization scales
   - Add validation to prevent conflicting scale factors

3. Implementation Tasks
   - Add AR mode checks before applying visualization scaling
   - Normalize all scales relative to room_scale in AR mode
   - Add scale factor validation in settings

## Next Steps
1. Review and validate all scale factor usage
2. Implement AR mode checks
3. Consolidate configuration settings
4. Add scale factor validation
5. Test AR space matching with different scale configurations