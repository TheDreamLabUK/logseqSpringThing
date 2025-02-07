# Metadata Text Visibility Fix Plan

## Issues Identified

1. Settings Access
- MetadataVisualizer uses incorrect settings property paths
- Default metadata visualization is disabled
- Billboard mode property name mismatch

2. Text Scaling
- Text meshes are scaled too small (0.1)
- Font size settings not properly utilized

3. Font Loading
- No font loading verification
- Missing retry mechanism
- Silent failure handling

## Implementation Plan

### 1. Fix Settings Access

Update MetadataVisualizer.ts to use correct settings paths:
- Replace `settings.labelSize` with `settings.labels.desktopFontSize`
- Replace `settings.labelColor` with `settings.labels.textColor`
- Replace `settings.labels.billboard_mode` with `settings.labels.billboardMode`

### 2. Improve Text Visibility

Update text creation in MetadataVisualizer.ts:
- Increase base text scale from 0.1 to appropriate size based on desktopFontSize
- Add text outline using textOutlineColor and textOutlineWidth from settings
- Ensure proper text positioning above nodes
- Add distance-based scaling for better readability

### 3. Enhance Font Loading

Add robust font loading:
- Add font existence verification
- Implement retry mechanism with configurable attempts
- Add proper error handling with user feedback
- Cache font after successful load

### 4. Testing Steps

1. Verify settings are properly accessed
2. Confirm text is visible and properly scaled
3. Test font loading reliability
4. Verify text remains readable at different distances
5. Check billboard behavior works correctly

## Migration Notes

- Changes are backward compatible
- No database updates required
- No API changes needed