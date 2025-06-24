# Phase 2 Performance Optimization Guide

This guide documents the performance optimizations implemented for the Settings Panel and related components.

## Optimizations Implemented

### 1. Virtualization with react-window

**File**: `VirtualizedSettingsGroup.tsx`

The settings panel now uses virtualization to only render visible settings items, dramatically improving performance for long lists.

```tsx
import { VirtualizedSettingsGroup } from './VirtualizedSettingsGroup';

// Usage in settings panel
<VirtualizedSettingsGroup
  title="Node Appearance"
  items={settingItems}
  isExpanded={isExpanded}
  onToggle={toggleGroup}
  // ... other props
/>
```

**Benefits**:
- Only renders visible items (8 at a time)
- Smooth scrolling even with hundreds of settings
- Reduced memory usage

### 2. React.memo for Heavy Components

**Files Modified**:
- `SettingControlComponent.tsx` - Memoized with custom comparison
- `HologramVisualisation.tsx` - Memoized 3D rendering component
- `SettingsPanelRedesignOptimized.tsx` - Fully optimized panel

**Example**:
```tsx
export const SettingControlComponent = React.memo(({ 
  path, 
  settingDef, 
  value, 
  onChange 
}: SettingControlProps) => {
  // Component implementation
}, (prevProps, nextProps) => {
  // Custom comparison for better performance
  return (
    prevProps.path === nextProps.path &&
    prevProps.value === nextProps.value &&
    prevProps.settingDef === nextProps.settingDef
  );
});
```

### 3. Selective Zustand Subscriptions

**File**: `useSelectiveSettingsStore.ts`

Custom hooks for subscribing only to specific settings paths:

```tsx
// Single setting subscription
const nodeColor = useSelectiveSetting<string>('visualisation.nodes.baseColor');

// Multiple settings subscription
const visualSettings = useSelectiveSettings({
  nodeOpacity: 'visualisation.nodes.opacity',
  edgeColor: 'visualisation.edges.color',
  bloomStrength: 'visualisation.bloom.strength'
});

// Batched updates
const { batchedSet } = useSettingSetter();
batchedSet({
  'visualisation.nodes.opacity': 0.8,
  'visualisation.edges.color': '#00ff00',
  'visualisation.bloom.strength': 1.5
});
```

### 4. Code Splitting with Lazy Loading

**File**: `LazySettingsSections.tsx`

Heavy setting sections are now lazy-loaded:

```tsx
const LazyXRSettings = lazy(() => 
  import('./panels/XRPanel').then(m => ({ default: m.XRPanel }))
);

// Usage with Suspense
<Suspense fallback={<LoadingSpinner />}>
  <LazyXRSettings />
</Suspense>
```

### 5. Performance Monitoring

**File**: `performanceMonitor.ts`

Built-in performance monitoring for development:

```tsx
// In component
React.useEffect(() => {
  const endMeasure = performanceMonitor.startMeasure('ComponentName');
  return endMeasure;
});

// View performance report in console
performanceMonitor.logReport();
```

## Integration Steps

### 1. Install Dependencies

```bash
cd client
npm install react-window @types/react-window --save
```

### 2. Replace Settings Panel

Replace the current `SettingsPanelRedesign` import with the optimized version:

```tsx
// Before
import { SettingsPanelRedesign } from './SettingsPanelRedesign';

// After
import { SettingsPanelRedesignOptimized } from './SettingsPanelRedesignOptimized';
```

### 3. Use Selective Subscriptions

In components that read settings, replace direct store access:

```tsx
// Before
const { settings } = useSettingsStore();
const nodeColor = settings.visualisation.nodes.baseColor;

// After
const nodeColor = useSelectiveSetting<string>('visualisation.nodes.baseColor');
```

### 4. Implement Memoization

Add React.memo to components that receive settings as props:

```tsx
export const MyComponent = React.memo(({ setting, onChange }) => {
  // Component implementation
}, (prevProps, nextProps) => {
  return prevProps.setting === nextProps.setting;
});
```

## Performance Metrics

Expected improvements:
- **Initial render time**: 50-70% reduction
- **Re-render frequency**: 80-90% reduction
- **Memory usage**: 40-60% reduction for large setting lists
- **Scroll performance**: Smooth 60fps even with 1000+ settings

## Best Practices

1. **Always use selective subscriptions** for individual settings
2. **Batch related updates** to reduce re-renders
3. **Memoize callbacks** passed to child components
4. **Lazy load** heavy setting sections
5. **Monitor performance** in development

## Example Implementation

See `PerformanceOptimizedExample.tsx` for a complete example demonstrating all optimizations.

## Troubleshooting

1. **Settings not updating**: Ensure you're using the correct path in selective subscriptions
2. **Performance still slow**: Check for unnecessary dependencies in useEffect/useMemo
3. **Virtualization issues**: Ensure consistent item heights in VirtualizedSettingsGroup

## Future Optimizations

1. Implement virtual scrolling for nested setting groups
2. Add request idle callback for non-critical updates
3. Implement setting value caching
4. Add WebWorker for heavy computations