import React, { useMemo, useCallback } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { performanceMonitor } from '@/utils/performanceMonitor';
import { Button } from '@/ui/Button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/Card';

/**
 * Example of a performance-optimized settings component
 * Demonstrates all Phase 2 optimizations:
 * 1. Selective store subscriptions
 * 2. React.memo with proper dependencies
 * 3. Performance monitoring
 * 4. Batched updates
 */
export const PerformanceOptimizedExample = React.memo(() => {
  // Performance monitoring
  React.useEffect(() => {
    const endMeasure = performanceMonitor.startMeasure('PerformanceOptimizedExample');
    return endMeasure;
  });

  // Selective subscriptions - only re-render when these specific settings change
  const nodeColor = useSelectiveSetting<string>('visualisation.nodes.baseColor');
  const bloomEnabled = useSelectiveSetting<boolean>('visualisation.bloom.enabled');
  
  // Multiple selective subscriptions
  const visualSettings = useSelectiveSettings({
    nodeOpacity: 'visualisation.nodes.opacity',
    edgeColor: 'visualisation.edges.color',
    bloomStrength: 'visualisation.bloom.strength'
  });

  // Optimized setter with batching
  const { set, batchedSet } = useSettingSetter();

  // Memoized callbacks to prevent unnecessary re-renders
  const handleColorChange = useCallback((color: string) => {
    set('visualisation.nodes.baseColor', color);
  }, [set]);

  const handleBloomToggle = useCallback(() => {
    set('visualisation.bloom.enabled', !bloomEnabled);
  }, [set, bloomEnabled]);

  const handleBatchUpdate = useCallback(() => {
    // Batch multiple updates together for better performance
    batchedSet({
      'visualisation.nodes.opacity': 0.8,
      'visualisation.edges.color': '#00ff00',
      'visualisation.bloom.strength': 1.5,
      'visualisation.bloom.radius': 0.5
    });
  }, [batchedSet]);

  // Memoized heavy computations
  const colorPresets = useMemo(() => [
    { label: 'Cyan', value: '#00ffff' },
    { label: 'Purple', value: '#ff00ff' },
    { label: 'Green', value: '#00ff00' },
    { label: 'Orange', value: '#ff8800' }
  ], []);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Performance Optimized Settings</CardTitle>
        <CardDescription>
          This component demonstrates all Phase 2 performance optimizations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Values Display */}
        <div className="p-4 bg-muted rounded-lg space-y-2">
          <h4 className="font-medium">Current Values (Selective Subscriptions)</h4>
          <div className="text-sm space-y-1">
            <div>Node Color: <span className="font-mono">{nodeColor}</span></div>
            <div>Bloom Enabled: <span className="font-mono">{String(bloomEnabled)}</span></div>
            <div>Node Opacity: <span className="font-mono">{visualSettings.nodeOpacity}</span></div>
            <div>Edge Color: <span className="font-mono">{visualSettings.edgeColor}</span></div>
            <div>Bloom Strength: <span className="font-mono">{visualSettings.bloomStrength}</span></div>
          </div>
        </div>

        {/* Color Presets */}
        <div className="space-y-2">
          <h4 className="font-medium">Quick Color Presets</h4>
          <div className="flex gap-2">
            {colorPresets.map(preset => (
              <Button
                key={preset.value}
                variant="outline"
                size="sm"
                onClick={() => handleColorChange(preset.value)}
                style={{ borderColor: preset.value }}
              >
                {preset.label}
              </Button>
            ))}
          </div>
        </div>

        {/* Toggle Controls */}
        <div className="space-y-2">
          <h4 className="font-medium">Feature Toggles</h4>
          <Button
            variant={bloomEnabled ? 'default' : 'outline'}
            onClick={handleBloomToggle}
          >
            {bloomEnabled ? 'Disable' : 'Enable'} Bloom Effect
          </Button>
        </div>

        {/* Batch Update Demo */}
        <div className="space-y-2">
          <h4 className="font-medium">Batch Updates</h4>
          <Button
            variant="secondary"
            onClick={handleBatchUpdate}
          >
            Apply Preset Configuration (4 settings at once)
          </Button>
        </div>

        {/* Performance Info */}
        <div className="text-xs text-muted-foreground space-y-1">
          <p>• Only re-renders when subscribed settings change</p>
          <p>• Uses memoized callbacks to prevent child re-renders</p>
          <p>• Batches multiple updates for better performance</p>
          <p>• Monitors render performance in development</p>
        </div>
      </CardContent>
    </Card>
  );
}, () => {
  // Always use the same instance - no props to compare
  return true;
});

PerformanceOptimizedExample.displayName = 'PerformanceOptimizedExample';

/**
 * Example of a child component that won't re-render unnecessarily
 */
const OptimizedChild = React.memo(({ 
  value, 
  onChange 
}: { 
  value: string; 
  onChange: (value: string) => void;
}) => {
  React.useEffect(() => {
    performanceMonitor.startMeasure('OptimizedChild')();
  });

  return (
    <div className="p-2 border rounded">
      <input
        type="color"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-8"
      />
    </div>
  );
}, (prevProps, nextProps) => {
  return (
    prevProps.value === nextProps.value &&
    prevProps.onChange === nextProps.onChange
  );
});

OptimizedChild.displayName = 'OptimizedChild';