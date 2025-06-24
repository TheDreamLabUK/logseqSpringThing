import React, { useState, useMemo, useCallback, lazy, Suspense } from 'react';
import Tabs from '@/ui/Tabs';
import { Card } from '@/ui/Card';
import { Button } from '@/ui/Button';
import { SearchInput } from '@/ui/SearchInput';
import {
  Eye,
  Settings,
  Smartphone,
  Info,
  ChevronDown,
  ChevronUp,
  Search,
  Keyboard
} from 'lucide-react';
import { useSettingsStore } from '@/store/settingsStore';
import { settingsUIDefinition } from '../../config/settingsUIDefinition';
import { cn } from '@/utils/cn';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { LoadingSpinner } from '@/ui/LoadingSpinner';
import { SkeletonSetting } from '@/ui/LoadingSkeleton';
import { useErrorHandler } from '@/hooks/useErrorHandler';
import { useToast } from '@/ui/useToast';
import { VirtualizedSettingsGroup } from '../VirtualizedSettingsGroup';
import { performanceMonitor } from '@/utils/performanceMonitor';

// Lazy load heavy modals
const KeyboardShortcutsModal = lazy(() => 
  import('@/components/KeyboardShortcutsModal').then(m => ({ default: m.KeyboardShortcutsModal }))
);

interface SettingItem {
  key: string;
  path: string;
  definition: any;
  isPowerUser?: boolean;
}

interface SettingGroup {
  title: string;
  description?: string;
  items: SettingItem[];
  isPowerUser?: boolean;
}

interface SettingsPanelRedesignOptimizedProps {
  toggleLowerRightPaneDock: () => void;
  isLowerRightPaneDocked: boolean;
}

// Memoized search component
const SearchSection = React.memo(({ 
  searchQuery, 
  onSearchChange,
  onShowShortcuts 
}: {
  searchQuery: string;
  onSearchChange: (value: string) => void;
  onShowShortcuts: () => void;
}) => (
  <div className="px-4 pb-3">
    <SearchInput
      value={searchQuery}
      onChange={onSearchChange}
      placeholder="Search settings..."
      className="w-full"
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          onSearchChange('');
        }
      }}
    />
    <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
      <span>Press <kbd className="px-1 py-0.5 bg-muted rounded text-xs">Ctrl+/</kbd> to search</span>
      <button
        onClick={onShowShortcuts}
        className="flex items-center gap-1 hover:text-foreground transition-colors"
      >
        <Keyboard className="h-3 w-3" />
        <span>View shortcuts</span>
      </button>
    </div>
  </div>
));

SearchSection.displayName = 'SearchSection';

export const SettingsPanelRedesignOptimized = React.memo(({ 
  toggleLowerRightPaneDock, 
  isLowerRightPaneDocked 
}: SettingsPanelRedesignOptimizedProps) => {
  // Use selective subscriptions for better performance
  const isPowerUser = useSettingsStore(state => state.isPowerUser);
  const settings = useSettingsStore(state => state.settings);
  
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['Node Appearance']));
  const [savedNotification, setSavedNotification] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [loadingSettings, setLoadingSettings] = useState<Set<string>>(new Set());
  const [isInitializing, setIsInitializing] = useState(true);
  const { handleError } = useErrorHandler();
  const { toast } = useToast();

  // Performance monitoring
  React.useEffect(() => {
    const endMeasure = performanceMonitor.startMeasure('SettingsPanelRedesignOptimized');
    return endMeasure;
  });

  // Organize settings into logical groups with better structure
  const settingsStructure = useMemo(() => ({
    appearance: {
      label: 'Appearance',
      icon: <Eye className="h-4 w-4" />,
      groups: [
        {
          title: 'Node Appearance',
          description: 'Customize how nodes look',
          items: [
            { key: 'baseColor', path: 'visualisation.nodes.baseColor', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.baseColor },
            { key: 'opacity', path: 'visualisation.nodes.opacity', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.opacity },
            { key: 'metalness', path: 'visualisation.nodes.metalness', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.metalness },
            { key: 'roughness', path: 'visualisation.nodes.roughness', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.roughness },
            { key: 'nodeSize', path: 'visualisation.nodes.nodeSize', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.nodeSize },
            { key: 'enableInstancing', path: 'visualisation.nodes.enableInstancing', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.enableInstancing },
            { key: 'enableHologram', path: 'visualisation.nodes.enableHologram', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.enableHologram },
            { key: 'enableMetadataShape', path: 'visualisation.nodes.enableMetadataShape', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.enableMetadataShape },
            { key: 'enableMetadataVisualisation', path: 'visualisation.nodes.enableMetadataVisualisation', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.enableMetadataVisualisation },
          ]
        },
        {
          title: 'Edge Appearance',
          description: 'Customize connection lines',
          items: [
            { key: 'color', path: 'visualisation.edges.color', definition: settingsUIDefinition.visualisation.subsections.edges.settings.color },
            { key: 'opacity', path: 'visualisation.edges.opacity', definition: settingsUIDefinition.visualisation.subsections.edges.settings.opacity },
            { key: 'baseWidth', path: 'visualisation.edges.baseWidth', definition: settingsUIDefinition.visualisation.subsections.edges.settings.baseWidth },
            { key: 'enableArrows', path: 'visualisation.edges.enableArrows', definition: settingsUIDefinition.visualisation.subsections.edges.settings.enableArrows },
            { key: 'arrowSize', path: 'visualisation.edges.arrowSize', definition: settingsUIDefinition.visualisation.subsections.edges.settings.arrowSize },
            { key: 'widthRange', path: 'visualisation.edges.widthRange', definition: settingsUIDefinition.visualisation.subsections.edges.settings.widthRange },
            { key: 'enableFlowEffect', path: 'visualisation.edges.enableFlowEffect', definition: settingsUIDefinition.visualisation.subsections.edges.settings.enableFlowEffect },
            { key: 'flowSpeed', path: 'visualisation.edges.flowSpeed', definition: settingsUIDefinition.visualisation.subsections.edges.settings.flowSpeed },
            { key: 'flowIntensity', path: 'visualisation.edges.flowIntensity', definition: settingsUIDefinition.visualisation.subsections.edges.settings.flowIntensity },
            { key: 'glowStrength', path: 'visualisation.edges.glowStrength', definition: settingsUIDefinition.visualisation.subsections.edges.settings.glowStrength },
            { key: 'distanceIntensity', path: 'visualisation.edges.distanceIntensity', definition: settingsUIDefinition.visualisation.subsections.edges.settings.distanceIntensity },
            { key: 'useGradient', path: 'visualisation.edges.useGradient', definition: settingsUIDefinition.visualisation.subsections.edges.settings.useGradient },
            { key: 'gradientColors', path: 'visualisation.edges.gradientColors', definition: settingsUIDefinition.visualisation.subsections.edges.settings.gradientColors },
          ]
        },
        {
          title: 'Labels',
          description: 'Text display settings',
          items: [
            { key: 'enableLabels', path: settingsUIDefinition.visualisation.subsections.labels.settings.enableLabels.path, definition: settingsUIDefinition.visualisation.subsections.labels.settings.enableLabels },
            { key: 'desktopFontSize', path: 'visualisation.labels.desktopFontSize', definition: settingsUIDefinition.visualisation.subsections.labels.settings.desktopFontSize },
            { key: 'textColor', path: 'visualisation.labels.textColor', definition: settingsUIDefinition.visualisation.subsections.labels.settings.textColor },
            { key: 'textOutlineColor', path: 'visualisation.labels.textOutlineColor', definition: settingsUIDefinition.visualisation.subsections.labels.settings.textOutlineColor },
            { key: 'textOutlineWidth', path: 'visualisation.labels.textOutlineWidth', definition: settingsUIDefinition.visualisation.subsections.labels.settings.textOutlineWidth },
            { key: 'textResolution', path: 'visualisation.labels.textResolution', definition: settingsUIDefinition.visualisation.subsections.labels.settings.textResolution },
            { key: 'textPadding', path: 'visualisation.labels.textPadding', definition: settingsUIDefinition.visualisation.subsections.labels.settings.textPadding },
            { key: 'billboardMode', path: 'visualisation.labels.billboardMode', definition: settingsUIDefinition.visualisation.subsections.labels.settings.billboardMode },
          ]
        },
        {
          title: 'Visual Effects',
          description: 'Bloom and glow effects',
          items: [
            { key: 'bloomEnabled', path: 'visualisation.bloom.enabled', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.enabled },
            { key: 'bloomStrength', path: 'visualisation.bloom.strength', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.strength },
            { key: 'bloomRadius', path: 'visualisation.bloom.radius', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.radius },
            { key: 'edgeBloomStrength', path: 'visualisation.bloom.edgeBloomStrength', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.edgeBloomStrength },
            { key: 'environmentBloomStrength', path: 'visualisation.bloom.environmentBloomStrength', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.environmentBloomStrength },
            { key: 'nodeBloomStrength', path: 'visualisation.bloom.nodeBloomStrength', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.nodeBloomStrength },
            { key: 'threshold', path: 'visualisation.bloom.threshold', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.threshold },
          ]
        }
      ]
    },
    performance: {
      label: 'Performance',
      icon: <Settings className="h-4 w-4" />,
      groups: [
        {
          title: 'Rendering Quality',
          description: 'Balance quality and performance',
          items: [
            { key: 'nodeQuality', path: 'visualisation.nodes.quality', definition: settingsUIDefinition.visualisation.subsections.nodes.settings.quality },
            { key: 'edgeQuality', path: 'visualisation.edges.quality', definition: settingsUIDefinition.visualisation.subsections.edges.settings.quality },
            { key: 'enableAntialiasing', path: 'visualisation.rendering.enableAntialiasing', definition: settingsUIDefinition.visualisation.subsections.rendering.settings.enableAntialiasing },
          ]
        },
        {
          title: 'Physics Engine',
          description: 'Node movement behavior',
          items: [
            { key: 'physicsEnabled', path: 'visualisation.physics.enabled', definition: settingsUIDefinition.visualisation.subsections.physics.settings.enabled },
            { key: 'iterations', path: 'visualisation.physics.iterations', definition: settingsUIDefinition.visualisation.subsections.physics.settings.iterations },
            { key: 'damping', path: 'visualisation.physics.damping', definition: settingsUIDefinition.visualisation.subsections.physics.settings.damping },
            { key: 'boundsSize', path: 'visualisation.physics.boundsSize', definition: settingsUIDefinition.visualisation.subsections.physics.settings.boundsSize },
            { key: 'collisionRadius', path: 'visualisation.physics.collisionRadius', definition: settingsUIDefinition.visualisation.subsections.physics.settings.collisionRadius },
            { key: 'enableBounds', path: 'visualisation.physics.enableBounds', definition: settingsUIDefinition.visualisation.subsections.physics.settings.enableBounds },
            { key: 'maxVelocity', path: 'visualisation.physics.maxVelocity', definition: settingsUIDefinition.visualisation.subsections.physics.settings.maxVelocity },
            { key: 'repulsionDistance', path: 'visualisation.physics.repulsionDistance', definition: settingsUIDefinition.visualisation.subsections.physics.settings.repulsionDistance },
            { key: 'massScale', path: 'visualisation.physics.massScale', definition: settingsUIDefinition.visualisation.subsections.physics.settings.massScale },
            { key: 'boundaryDamping', path: 'visualisation.physics.boundaryDamping', definition: settingsUIDefinition.visualisation.subsections.physics.settings.boundaryDamping },
          ]
        }
      ]
    },
    xr: {
      label: 'XR/VR',
      icon: <Smartphone className="h-4 w-4" />,
      groups: [
        {
          title: 'XR Mode',
          description: 'Virtual reality settings',
          items: [
            { key: 'clientSideEnableXR', path: settingsUIDefinition.xr.subsections.general.settings.clientSideEnableXR.path, definition: settingsUIDefinition.xr.subsections.general.settings.clientSideEnableXR },
            { key: 'displayMode', path: settingsUIDefinition.xr.subsections.general.settings.displayMode.path, definition: settingsUIDefinition.xr.subsections.general.settings.displayMode },
            { key: 'quality', path: settingsUIDefinition.xr.subsections.general.settings.quality.path, definition: settingsUIDefinition.xr.subsections.general.settings.quality },
          ]
        }
      ]
    }
  }), []);

  // Filter settings based on search query
  const filterSettings = useCallback((groups: SettingGroup[], query: string): SettingGroup[] => {
    if (!query.trim()) return groups;
    
    const lowerQuery = query.toLowerCase();
    return groups
      .map(group => {
        const filteredItems = group.items.filter(item => {
          const matchesKey = item.key.toLowerCase().includes(lowerQuery);
          const matchesLabel = item.definition?.label?.toLowerCase().includes(lowerQuery);
          const matchesDescription = item.definition?.description?.toLowerCase().includes(lowerQuery);
          const matchesGroup = group.title.toLowerCase().includes(lowerQuery);
          const matchesGroupDesc = group.description?.toLowerCase().includes(lowerQuery);
          
          return matchesKey || matchesLabel || matchesDescription || matchesGroup || matchesGroupDesc;
        });
        
        if (filteredItems.length > 0) {
          return { ...group, items: filteredItems };
        }
        return null;
      })
      .filter(group => group !== null) as SettingGroup[];
  }, []);

  // Auto-expand groups when searching
  React.useEffect(() => {
    if (searchQuery.trim()) {
      const groupsToExpand = new Set<string>();
      Object.values(settingsStructure).forEach(section => {
        const filtered = filterSettings(section.groups, searchQuery);
        filtered.forEach(group => {
          groupsToExpand.add(group.title);
        });
      });
      setExpandedGroups(groupsToExpand);
    }
  }, [searchQuery, settingsStructure, filterSettings]);

  // Register keyboard shortcuts
  useKeyboardShortcuts({
    'settings-search': {
      key: '/',
      ctrl: true,
      description: 'Focus search in settings',
      category: 'Settings',
      handler: () => {
        const searchInput = document.querySelector('input[placeholder="Search settings..."]') as HTMLInputElement;
        searchInput?.focus();
      }
    },
    'settings-shortcuts': {
      key: '?',
      shift: true,
      description: 'Show keyboard shortcuts',
      category: 'General',
      handler: () => setShowShortcuts(true)
    },
    'settings-clear-search': {
      key: 'Escape',
      description: 'Clear search',
      category: 'Settings',
      handler: () => setSearchQuery(''),
      enabled: !!searchQuery
    }
  });

  // Simulate initial loading
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setIsInitializing(false);
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  const toggleGroup = useCallback((groupTitle: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(groupTitle)) {
        next.delete(groupTitle);
      } else {
        next.add(groupTitle);
      }
      return next;
    });
  }, []);

  const handleSettingChange = useCallback(async (path: string, value: any) => {
    setLoadingSettings(prev => new Set(prev).add(path));
    
    try {
      await new Promise((resolve) => setTimeout(resolve, 100));
      
      useSettingsStore.getState().set(path, value);

      toast({
        title: 'Setting saved',
        description: `Successfully updated ${path.split('.').pop()}`,
      });
      
      setSavedNotification(path);
      setTimeout(() => setSavedNotification(null), 2000);
    } catch (error) {
      handleError(error, {
        title: 'Failed to save setting',
        actionLabel: 'Retry',
        onAction: () => handleSettingChange(path, value)
      });
    } finally {
      setLoadingSettings(prev => {
        const next = new Set(prev);
        next.delete(path);
        return next;
      });
    }
  }, [handleError, toast]);

  const renderTabContent = useCallback((tabKey: string) => {
    const tab = settingsStructure[tabKey];
    if (!tab) return null;

    if (tab.isPowerUser && !isPowerUser) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-center p-6">
          <Settings className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium mb-2">Power User Features</h3>
          <p className="text-sm text-muted-foreground max-w-sm">
            Authenticate with Nostr to unlock advanced settings and features.
          </p>
        </div>
      );
    }

    const filteredGroups = filterSettings(tab.groups, searchQuery);
    
    if (searchQuery && filteredGroups.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-center p-6">
          <Search className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium mb-2">No results found</h3>
          <p className="text-sm text-muted-foreground max-w-sm">
            Try searching with different keywords or browse categories.
          </p>
        </div>
      );
    }

    return (
      <div className="flex-1 min-h-0 space-y-3">
        {isInitializing ? (
          <>
            <SkeletonSetting />
            <SkeletonSetting />
            <SkeletonSetting />
          </>
        ) : (
          filteredGroups.map((group, index) => (
            <VirtualizedSettingsGroup
              key={group.title}
              title={group.title}
              description={group.description}
              items={group.items}
              isPowerUser={group.isPowerUser}
              isExpanded={expandedGroups.has(group.title)}
              onToggle={() => toggleGroup(group.title)}
              savedNotification={savedNotification}
              loadingSettings={loadingSettings}
              onSettingChange={handleSettingChange}
              groupIndex={index}
            />
          ))
        )}
      </div>
    );
  }, [settingsStructure, isPowerUser, searchQuery, filterSettings, expandedGroups, toggleGroup, savedNotification, loadingSettings, handleSettingChange, isInitializing]);

  // Create tabs array for the Tabs component
  const tabs = useMemo(() => 
    Object.entries(settingsStructure).map(([key, section]) => ({
      label: section.label,
      icon: section.icon,
      content: renderTabContent(key)
    })),
    [settingsStructure, renderTabContent]
  );

  return (
    <div className="w-full h-full flex flex-col min-h-0 bg-background text-foreground">
      <div className="border-b border-border">
        <div className="px-4 py-3 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Settings</h2>
            <p className="text-sm text-muted-foreground">
              Customize your visualization
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleLowerRightPaneDock}
            title={isLowerRightPaneDocked ? "Expand lower panels" : "Collapse lower panels"}
          >
            {isLowerRightPaneDocked ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </Button>
        </div>
        <SearchSection
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onShowShortcuts={() => setShowShortcuts(true)}
        />
      </div>

      <div className="flex-1 overflow-auto">
        <Tabs
          tabs={tabs}
          className="h-full"
          tabListClassName="px-4 bg-muted/30"
          tabContentClassName="px-4 py-3"
        />
      </div>

      <div className="px-4 py-2 border-t border-border bg-muted/30 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <Info className="h-3 w-3" />
          <span>Changes save automatically</span>
        </div>
        {isPowerUser && (
          <div className="flex items-center gap-1 text-primary">
            <Settings className="h-3 w-3" />
            <span>Power User</span>
          </div>
        )}
      </div>

      <Suspense fallback={<LoadingSpinner />}>
        {showShortcuts && (
          <KeyboardShortcutsModal
            isOpen={showShortcuts}
            onClose={() => setShowShortcuts(false)}
          />
        )}
      </Suspense>
    </div>
  );
}, (prevProps, nextProps) => {
  return (
    prevProps.isLowerRightPaneDocked === nextProps.isLowerRightPaneDocked &&
    prevProps.toggleLowerRightPaneDock === nextProps.toggleLowerRightPaneDock
  );
});

SettingsPanelRedesignOptimized.displayName = 'SettingsPanelRedesignOptimized';

export default SettingsPanelRedesignOptimized;