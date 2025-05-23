import React, { useState, useMemo } from 'react';
import Tabs from '@/ui/Tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/Card';
import { Button } from '@/ui/Button';
import {
  Eye,
  Settings,
  Lock,
  Smartphone,
  Info,
  ChevronDown,
  Check
} from 'lucide-react';
import { useSettingsStore } from '@/store/settingsStore';
import { SettingControlComponent } from '../SettingControlComponent';
import { settingsUIDefinition } from '../../config/settingsUIDefinition';
import { cn } from '@/utils/cn';

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

export function SettingsPanelRedesign() {
  const { settings, isPowerUser } = useSettingsStore();
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['Node Appearance']));
  const [savedNotification, setSavedNotification] = useState<string | null>(null);

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
          ]
        },
        {
          title: 'Labels',
          description: 'Text display settings',
          items: [
            { key: 'enableLabels', path: settingsUIDefinition.visualisation.subsections.labels.settings.enableLabels.path, definition: settingsUIDefinition.visualisation.subsections.labels.settings.enableLabels },
            { key: 'desktopFontSize', path: 'visualisation.labels.desktopFontSize', definition: settingsUIDefinition.visualisation.subsections.labels.settings.desktopFontSize },
            { key: 'textColor', path: 'visualisation.labels.textColor', definition: settingsUIDefinition.visualisation.subsections.labels.settings.textColor },
          ]
        },
        {
          title: 'Visual Effects',
          description: 'Bloom and glow effects',
          items: [
            { key: 'bloomEnabled', path: 'visualisation.bloom.enabled', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.enabled },
            { key: 'bloomStrength', path: 'visualisation.bloom.strength', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.strength },
            { key: 'bloomRadius', path: 'visualisation.bloom.radius', definition: settingsUIDefinition.visualisation.subsections.bloom.settings.radius },
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
          ]
        },
        {
          title: 'Network Settings',
          description: 'Connection optimization',
          items: [
            { key: 'updateRate', path: 'system.websocket.updateRate', definition: settingsUIDefinition.system.subsections.websocket.settings.updateRate },
            { key: 'compressionEnabled', path: 'system.websocket.compressionEnabled', definition: settingsUIDefinition.system.subsections.websocket.settings.compressionEnabled },
          ],
          isPowerUser: true
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
        },
        {
          title: 'Interaction',
          description: 'Hand tracking and controls',
          items: [
            { key: 'handTracking', path: settingsUIDefinition.xr.subsections.handFeatures.settings.handTracking.path, definition: settingsUIDefinition.xr.subsections.handFeatures.settings.handTracking },
            { key: 'haptics', path: settingsUIDefinition.xr.subsections.handFeatures.settings.enableHaptics.path, definition: settingsUIDefinition.xr.subsections.handFeatures.settings.enableHaptics },
          ],
          isPowerUser: true
        }
      ]
    },
    advanced: {
      label: 'Advanced',
      icon: <Lock className="h-4 w-4" />,
      isPowerUser: true,
      groups: [
        {
          title: 'Debug Options',
          description: 'Developer tools',
          items: [
            { key: 'debugMode', path: settingsUIDefinition.system.subsections.debug.settings.enabled.path, definition: settingsUIDefinition.system.subsections.debug.settings.enabled },
            { key: 'logLevel', path: 'system.debug.logLevel', definition: settingsUIDefinition.system.subsections.debug.settings.logLevel },
          ]
        },
        {
          title: 'AI Services',
          description: 'API configuration',
          items: [
            { key: 'ragflowApiKey', path: settingsUIDefinition.ai.subsections.ragflow.settings.apiKey.path, definition: settingsUIDefinition.ai.subsections.ragflow.settings.apiKey },
            { key: 'perplexityKey', path: settingsUIDefinition.ai.subsections.perplexity.settings.apiKey.path, definition: settingsUIDefinition.ai.subsections.perplexity.settings.apiKey },
            { key: 'openaiKey', path: settingsUIDefinition.ai.subsections.openai.settings.apiKey.path, definition: settingsUIDefinition.ai.subsections.openai.settings.apiKey },
          ]
        }
      ]
    }
  }), []);

  const toggleGroup = (groupTitle: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(groupTitle)) {
        next.delete(groupTitle);
      } else {
        next.add(groupTitle);
      }
      return next;
    });
  };

  const handleSettingChange = (path: string, value: any) => {
    useSettingsStore.getState().set(path, value);
    
    // Show save notification
    setSavedNotification(path);
    setTimeout(() => setSavedNotification(null), 2000);
  };

  const renderSettingGroup = (group: SettingGroup) => {
    if (group.isPowerUser && !isPowerUser) return null;

    const isExpanded = expandedGroups.has(group.title);

    return (
      <Card key={group.title} className="mb-3 overflow-hidden">
        <CardHeader 
          className="cursor-pointer py-3 px-4 hover:bg-muted/50 transition-colors"
          onClick={() => toggleGroup(group.title)}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                {group.title}
                {group.isPowerUser && (
                  <span className="text-xs px-1.5 py-0.5 bg-primary/10 text-primary rounded">
                    Pro
                  </span>
                )}
              </CardTitle>
              {group.description && (
                <CardDescription className="text-xs mt-1">
                  {group.description}
                </CardDescription>
              )}
            </div>
            <ChevronDown
              className={cn(
                "h-4 w-4 transition-transform duration-200",
                isExpanded ? "" : "-rotate-90"
              )}
            />
          </div>
        </CardHeader>
        
        {isExpanded && (
          <CardContent className="pt-0 pb-3 px-4 space-y-3">
            {group.items.map((item) => {
              if (item.isPowerUser && !isPowerUser) return null;
              
              const value = useSettingsStore.getState().get(item.path);
              
              return (
                <div key={item.key} className="relative">
                  <SettingControlComponent
                    path={item.path}
                    settingDef={item.definition}
                    value={value}
                    onChange={(newValue) => handleSettingChange(item.path, newValue)}
                  />
                  {savedNotification === item.path && (
                    <div className="absolute -top-1 -right-1 flex items-center gap-1 text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                      <Check className="h-3 w-3" />
                      Saved
                    </div>
                  )}
                </div>
              );
            })}
          </CardContent>
        )}
      </Card>
    );
  };

  const renderTabContent = (tabKey: string) => {
    const tab = settingsStructure[tabKey];
    if (!tab) return null;

    if (tab.isPowerUser && !isPowerUser) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-center p-6">
          <Lock className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium mb-2">Power User Features</h3>
          <p className="text-sm text-muted-foreground max-w-sm">
            Authenticate with Nostr to unlock advanced settings and features.
          </p>
        </div>
      );
    }

    return (
      <div className="space-y-3">
        {tab.groups.map(group => renderSettingGroup(group))}
      </div>
    );
  };

  // Create tabs array for the Tabs component
  const tabs = Object.entries(settingsStructure).map(([key, section]) => ({
    label: section.label,
    icon: section.icon,
    content: renderTabContent(key)
  }));

  return (
    <div className="w-full h-full flex flex-col">
      <div className="px-4 py-3 border-b">
        <h2 className="text-lg font-semibold">Settings</h2>
        <p className="text-sm text-muted-foreground">
          Customize your visualization
        </p>
      </div>

      <div className="flex-1 overflow-hidden">
        <Tabs 
          tabs={tabs}
          className="h-full"
          tabListClassName="px-4"
          tabContentClassName="px-4 py-3"
        />
      </div>

      {/* Status bar */}
      <div className="px-4 py-2 border-t bg-muted/30 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Info className="h-3 w-3" />
          <span>Changes save automatically</span>
        </div>
        {isPowerUser && (
          <div className="flex items-center gap-1">
            <Lock className="h-3 w-3 text-primary" />
            <span className="text-primary">Power User</span>
          </div>
        )}
      </div>
    </div>
  );
}