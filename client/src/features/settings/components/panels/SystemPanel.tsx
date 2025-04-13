import { useState, useEffect } from 'react';
import { useSettingsStore } from '../../../../store/settingsStore';
// Removed imports for deleted Panel and PanelToolbar components
import { formatSettingLabel } from '../../types/settingsSchema';
import { FormGroup, FormGroupControl } from '../../../../ui/formGroup/FormGroup';
import { UISetting, isUISetting } from '../../types/uiSetting';
import { createLogger } from '../../../../utils/logger';
import { graphDataManager } from '../../../graph/managers/graphDataManager';

const logger = createLogger('SystemPanel');

// Subsections for system settings
const SYSTEM_SUBSECTIONS = [
  { id: 'api', title: 'API' },
  { id: 'debug', title: 'Debug' }
];

interface SystemPanelProps {
  /**
   * Panel ID for the panel system
   */
  panelId: string;
}

/**
 * SystemPanel provides access to system-level settings and debug options.
 * Panel ID is no longer needed as it's not rendered in the panel system.
 * Panel ID is no longer needed.
 */
// panelId: string; // Removed panelId prop definition
// panelId: string; // Removed panelId prop
const SystemPanel = ({
  // panelId // Prop removed
}: SystemPanelProps) => {
  const [activeSubsection, setActiveSubsection] = useState('api');
  const [apiStatus, setApiStatus] = useState({
    isConnected: false,
    lastFetchTime: null as string | null,
    nodesCount: 0,
    edgesCount: 0
  });

  const settings = useSettingsStore(state => state.settings);
  const setSettings = useSettingsStore(state => state.set);

  // Get system settings for the active subsection
  const systemSettings: Record<string, UISetting> =
    settings.system &&
    settings.system[activeSubsection] ?
    settings.system[activeSubsection] as Record<string, UISetting> : {};

  // Update a specific setting
  const updateSetting = (path: string, value: any) => {
    const fullPath = `system.${activeSubsection}.${path}`;
    logger.debug(`Updating setting: ${fullPath}`, value);
    setSettings(fullPath, value);
  };

  // Toggle a boolean setting
  const toggleSetting = (path: string) => {
    const currentValue = systemSettings[path]?.value;
    if (typeof currentValue === 'boolean') {
      updateSetting(path, !currentValue);
    }
  };

  // Update API status based on current graph data
  useEffect(() => {
    const updateApiStatus = () => {
      try {
        // Get current graph data from the manager
        const currentData = graphDataManager.getGraphData();

        // Update API status based on current data
        setApiStatus({
          isConnected: true, // Assume connected if we have data
          lastFetchTime: new Date().toLocaleTimeString(),
          nodesCount: currentData.nodes.length,
          edgesCount: currentData.edges.length
        });
      } catch (error) {
        logger.error('Failed to update API status:', error);
        setApiStatus(prev => ({
          ...prev,
          isConnected: false
        }));
      }
    };

    // Initial update
    updateApiStatus();

    // Subscribe to graph data changes to update API status
    const unsubscribe = graphDataManager.onGraphDataChange(() => {
      updateApiStatus();
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // Handle debug actions
  const handleClearConsole = () => {
    console.clear();
    logger.info('Console cleared');
  };

  const handleResetGraph = async () => {
    try {
      // Only fetch initial data if explicitly requested by the user
      await graphDataManager.fetchInitialData();
      logger.info('Graph data reset successfully');
    } catch (error) {
      logger.error('Failed to reset graph:', error);
    }
  };

  const handleExportLogs = () => {
    const logs = logger.getLogs();
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `logs-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleMemoryUsage = () => {
    try {
      // @ts-ignore - performance.memory is Chrome-specific
      const memory = window.performance.memory;
      if (memory) {
        logger.info('Memory usage:', {
          usedJSHeapSize: `${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
          totalJSHeapSize: `${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
          jsHeapSizeLimit: `${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`
        });
      } else {
        logger.warn('Memory usage information not available in this browser');
      }
    } catch (error) {
      logger.error('Failed to get memory usage:', error);
    }
  };

  // Return the content directly without Panel wrapper or Toolbar
  return (
      // The div below is now the top-level returned element
      <div className="flex flex-col h-full">
        {/* Subsection Tabs */}
        <div className="flex border-b border-border overflow-x-auto">
          {SYSTEM_SUBSECTIONS.map(subsection => (
            <button
              key={subsection.id}
              className={`px-3 py-2 ${
                activeSubsection === subsection.id
                  ? 'border-b-2 border-primary font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
              onClick={() => setActiveSubsection(subsection.id)}
            >
              {subsection.title}
            </button>
          ))}
        </div>

        {/* Settings Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* API Settings */}
          {activeSubsection === 'api' && (
            <div className="space-y-6">
              {/* API Connection Status */}
              <div className="mt-6 p-4 bg-muted rounded-md">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">API Status</span>
                  <div className="flex items-center space-x-2">
                    <span className={`inline-block w-2 h-2 rounded-full ${
                      apiStatus.isConnected ? 'bg-green-500' : 'bg-red-500'
                    }`}></span>
                    <span className="text-sm text-muted-foreground">
                      {apiStatus.isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                  </div>
                </div>

                <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                  <div className="text-muted-foreground">Last Fetch</div>
                  <div>{apiStatus.lastFetchTime || 'Never'}</div>

                  <div className="text-muted-foreground">Nodes</div>
                  <div>{apiStatus.nodesCount}</div>

                  <div className="text-muted-foreground">Edges</div>
                  <div>{apiStatus.edgesCount}</div>
                </div>

                <div className="mt-4 flex justify-end">
                  <button
                    className="px-3 py-1 text-xs bg-secondary rounded-md hover:bg-secondary/80"
                    onClick={() => graphDataManager.fetchInitialData()}
                  >
                    Refresh Data
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Debug Settings */}
          {activeSubsection === 'debug' && (
            <div className="space-y-6">
              {/* Debug Settings */}
              {Object.entries(systemSettings).map(([key, setting]) => {
                if (typeof setting !== 'object' || setting === null) {
                  return null;
                }

                if (!isUISetting(setting)) {
                  return null;
                }

                const label = formatSettingLabel(key);

                return (
                  <FormGroup
                    key={key}
                    label={label}
                    id={key}
                    helpText={setting.description}
                  >
                    <FormGroupControl>
                      {setting.type === 'checkbox' && (
                        <div className="flex items-center space-x-2">
                          <input
                            id={key}
                            type="checkbox"
                            checked={setting.value}
                            className="rounded"
                            onChange={() => toggleSetting(key)}
                          />
                          <span className="text-sm text-muted-foreground">
                            {setting.value ? 'Enabled' : 'Disabled'}
                          </span>
                        </div>
                      )}

                      {setting.type === 'select' && setting.options && (
                        <select
                          id={key}
                          value={setting.value}
                          className="w-full rounded-md border border-input bg-transparent px-3 py-1"
                          onChange={(e) => updateSetting(key, e.target.value)}
                        >
                          {setting.options.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      )}
                    </FormGroupControl>
                  </FormGroup>
                );
              })}

              {/* Debug Actions */}
              <div className="mt-6">
                <h3 className="text-sm font-medium mb-3">Debug Actions</h3>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    className="px-3 py-2 bg-secondary rounded-md hover:bg-secondary/80 text-sm"
                    onClick={handleClearConsole}
                  >
                    Clear Console
                  </button>

                  <button
                    className="px-3 py-2 bg-secondary rounded-md hover:bg-secondary/80 text-sm"
                    onClick={handleResetGraph}
                  >
                    Reset Graph
                  </button>

                  <button
                    className="px-3 py-2 bg-secondary rounded-md hover:bg-secondary/80 text-sm"
                    onClick={handleExportLogs}
                  >
                    Export Logs
                  </button>

                  <button
                    className="px-3 py-2 bg-secondary rounded-md hover:bg-secondary/80 text-sm"
                    onClick={handleMemoryUsage}
                  >
                    Memory Usage
                  </button>
                </div>

                <div className="mt-4 p-3 bg-muted rounded-md text-xs font-mono h-32 overflow-auto">
                  <div className="text-green-500">● Session started at {new Date().toLocaleTimeString()}</div>
                  <div className="text-muted-foreground">● Loaded configuration</div>
                  <div className="text-muted-foreground">● API connected</div>
                  <div className="text-muted-foreground">● Graph initialized with {apiStatus.nodesCount} nodes</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    // No closing tag needed here as the div above is the root
  );
};

export default SystemPanel;