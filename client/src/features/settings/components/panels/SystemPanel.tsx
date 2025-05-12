import React, { useState, useEffect } from 'react'; // Added React
import { useSettingsStore } from '../../../../store/settingsStore';
import { createLogger } from '../../../../utils/logger';
import { graphDataManager } from '../../../graph/managers/graphDataManager';
import { SettingsSection } from '../SettingsSection'; // Import SettingsSection
import { UICategoryDefinition } from '../../config/settingsUIDefinition'; // Import definition type

const logger = createLogger('SystemPanel');

// Removed SYSTEM_SUBSECTIONS constant

export interface SystemPanelProps { // Renamed interface
  settingsDef: UICategoryDefinition;
}

const SystemPanel: React.FC<SystemPanelProps> = ({ settingsDef }) => {
  // Removed activeSubsection state
  const [apiStatus, setApiStatus] = useState({
    isConnected: false,
    lastFetchTime: null as string | null,
    nodesCount: 0,
    edgesCount: 0
  });

  // Removed settings and setSettings from useSettingsStore as they are handled by SettingsSection/SettingControlComponent

  // Removed old systemSettings retrieval and updateSetting/toggleSetting functions

  useEffect(() => {
    const updateApiStatus = () => {
      try {
        const currentData = graphDataManager.getGraphData();
        setApiStatus({
          isConnected: true,
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
    updateApiStatus();
    const unsubscribe = graphDataManager.onGraphDataChange(updateApiStatus);
    return () => unsubscribe();
  }, []);

  const handleClearConsole = () => {
    console.clear();
    logger.info('Console cleared');
  };

  const handleResetGraph = async () => {
    try {
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

  return (
    <div className="p-4 space-y-6 overflow-y-auto h-full custom-scrollbar">
      {/* Iterate through subsections defined in settingsDef */}
      {Object.entries(settingsDef.subsections).map(([subsectionKey, subsectionDef]) => (
        <SettingsSection
          key={subsectionKey}
          id={`settings-${settingsDef.label.toLowerCase()}-${subsectionKey}`}
          title={subsectionDef.label}
          subsectionSettings={subsectionDef.settings}
        />
      ))}

      {/* Custom System Panel Content: API Status and Debug Actions */}
      <div className="space-y-6 pt-4 border-t border-border mt-6">
        <h3 className="text-lg font-semibold text-foreground">API Status & Diagnostics</h3>
        {/* API Connection Status */}
        <div className="p-4 bg-muted rounded-md shadow">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-foreground">API Status</span>
            <div className="flex items-center space-x-2">
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${
                apiStatus.isConnected ? 'bg-green-500' : 'bg-red-500'
              }`}></span>
              <span className="text-sm text-muted-foreground">
                {apiStatus.isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <div className="text-muted-foreground">Last Fetch:</div>
            <div className="text-foreground">{apiStatus.lastFetchTime || 'N/A'}</div>
            <div className="text-muted-foreground">Nodes:</div>
            <div className="text-foreground">{apiStatus.nodesCount}</div>
            <div className="text-muted-foreground">Edges:</div>
            <div className="text-foreground">{apiStatus.edgesCount}</div>
          </div>
          <div className="mt-3 flex justify-end">
            <button
              className="px-3 py-1.5 text-xs bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors"
              onClick={() => graphDataManager.fetchInitialData()}
            >
              Refresh Data
            </button>
          </div>
        </div>

        {/* Debug Actions */}
        <div className="p-4 bg-muted rounded-md shadow">
          <h4 className="text-sm font-medium text-foreground mb-3">Debug Actions</h4>
          <div className="grid grid-cols-2 gap-2">
            <button
              className="px-3 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 text-sm transition-colors"
              onClick={handleClearConsole}
            >
              Clear Console
            </button>
            <button
              className="px-3 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 text-sm transition-colors"
              onClick={handleResetGraph}
            >
              Reset Graph
            </button>
            <button
              className="px-3 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 text-sm transition-colors"
              onClick={handleExportLogs}
            >
              Export Logs
            </button>
            <button
              className="px-3 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 text-sm transition-colors"
              onClick={handleMemoryUsage}
            >
              Memory Usage
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemPanel;