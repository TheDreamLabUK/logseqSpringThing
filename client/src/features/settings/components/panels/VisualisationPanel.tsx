import React from 'react';
import { SettingsSection } from '../SettingsSection'; // Adjust path if necessary
import { UICategoryDefinition } from '../../config/settingsUIDefinition'; // Adjust path if necessary
// Removed unused imports like useState, useMemo, useSettingsStore, formatSettingLabel, createLogger, specific icons, UI components (Input, Switch, etc.), useTheme

// Removed logger initialization
// Removed VISUALIZATION_SUBSECTIONS constant

export interface VisualisationPanelProps { // Renamed interface for clarity
  settingsDef: UICategoryDefinition;
}

/**
 * VisualisationPanel renders settings controls for the 'Visualisation' category,
 * driven by the provided settings definition.
 */
const VisualisationPanel: React.FC<VisualisationPanelProps> = ({ settingsDef }) => {
  // Removed old state (activeSubsection) and settings retrieval logic (useSettingsStore, useMemo)
  // Removed updateSetting function (will be handled within SettingsSection/SettingControlComponent)
  // Removed theme logic (assuming handled by ThemeProvider globally)

  return (
    // Main container: Added padding, space between sections, overflow for scrolling, and full height
    <div className="p-4 space-y-6 overflow-y-auto h-full custom-scrollbar">
      {/* Iterate through subsections defined in settingsDef */}
      {Object.entries(settingsDef.subsections).map(([subsectionKey, subsectionDef]) => (
        <SettingsSection
          key={subsectionKey}
          // Generate a unique ID for accessibility/linking if needed
          id={`settings-${settingsDef.label.toLowerCase()}-${subsectionKey}`} // e.g., settings-visualisation-nodes
          title={subsectionDef.label}
          // Pass the specific settings definitions for this subsection
          subsectionSettings={subsectionDef.settings}
        />
      ))}
    </div>
  );
};

export default VisualisationPanel;
