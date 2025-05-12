import React from 'react';
// Removed Card, Label, Input, useSettingsStore, produce imports as they are handled by child components or not needed.
import { SettingsSection } from '../SettingsSection'; // Import SettingsSection
import { UICategoryDefinition } from '../../config/settingsUIDefinition'; // Import definition type

export interface AIPanelProps { // Renamed interface
  settingsDef: UICategoryDefinition;
}

const AIPanel: React.FC<AIPanelProps> = ({ settingsDef }) => {
  // Removed settings store access and handleChange function.
  // This will now be handled by SettingsSection and SettingControlComponent via settingsDef.

  return (
    <div className="p-4 space-y-6 overflow-y-auto h-full custom-scrollbar">
      {/* Iterate through AI service subsections defined in settingsDef */}
      {Object.entries(settingsDef.subsections).map(([subsectionKey, subsectionDef]) => (
        <SettingsSection
          key={subsectionKey}
          id={`settings-${settingsDef.label.toLowerCase().replace(/\s+/g, '-')}-${subsectionKey}`} // e.g., settings-ai-services-ragflow
          title={subsectionDef.label}
          subsectionSettings={subsectionDef.settings}
        />
      ))}
    </div>
  );
};

export default AIPanel;
