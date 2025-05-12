import React from 'react'; // Removed useState
import { useSettingsStore } from '../../../../store/settingsStore'; // Keep for persisting preference if needed
import { createLogger } from '../../../../utils/logger';
import { SettingsSection } from '../SettingsSection';
import { UICategoryDefinition, UISettingDefinition } from '../../config/settingsUIDefinition';
import { useApplicationMode } from '../../../../contexts/ApplicationModeContext'; // For XR mode toggle
import { Switch } from '../../../../ui/Switch'; // For XR mode toggle
import { Label } from '../../../../ui/Label'; // For XR mode toggle label
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/ui/Tooltip'; // For XR mode toggle tooltip
import { Info } from 'lucide-react'; // For XR mode toggle tooltip icon
import { FormGroup } from '@/ui/formGroup/FormGroup'; // For XR mode toggle styling

const logger = createLogger('XRPanel');

// Removed XR_SUBSECTIONS constant

export interface XRPanelProps { // Renamed interface
  settingsDef: UICategoryDefinition;
}

const XRPanel: React.FC<XRPanelProps> = ({ settingsDef }) => {
  const { mode: applicationMode, setMode: setApplicationMode } = useApplicationMode();
  const settingsStoreSet = useSettingsStore(state => state.set); // For persisting preference

  // Find the clientSideEnableXR definition for direct rendering
  let clientSideEnableXRDef: UISettingDefinition | undefined;
  for (const subsec of Object.values(settingsDef.subsections)) {
    if (subsec.settings['clientSideEnableXR']) {
      clientSideEnableXRDef = subsec.settings['clientSideEnableXR'];
      break;
    }
  }

  const handleXRModeToggle = (checked: boolean) => {
    setApplicationMode(checked ? 'xr' : 'desktop');
    // Optionally persist this preference to settings store if clientSideEnableXRDef.path is valid
    if (clientSideEnableXRDef?.path) {
      settingsStoreSet(clientSideEnableXRDef.path, checked);
      logger.info(`XR mode preference (${clientSideEnableXRDef.path}) set to: ${checked}`);
    }
  };

  const isClientXRModeEnabled = applicationMode === 'xr';

  return (
    <div className="p-4 space-y-6 overflow-y-auto h-full custom-scrollbar">
      {/* Special handling for clientSideEnableXR toggle */}
      {clientSideEnableXRDef && (
        <FormGroup
            label={clientSideEnableXRDef.label}
            id="client-xr-toggle"
            helpText={clientSideEnableXRDef.description}
            className="border-b border-border pb-4 mb-4"
        >
            <div className="flex items-center justify-between">
                <Label htmlFor="client-xr-toggle-switch" className="text-sm flex items-center gap-1">
                    <span>{clientSideEnableXRDef.label}</span>
                    {clientSideEnableXRDef.description && (
                    <TooltipProvider delayDuration={100}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent side="top" align="start" className="max-w-xs z-[4000]">
                                <p>{clientSideEnableXRDef.description}</p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                    )}
                </Label>
                <Switch
                    id="client-xr-toggle-switch"
                    checked={isClientXRModeEnabled}
                    onCheckedChange={handleXRModeToggle}
                />
            </div>
        </FormGroup>
      )}

      {/* Iterate through subsections defined in settingsDef */}
      {Object.entries(settingsDef.subsections).map(([subsectionKey, subsectionDef]) => {
        // Filter out the clientSideEnableXR setting if it was handled above, to avoid rendering it twice
        const filteredSettings = { ...subsectionDef.settings };
        if (clientSideEnableXRDef && filteredSettings[clientSideEnableXRDef.path.split('.').pop()!]) {
             // Check if the current subsection contains the clientSideEnableXR setting
            if (clientSideEnableXRDef.path.startsWith(`${settingsDef.label.toLowerCase()}.${subsectionKey}`)) {
                delete filteredSettings[clientSideEnableXRDef.path.split('.').pop()!];
            }
        }

        // If after filtering, the subsection has no settings left (and it's not the one containing the special toggle),
        // or if it only contained the special toggle, don't render the section.
        if (Object.keys(filteredSettings).length === 0) {
            return null;
        }
        
        return (
          <SettingsSection
            key={subsectionKey}
            id={`settings-${settingsDef.label.toLowerCase()}-${subsectionKey}`}
            title={subsectionDef.label}
            subsectionSettings={filteredSettings} // Pass filtered settings
          />
        );
      })}

      {/* Retain custom informational text if needed */}
      <div className="space-y-6 pt-4 border-t border-border mt-6">
        <div className="bg-muted p-4 rounded-md text-sm shadow">
          <h4 className="font-medium mb-2 text-foreground">XR Control Information</h4>
          <p className="text-muted-foreground mb-2">
            These settings control how interaction works in VR and AR modes.
            When using a VR headset, you can use the controllers to interact with the visualisation.
          </p>
          <ul className="list-disc list-inside text-muted-foreground space-y-1">
            <li>Trigger button: Select</li>
            <li>Grip button: Grab and move</li>
            <li>Thumbstick: Navigate and rotate</li>
          </ul>
        </div>
        <div className="bg-muted p-4 rounded-md text-sm shadow">
          <h4 className="font-medium mb-2 text-foreground">XR Environment</h4>
          <p className="text-muted-foreground">
            These settings control the visual environment in VR and AR modes,
            including background, lighting, and scale.
          </p>
        </div>
      </div>
    </div>
  );
};

export default XRPanel;