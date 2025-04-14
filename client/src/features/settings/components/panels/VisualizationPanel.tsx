import React, { useState, useMemo } from 'react';
import { useSettingsStore } from '../../../../store/settingsStore';
import { formatSettingLabel } from '../../types/settingsSchema';
import { createLogger } from '../../../../utils/logger';
import { Eye, CircleDashed, Circle, MoveHorizontal } from 'lucide-react';
// Import UI components
import { Input } from '../../../../ui/Input';
import { Switch } from '../../../../ui/Switch';
import { Slider } from '../../../../ui/Slider';
import { Label } from '../../../../ui/Label';
import { RadioGroup, RadioGroupItem } from '../../../../ui/RadioGroup';
import { useTheme } from '../../../../ui/ThemeProvider';

const logger = createLogger('VisualizationPanel');

// Subsections for visualization settings
const VISUALIZATION_SUBSECTIONS = [
  { id: 'rendering', title: 'Rendering', icon: <Eye className="h-4 w-4" /> },
  { id: 'nodes', title: 'Nodes', icon: <Circle className="h-4 w-4" /> },
  { id: 'edges', title: 'Edges', icon: <MoveHorizontal className="h-4 w-4" /> },
  { id: 'labels', title: 'Labels', icon: <Circle className="h-4 w-4" /> },
  { id: 'physics', title: 'Physics', icon: <CircleDashed className="h-4 w-4" /> },
  { id: 'bloom', title: 'Bloom', icon: <Circle className="h-4 w-4" /> },
  { id: 'hologram', title: 'Hologram', icon: <Circle className="h-4 w-4" /> },
  { id: 'animations', title: 'Animations', icon: <Circle className="h-4 w-4" /> },
];

interface VisualizationPanelProps {
  /**
   * Panel ID for the panel system
   * Panel ID is no longer needed.
   */
  // panelId: string; // Removed panelId prop

  /**
   * Horizontal layout is no longer relevant.
   */
  // horizontal?: boolean; // Removed horizontal prop
 }

/**
 * VisualizationPanel provides a comprehensive interface for managing all visualization settings.
 * This includes rendering options, node/edge appearance, and physics simulation parameters.
 */
const VisualizationPanel = ({
  // panelId, // Prop removed
  // horizontal // Prop removed
}: VisualizationPanelProps) => {
  const [activeSubsection, setActiveSubsection] = useState('rendering');

  const settings = useSettingsStore(state => state.settings);
  const setSettings = useSettingsStore(state => state.set); // Keep setSettings

  // Get the actual settings *values* for the active subsection using useMemo
  const activeSettingsValues = useMemo(() => {
    const vizSettings = settings?.visualization;
    if (!vizSettings || !activeSubsection || !(activeSubsection in vizSettings)) {
      logger.warn(`Subsection '${activeSubsection}' not found in visualization settings.`);
      return {};
    }
    // Type assertion might be needed if TypeScript can't infer the structure
    return vizSettings[activeSubsection as keyof typeof vizSettings] || {};
  }, [settings, activeSubsection]);


  // Update a specific setting (remains the same)
  const updateSetting = (path: string, value: any) => {
    const fullPath = `visualization.${activeSubsection}.${path}`;
    logger.debug(`Updating setting: ${fullPath}`, value);
    setSettings(fullPath, value);
  };

  // Use the theme context to ensure dark mode is applied
  const { theme } = useTheme();

  // Create a class based on the current theme
  const themeClass = theme === 'dark' ? 'dark' : '';

  return (
    // Apply dark theme classes directly to the fragment's container div
    <div className={`${themeClass} bg-card text-card-foreground dark:bg-gray-900 dark:text-gray-100 h-full`}>
      {/* Panel Content */}
      {/* Vertical flex layout with proper height constraints */}
      <div className="h-full flex flex-col">
        {/* Subsection Tabs - Horizontal scrollable tabs */}
        <div className="flex border-b border-border dark:border-gray-700 overflow-x-auto bg-card dark:bg-gray-800 no-scrollbar">
          {VISUALIZATION_SUBSECTIONS.map(subsection => (
            <button
              key={subsection.id}
              className={`flex items-center px-4 py-3 transition-colors whitespace-nowrap ${
                activeSubsection === subsection.id
                  ? 'border-b-2 border-primary text-primary font-medium dark:text-blue-400 dark:border-blue-400'
                  : 'text-muted-foreground hover:text-foreground dark:text-gray-400 dark:hover:text-gray-200'
              }`}
              onClick={() => setActiveSubsection(subsection.id)}
            >
              {subsection.icon && <span className="mr-2">{subsection.icon}</span>}
              {subsection.title}
            </button>
          ))}
        </div>

        {/* Settings Content - Improved scrolling with custom scrollbar */}
        <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 dark:bg-gray-900 settings-panel-scroll">
          {/* Enhanced Input Renderer */}
          {Object.entries(activeSettingsValues).map(([key, value]) => {
            const label = formatSettingLabel(key); // Format the key for display

            // Determine input type based on value type and key name
            let inputType: React.HTMLInputTypeAttribute = 'text';
            // Initialize controlValue without assigning 'value' yet
            let controlValue: string | number | boolean | undefined = undefined;
            let additionalProps: Record<string, any> = {};

            // Check if this setting should use a radio button
            const radioOptions: string[] = [];

            // Identify settings that should use radio buttons
            if (key === 'billboardMode' && typeof value === 'string') {
              inputType = 'radio';
              controlValue = value;
              radioOptions.push('camera', 'fixed', 'horizontal');
            } else if (key === 'quality' && typeof value === 'string') {
              inputType = 'radio';
              controlValue = value;
              radioOptions.push('low', 'medium', 'high');
            } else if (key === 'position' && typeof value === 'string') {
              inputType = 'radio';
              controlValue = value;
              radioOptions.push('top-left', 'top-right', 'bottom-left', 'bottom-right');
            } else if (typeof value === 'boolean') {
              inputType = 'checkbox';
              controlValue = value; // Assign directly, type is known
            } else if (typeof value === 'number') {
              inputType = 'number';
              controlValue = value; // Assign directly, type is known
              // Basic step for numbers
              additionalProps.step = 0.1;
            } else if (typeof value === 'string' && /^#([0-9A-F]{3}){1,2}$/i.test(value)) {
              inputType = 'color';
              controlValue = value; // Assign directly, type is known
            } else if (typeof value === 'string') {
              inputType = 'text';
              controlValue = value; // Assign directly, type is known
            } else {
              // Skip complex types like arrays/objects for now
              return null;
            }

            // Determine default slider range/step based on key or value
            let sliderMin = 0;
            let sliderMax = 100;
            let sliderStep = 1;
            if (typeof controlValue === 'number') {
              if (controlValue >= 0 && controlValue <= 1) {
                sliderMax = 1;
                sliderStep = 0.01;
              } else if (controlValue >= 0 && controlValue <= 10) {
                 sliderMax = 10;
                 sliderStep = 0.1;
              } else if (controlValue > 100) {
                 sliderMax = Math.max(sliderMax, controlValue * 1.5); // Adjust max if value is large
              }
            }

            // Add specific overrides based on key name if needed
            if (key.toLowerCase().includes('opacity') || key.toLowerCase().includes('intensity') || key.toLowerCase().includes('strength') || key.toLowerCase().includes('damping')) {
                sliderMax = Math.max(1, sliderMax); // Ensure max is at least 1 for these
                sliderStep = 0.01;
            }
            if (key.toLowerCase().includes('size') || key.toLowerCase().includes('width') || key.toLowerCase().includes('radius') || key.toLowerCase().includes('distance')) {
                sliderMax = Math.max(10, sliderMax); // Ensure max is at least 10
                sliderStep = 0.1;
            }
            if (key.toLowerCase().includes('count') || key.toLowerCase().includes('iterations')) {
                sliderMin = 0;
                sliderMax = Math.max(100, sliderMax); // Ensure max is at least 100
                sliderStep = 1;
            }

            // Create a unique ID for each control
            const controlId = `viz-${activeSubsection}-${key}`;

            return (
              <div key={key} className="grid grid-cols-3 items-center gap-4 text-sm border-b border-border/50 pb-3 dark:border-gray-700">
                <Label
                  htmlFor={controlId}
                  className="font-medium text-foreground/90 col-span-1 dark:text-gray-200"
                >
                  {label}
                </Label>
                <div className="col-span-2 flex items-center space-x-3"> {/* Increased spacing */}
                  {inputType === 'checkbox' ? (
                    <div className="flex items-center space-x-2">
                      <Switch
                        id={controlId}
                        checked={controlValue as boolean}
                        onCheckedChange={(checked: boolean) => updateSetting(key, checked)}
                      />
                      <span className="text-sm text-muted-foreground dark:text-gray-400">
                        {(controlValue as boolean) ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  ) : inputType === 'radio' ? (
                    <RadioGroup
                      value={controlValue as string}
                      onValueChange={(value: string) => updateSetting(key, value)}
                      className="flex flex-col space-y-1"
                    >
                      {radioOptions.map((option) => (
                        <div key={option} className="flex items-center space-x-2">
                          <RadioGroupItem value={option} id={`${controlId}-${option}`} />
                          <Label
                            htmlFor={`${controlId}-${option}`}
                            className="text-sm font-normal cursor-pointer dark:text-gray-300"
                          >
                            {formatSettingLabel(option)}
                          </Label>
                        </div>
                      ))}
                    </RadioGroup>
                  ) : inputType === 'color' ? (
                    <div className="flex items-center space-x-3">
                      <Input
                        id={controlId}
                        type="color"
                        value={controlValue as string}
                        className="w-10 h-10 p-0 border-none rounded cursor-pointer bg-transparent"
                        onChange={(e) => updateSetting(key, e.target.value)}
                      />
                      <span className="text-sm font-mono dark:text-gray-300">
                        {controlValue as string}
                      </span>
                    </div>
                  ) : inputType === 'number' ? (
                    <div className="flex flex-col w-full space-y-2">
                      <div className="flex items-center justify-between w-full">
                        <Slider
                          id={controlId}
                          value={[controlValue as number]} // Slider expects an array
                          min={sliderMin}
                          max={sliderMax}
                          step={sliderStep}
                          onValueChange={(value: number[]) => updateSetting(key, value[0])}
                          className="flex-1 mr-4" // Allow slider to take space
                        />
                        <Input
                          type="number"
                          value={controlValue as number}
                          className="w-20 text-right dark:bg-gray-800 dark:text-gray-200"
                          onChange={(e) => {
                            const newValue = parseFloat(e.target.value);
                            if (!isNaN(newValue)) {
                              updateSetting(key, newValue);
                            }
                          }}
                          step={sliderStep}
                          min={sliderMin}
                          max={sliderMax}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-muted-foreground dark:text-gray-500">
                        <span>{sliderMin}</span>
                        <span>{sliderMax}</span>
                      </div>
                    </div>
                  ) : ( // Default to text input
                    <Input
                      id={controlId}
                      type="text" // Always text for non-specific strings
                      value={controlValue as string}
                      className="flex-1 dark:bg-gray-800 dark:text-gray-200 dark:border-gray-700"
                      onChange={(e) => updateSetting(key, e.target.value)}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div> // Close the themed container div
  );
};

export default VisualizationPanel;
