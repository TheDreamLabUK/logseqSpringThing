import React, { useState, useEffect, useCallback } from 'react';
import { UISettingDefinition } from '../config/settingsUIDefinition'; // Import the new definition type
import { Label } from '@/ui/Label';
import { Slider } from '@/ui/Slider';
import { Switch } from '@/ui/Switch';
import { Input } from '@/ui/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/Select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/ui/Tooltip';
import { Button } from '@/ui/Button';
import { Info } from 'lucide-react';

// Simple inline useDebounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

// Define props based on the plan
export interface SettingControlProps {
  path: string;
  settingDef: UISettingDefinition;
  value: any;
  onChange: (value: any) => void;
}

export function SettingControlComponent({ path, settingDef, value, onChange }: SettingControlProps) {
  // State for debounced inputs
  const [inputValue, setInputValue] = useState(String(value ?? ''));
  const debouncedInputValue = useDebounce(inputValue, 300); // 300ms debounce

  // Update internal state when the external value changes
  useEffect(() => {
    // Only update if the debounced value isn't the source of the change
    // This prevents loops but might need refinement depending on useDebounce implementation
    if (String(value) !== inputValue) {
       if (settingDef.type === 'rangeSlider' || settingDef.type === 'dualColorPicker') {
         // For array types, handle string conversion carefully if needed, or maybe skip input state?
         // For now, let's assume direct value prop usage for sliders/pickers is better for arrays.
       } else {
         setInputValue(String(value ?? ''));
       }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, settingDef.type]); // Rerun if value or type changes

  // Effect to call onChange when debounced value changes
  useEffect(() => {
    if (settingDef.type === 'textInput' || settingDef.type === 'numberInput') {
      // Avoid calling onChange with the initial value or if it hasn't changed
      if (debouncedInputValue !== String(value ?? '')) {
        if (settingDef.type === 'numberInput') {
          const numValue = parseFloat(debouncedInputValue);
          if (!isNaN(numValue)) {
            onChange(numValue);
          }
        } else {
          onChange(debouncedInputValue);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedInputValue, settingDef.type, onChange]); // Depend on debounced value

  // Handler for immediate input changes (updates local state)
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  // Render appropriate control based on settingDef.type
  const renderControl = () => {
    switch (settingDef.type) {
      case 'toggle':
        return (
          <Switch
            id={path}
            checked={Boolean(value)}
            onCheckedChange={onChange}
          />
        );

      case 'slider':
        return (
          <div className="flex w-full items-center gap-3">
            <Slider
              id={path}
              value={[value as number]} // Slider expects an array
              min={settingDef.min ?? 0}
              max={settingDef.max ?? 1}
              step={settingDef.step ?? 0.01} // Sensible default step
              onValueChange={([val]) => onChange(val)}
              className="flex-1"
            />
            <span className="text-xs font-mono w-12 text-right">
              {(value as number)?.toFixed ? (value as number).toFixed(settingDef.step && settingDef.step < 1 ? 2 : 0) : value}
              {settingDef.unit}
            </span>
          </div>
        );

      case 'numberInput':
        // If min and max are defined, prefer Slider for a more intuitive UI
        if (typeof settingDef.min === 'number' && typeof settingDef.max === 'number') {
          return (
            <div className="flex w-full items-center gap-3">
              <Slider
                id={path}
                value={[value as number]} // Slider expects an array
                min={settingDef.min}
                max={settingDef.max}
                step={settingDef.step ?? 0.01} // Default step for slider
                onValueChange={([val]) => onChange(val)} // Direct change
                className="flex-1"
              />
              <span className="text-xs font-mono w-12 text-right">
                {(value as number)?.toFixed ? (value as number).toFixed(settingDef.step && settingDef.step < 1 ? 2 : 0) : value}
                {settingDef.unit}
              </span>
            </div>
          );
        }
        // Fallback to Input if min/max not defined for slider behavior
        return (
          <div className="flex items-center w-full">
            <Input
              id={path}
              type="number"
              value={inputValue} // Use local state for debouncing
              onChange={handleInputChange} // Update local state immediately
              min={settingDef.min}
              max={settingDef.max}
              step={settingDef.step ?? 1} // Default step for input
              className="h-8 flex-1"
            />
            {settingDef.unit && <span className="text-xs text-muted-foreground pl-2">{settingDef.unit}</span>}
          </div>
        );

      case 'textInput':
        // Special handling for obscured fields like API keys
        const isSensitive = settingDef.label.toLowerCase().includes('key') || settingDef.label.toLowerCase().includes('secret');
        return (
          <div className="flex items-center w-full">
            <Input
              id={path}
              type={isSensitive ? "password" : "text"}
              value={inputValue} // Use local state for debouncing
              onChange={handleInputChange} // Update local state immediately
              className="h-8 flex-1" // Allow input to grow
            />
            {settingDef.unit && <span className="text-xs text-muted-foreground pl-2">{settingDef.unit}</span>}
          </div>
        );

      case 'colorPicker':
        return (
          <div className="flex items-center gap-2">
            <Input
              id={path}
              type="color"
              value={value as string}
              onChange={(e) => onChange(e.target.value)}
              className="h-8 w-10 p-0.5 border-border cursor-pointer" // Adjusted size and style
            />
            <Input
              type="text"
              value={value as string}
              onChange={(e) => onChange(e.target.value)}
              className="h-8 flex-1 font-mono text-xs" // Take remaining space, smaller font for hex
              placeholder="Hex"
            />
          </div>
        );

      case 'select':
        return (
          <Select
            value={String(value)} // Ensure value is string for Select
            onValueChange={(val) => onChange(val)} // Pass the string value back
          >
            <SelectTrigger id={path} className="h-8 w-full">
              <SelectValue placeholder={settingDef.label} />
            </SelectTrigger>
            <SelectContent>
              {settingDef.options?.map(opt => (
                <SelectItem key={String(opt.value)} value={String(opt.value)}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );

      case 'rangeSlider': { // For [number, number] arrays
        const [minVal, maxVal] = Array.isArray(value) ? value : [settingDef.min ?? 0, settingDef.max ?? 1];
        const handleMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
          const newMin = parseFloat(e.target.value);
          if (!isNaN(newMin)) {
            onChange([newMin, maxVal]);
          }
        };
        const handleMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
          const newMax = parseFloat(e.target.value);
          if (!isNaN(newMax)) {
            onChange([minVal, newMax]);
          }
        };
        return (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <Label htmlFor={`${path}-min`} className="text-xs w-10">Min:</Label>
              <Input id={`${path}-min`} type="number" value={minVal} onChange={handleMinChange} min={settingDef.min} max={maxVal} step={settingDef.step} className="h-8 flex-1" placeholder="Min" />
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor={`${path}-max`} className="text-xs w-10">Max:</Label>
              <Input id={`${path}-max`} type="number" value={maxVal} onChange={handleMaxChange} min={minVal} max={settingDef.max} step={settingDef.step} className="h-8 flex-1" placeholder="Max" />
            </div>
            {settingDef.unit && <span className="text-xs text-muted-foreground self-end">{settingDef.unit}</span>}
          </div>
        );
      }

      case 'dualColorPicker': { // For [string, string] color arrays
        const [color1, color2] = Array.isArray(value) ? value : ['#ffffff', '#000000'];
         const handleColor1Change = (e: React.ChangeEvent<HTMLInputElement>) => {
           onChange([e.target.value, color2]);
         };
         const handleColor2Change = (e: React.ChangeEvent<HTMLInputElement>) => {
           onChange([color1, e.target.value]);
         };
        return (
          <div className="flex items-center gap-2">
            <Input type="color" value={color1} onChange={handleColor1Change} className="h-8 w-10 p-0.5 border-border cursor-pointer" title="Start Color" />
            <Input type="color" value={color2} onChange={handleColor2Change} className="h-8 w-10 p-0.5 border-border cursor-pointer" title="End Color" />
            <span className="text-xs font-mono text-muted-foreground">{color1} / {color2}</span>
          </div>
        );
      }


      case 'buttonAction':
        return (
          <Button onClick={settingDef.action} size="sm" variant="outline">
            {settingDef.label} {/* Button text is the label */}
          </Button>
        );

      default:
        // Render value as string for unknown types
        return <span className="text-sm text-muted-foreground">{JSON.stringify(value)}</span>;
    }
  };

  // For button actions, the label is the button itself, so we don't need a separate label.
  if (settingDef.type === 'buttonAction') {
    return renderControl();
  }

  return (
    <div className="setting-control flex items-center justify-between gap-4 py-2 border-b border-border/50 last:border-b-0">
      <Label htmlFor={path} className="text-sm flex items-center gap-1 flex-shrink-0 max-w-[40%]"> {/* Limit label width */}
        <span>{settingDef.label}</span>
        {settingDef.description && (
          <TooltipProvider delayDuration={100}>
            <Tooltip content={settingDef.description} side="top" align="start">
              {/* The Info icon will be the trigger, Tooltip content is via prop */}
              <Info className="h-3 w-3 text-muted-foreground cursor-help" />
            </Tooltip>
          </TooltipProvider>
        )}
      </Label>
      <div className="flex-1 min-w-0"> {/* Allow control area to grow and shrink */}
        {renderControl()}
      </div>
    </div>
  );
}