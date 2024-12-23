import React from 'react';
import { Select, FormControl, InputLabel, MenuItem } from '@mui/material';
import { useSettings } from '../../state/settings';

const FPS_OPTIONS = [1, 30, 60, 90];

export const FPSControl: React.FC = () => {
  const { settings, updateSettings } = useSettings();

  const handleChange = async (event: any) => {
    const newRate = event.target.value;
    try {
      await updateSettings('websocket', {
        updateRate: newRate
      });
    } catch (error) {
      console.error('Failed to update FPS:', error);
    }
  };

  return (
    <FormControl fullWidth>
      <InputLabel id="fps-select-label">Frame Rate</InputLabel>
      <Select
        labelId="fps-select-label"
        id="fps-select"
        value={settings.websocket.updateRate}
        label="Frame Rate"
        onChange={handleChange}
      >
        {FPS_OPTIONS.map((fps) => (
          <MenuItem key={fps} value={fps}>
            {fps} FPS
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};
