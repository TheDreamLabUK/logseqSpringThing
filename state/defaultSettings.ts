import { Setting } from './SettingsStore';

export const defaultSettings: Record<string, Setting> = {
    'visualization/update_rate': {
        value: 60,
        type: 'number',
        description: 'Update rate in frames per second',
        min: 1,
        max: 120
    },
    'visualization/color_scheme': {
        value: 'default',
        type: 'string',
        description: 'Color scheme for visualization',
        options: ['default', 'dark', 'light']
    }
}; 