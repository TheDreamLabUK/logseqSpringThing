import React from 'react';
import { useTheme } from './ThemeProvider';
import { Button } from './Button';
import { Check } from 'lucide-react';

const ThemeSelector = () => {
  const { theme, setTheme } = useTheme();

  return (
    <div className="space-y-4">
      <div className="flex flex-col space-y-2">
        <h3 className="text-sm font-medium">Theme</h3>
        <div className="flex flex-wrap gap-2">
          <Button
            variant={theme === 'light' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setTheme('light')}
            className="flex items-center"
          >
            Light
            {theme === 'light' && <Check className="h-3 w-3 ml-1" />}
          </Button>

          <Button
            variant={theme === 'dark' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setTheme('dark')}
            className="flex items-center"
          >
            Dark
            {theme === 'dark' && <Check className="h-3 w-3 ml-1" />}
          </Button>

          <Button
            variant={theme === 'system' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setTheme('system')}
            className="flex items-center"
          >
            System
            {theme === 'system' && <Check className="h-3 w-3 ml-1" />}
          </Button>
        </div>
      </div>

      <div className="mt-4 p-3 bg-muted rounded-md">
        <div className="text-sm text-muted-foreground">
          Theme changes are saved automatically and will persist between sessions.
        </div>
      </div>
    </div>
  );
};

export default ThemeSelector;