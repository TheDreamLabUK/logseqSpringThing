# UI Modernization Integration Guide

## Overview
All five phases of UI modernization have been successfully implemented. This guide provides instructions for integrating all components together.

## Required Dependencies

Add these dependencies to package.json:
```json
{
  "dependencies": {
    "react-window": "^1.8.10",
    "@types/react-window": "^1.8.8",
    "fuse.js": "^7.0.0"
  }
}
```

Run: `npm install react-window @types/react-window fuse.js`

## Integration Steps

### 1. Update App.tsx

```tsx
import { AppInitializer } from "./components/AppInitializer"
import { ThemeProvider } from "./features/design-system/ThemeProvider"
import { OnboardingProvider } from "./features/onboarding/OnboardingProvider"
import { HelpProvider } from "./features/help/HelpProvider"
import { CommandPaletteProvider } from "./features/command-palette/useCommandPalette"
import { Toaster } from "./ui/toaster"

export function App() {
  return (
    <ThemeProvider>
      <OnboardingProvider>
        <HelpProvider>
          <CommandPaletteProvider>
            <AppInitializer />
            <Toaster />
          </CommandPaletteProvider>
        </HelpProvider>
      </OnboardingProvider>
    </ThemeProvider>
  )
}
```

### 2. Replace Settings Panel

In any component importing SettingsPanelRedesign:

```tsx
// Replace this:
import { SettingsPanelRedesign } from "./features/settings/components/panels/SettingsPanelRedesign"

// With this:
import { SettingsPanelRedesignOptimized } from "./features/settings/components/panels/SettingsPanelRedesignOptimized"
```

### 3. Enable Keyboard Shortcuts

The keyboard shortcuts are automatically registered. Press `Shift+?` to see all available shortcuts.

### 4. Use Design System Components

Replace basic HTML elements with design system components:

```tsx
// Instead of:
<button onClick={handleClick}>Click me</button>

// Use:
import { Button } from "@/features/design-system/components/Button"
<Button onClick={handleClick} variant="primary">Click me</Button>
```

### 5. Add Loading States

```tsx
import { useAsyncOperation } from "@/hooks/useAsyncOperation"

const { execute, isLoading } = useAsyncOperation(async () => {
  // Your async operation
})

<Button onClick={execute} loading={isLoading}>
  Save Settings
</Button>
```

### 6. Use Command Palette

The command palette is available globally via `Ctrl+K`. To add custom commands:

```tsx
import { useCommandRegistry } from "@/features/command-palette/CommandRegistry"

const registry = useCommandRegistry()
registry.registerCommand({
  id: "my-action",
  name: "My Custom Action",
  shortcut: "Ctrl+M",
  action: () => console.log("Custom action!"),
  keywords: ["custom", "action"]
})
```

### 7. Add Help Content

```tsx
import { HelpTooltip } from "@/features/help/HelpTooltip"

<HelpTooltip
  content="This setting controls the size of nodes in the graph"
  helpKey="node-size"
>
  <Label>Node Size</Label>
</HelpTooltip>
```

### 8. Enable Undo/Redo

Undo/redo is automatically enabled for settings. Use `Ctrl+Z` to undo and `Ctrl+Shift+Z` to redo.

### 9. Accessibility Best Practices

Always include ARIA labels and keyboard support:

```tsx
<Button
  aria-label="Save changes"
  aria-busy={isLoading}
  onKeyDown={(e) => e.key === "Enter" && handleSave()}
>
  Save
</Button>
```

## Testing the Integration

1. **Performance**: Open DevTools Performance tab and record interactions
2. **Accessibility**: Use axe DevTools extension or screen reader
3. **Keyboard Navigation**: Navigate entire UI without mouse
4. **Theme Switching**: Toggle between light/dark themes
5. **Command Palette**: Try all commands via Ctrl+K
6. **Onboarding**: Clear localStorage and refresh to see onboarding

## Migration Checklist

- [ ] Install required dependencies
- [ ] Update App.tsx with providers
- [ ] Replace SettingsPanelRedesign imports
- [ ] Test keyboard shortcuts (Shift+?)
- [ ] Verify command palette (Ctrl+K)
- [ ] Check theme switching
- [ ] Test undo/redo in settings
- [ ] Verify loading states
- [ ] Test with screen reader
- [ ] Run performance profiler

## Troubleshooting

### Issue: Command palette not opening
- Ensure CommandPaletteProvider wraps your app
- Check for conflicting keyboard shortcuts

### Issue: Styles not applying
- Verify ThemeProvider is at root level
- Check that CSS variables are loading

### Issue: Performance degradation
- Use React DevTools Profiler
- Check for missing React.memo
- Verify virtualization is working

## Next Steps

1. Gradually migrate all components to use design system
2. Add more commands to command palette
3. Expand help content for all features
4. Create custom themes
5. Add more animation variants

All modernization phases are now complete and ready for production use!