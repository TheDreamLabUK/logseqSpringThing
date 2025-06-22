# UI Component Library

## Overview

The LogseqSpringThing client includes a comprehensive UI component library built with React and TypeScript. These components provide consistent styling and behavior across the application, with support for theming and responsive design.

## Component Categories

### Core UI Components

#### Button ([`client/src/ui/Button.tsx`](../../client/src/ui/Button.tsx))
A versatile button component with multiple variants and sizes.
- **Variants**: default, destructive, outline, secondary, ghost, link
- **Sizes**: default, sm, lg, icon
- **Features**: Loading states, disabled states, full-width option
- **Usage**:
```tsx
<Button variant="default" size="sm" onClick={handleClick}>
  Click me
</Button>
```

#### Input ([`client/src/ui/Input.tsx`](../../client/src/ui/Input.tsx))
Standard input field component with consistent styling.
- **Features**: Error states, disabled states, placeholder support
- **Integrates with form validation libraries
- **Usage**:
```tsx
<Input 
  type="text" 
  placeholder="Enter text..." 
  value={value} 
  onChange={onChange} 
/>
```

#### Label ([`client/src/ui/Label.tsx`](../../client/src/ui/Label.tsx))
Form label component for accessibility and consistent styling.
- **Features**: Required field indicators, error states
- **Automatically associates with form controls
- **Usage**:
```tsx
<Label htmlFor="input-id">Field Label</Label>
```

#### Select ([`client/src/ui/Select.tsx`](../../client/src/ui/Select.tsx))
Dropdown selection component with customizable options.
- **Features**: Single/multiple selection, searchable options
- **Keyboard navigation support
- **Usage**:
```tsx
<Select value={selected} onValueChange={setSelected}>
  <SelectTrigger>
    <SelectValue placeholder="Select an option" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="option1">Option 1</SelectItem>
  </SelectContent>
</Select>
```

#### Switch ([`client/src/ui/Switch.tsx`](../../client/src/ui/Switch.tsx))
Toggle switch component for boolean settings.
- **Features**: Animated transitions, disabled states
- **Accessible with keyboard navigation
- **Usage**:
```tsx
<Switch 
  checked={isEnabled} 
  onCheckedChange={setIsEnabled} 
/>
```

#### RadioGroup ([`client/src/ui/RadioGroup.tsx`](../../client/src/ui/RadioGroup.tsx))
Radio button group for single selection from multiple options.
- **Features**: Grouped radio buttons with consistent styling
- **Keyboard navigation between options
- **Usage**:
```tsx
<RadioGroup value={selected} onValueChange={setSelected}>
  <RadioGroupItem value="option1" id="r1" />
  <Label htmlFor="r1">Option 1</Label>
</RadioGroup>
```

#### Slider ([`client/src/ui/Slider.tsx`](../../client/src/ui/Slider.tsx))
Range slider component for numeric value selection.
- **Features**: Min/max bounds, step increments
- **Visual feedback during interaction
- **Usage**:
```tsx
<Slider 
  value={[value]} 
  onValueChange={([v]) => setValue(v)}
  min={0} 
  max={100} 
  step={1} 
/>
```

### Layout Components

#### Card ([`client/src/ui/Card.tsx`](../../client/src/ui/Card.tsx))
Container component for grouping related content.
- **Sub-components**: CardHeader, CardTitle, CardDescription, CardContent, CardFooter
- **Features**: Consistent padding and shadows
- **Usage**:
```tsx
<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>Card description</CardDescription>
  </CardHeader>
  <CardContent>
    Content goes here
  </CardContent>
</Card>
```

#### Tabs ([`client/src/ui/Tabs.tsx`](../../client/src/ui/Tabs.tsx))
Tabbed interface component for organizing content.
- **Sub-components**: TabsList, TabsTrigger, TabsContent
- **Features**: Keyboard navigation, animated transitions
- **Usage**:
```tsx
<Tabs value={activeTab} onValueChange={setActiveTab}>
  <TabsList>
    <TabsTrigger value="tab1">Tab 1</TabsTrigger>
    <TabsTrigger value="tab2">Tab 2</TabsTrigger>
  </TabsList>
  <TabsContent value="tab1">Tab 1 content</TabsContent>
  <TabsContent value="tab2">Tab 2 content</TabsContent>
</Tabs>
```

#### Collapsible ([`client/src/ui/Collapsible.tsx`](../../client/src/ui/Collapsible.tsx))
Expandable/collapsible content container.
- **Features**: Smooth animations, controlled/uncontrolled modes
- **Accessibility support with ARIA attributes
- **Usage**:
```tsx
<Collapsible open={isOpen} onOpenChange={setIsOpen}>
  <CollapsibleTrigger>Toggle Content</CollapsibleTrigger>
  <CollapsibleContent>
    Hidden content revealed when open
  </CollapsibleContent>
</Collapsible>
```

### Feedback Components

#### Toast ([`client/src/ui/Toast.tsx`](../../client/src/ui/Toast.tsx))
Notification system for user feedback.
- **Hook**: `useToast` for programmatic toast creation
- **Features**: Multiple toast types (success, error, warning, info)
- **Auto-dismiss with configurable duration
- **Usage**:
```tsx
const { toast } = useToast();

toast({
  title: "Success",
  description: "Operation completed successfully",
  variant: "success"
});
```

#### Tooltip ([`client/src/ui/Tooltip.tsx`](../../client/src/ui/Tooltip.tsx))
Contextual information on hover/focus.
- **Features**: Customizable placement, delay options
- **Keyboard accessible
- **Usage**:
```tsx
<Tooltip>
  <TooltipTrigger>Hover me</TooltipTrigger>
  <TooltipContent>
    Helpful information appears here
  </TooltipContent>
</Tooltip>
```

### Theme Components

#### ThemeProvider ([`client/src/ui/ThemeProvider.tsx`](../../client/src/ui/ThemeProvider.tsx))
Context provider for application theming.
- **Features**: Light/dark mode support
- **System preference detection
- **Persistent theme selection
- **Usage**: Wrap your app root with ThemeProvider

#### ThemeSelector ([`client/src/ui/ThemeSelector.tsx`](../../client/src/ui/ThemeSelector.tsx))
UI component for theme selection.
- **Features**: Dropdown with theme options
- **Preview of theme colors
- **Instant theme switching

### Specialized Components

#### MarkdownRenderer ([`client/src/ui/markdown/MarkdownRenderer.tsx`](../../client/src/ui/markdown/MarkdownRenderer.tsx))
Renders markdown content with custom styling.
- **Features**: 
  - Syntax highlighting for code blocks
  - Custom link handling
  - Table support
  - Embedded media rendering
- **Security**: Sanitized HTML output
- **Usage**:
```tsx
<MarkdownRenderer content={markdownString} />
```

#### FormGroup ([`client/src/ui/formGroup/FormGroup.tsx`](../../client/src/ui/formGroup/FormGroup.tsx))
Wrapper component for form field grouping.
- **Features**: 
  - Consistent spacing between form elements
  - Error message display
  - Help text support
- **Usage**:
```tsx
<FormGroup>
  <Label>Field Name</Label>
  <Input {...inputProps} />
  <FormDescription>Help text here</FormDescription>
  <FormMessage />
</FormGroup>
```

## Application-Specific Components

### ConversationPane ([`client/src/app/components/ConversationPane.tsx`](../../client/src/app/components/ConversationPane.tsx))
Chat interface for AI interactions.
- **Features**:
  - Message history display
  - Input field with send button
  - Loading states during AI responses
  - Support for different AI providers (RAGFlow, Perplexity)

### MarkdownDisplayPanel ([`client/src/app/components/MarkdownDisplayPanel.tsx`](../../client/src/app/components/MarkdownDisplayPanel.tsx))
Panel for displaying markdown content from selected nodes.
- **Features**:
  - File content display
  - Metadata visualization
  - Scroll synchronization

### NarrativeGoldminePanel ([`client/src/app/components/NarrativeGoldminePanel.tsx`](../../client/src/app/components/NarrativeGoldminePanel.tsx))
Specialized panel for narrative exploration features.
- **Features**:
  - Content discovery tools
  - Narrative thread visualization
  - Interactive exploration controls

### RightPaneControlPanel ([`client/src/app/components/RightPaneControlPanel.tsx`](../../client/src/app/components/RightPaneControlPanel.tsx))
Main control panel housing settings and authentication.
- **Features**:
  - Collapsible sections
  - Authentication status display
  - Quick access to common settings

## Design System

### Colors
The component library uses CSS variables for theming:
- `--primary`: Primary brand color
- `--secondary`: Secondary brand color
- `--background`: Background colors
- `--foreground`: Text colors
- `--muted`: Muted elements
- `--accent`: Accent colors
- `--destructive`: Error/danger colors

### Typography
Consistent typography using system font stacks:
- Headings: Inter, system-ui, sans-serif
- Body: Inter, system-ui, sans-serif
- Code: Fira Code, monospace

### Spacing
Standardized spacing scale:
- `space-1`: 0.25rem (4px)
- `space-2`: 0.5rem (8px)
- `space-3`: 0.75rem (12px)
- `space-4`: 1rem (16px)
- `space-6`: 1.5rem (24px)
- `space-8`: 2rem (32px)

### Responsive Design
All components support responsive design through:
- Flexible layouts using CSS Grid and Flexbox
- Responsive typography scaling
- Touch-friendly interaction areas on mobile
- Adaptive component behavior based on screen size

## Best Practices

1. **Accessibility**: All components follow WCAG 2.1 guidelines
2. **Keyboard Navigation**: Full keyboard support for all interactive elements
3. **Performance**: Components use React.memo and proper dependency arrays
4. **Type Safety**: Full TypeScript support with exported prop types
5. **Composition**: Prefer composition over configuration for flexibility
6. **Theming**: Use CSS variables for easy theme customization

## Usage Guidelines

### Importing Components
```tsx
import { Button, Card, Input } from '@/ui';
```

### Styling Components
Components accept standard React props including `className` for custom styling:
```tsx
<Button className="custom-class" style={{ marginTop: '1rem' }}>
  Custom Styled Button
</Button>
```

### Form Integration
Components work seamlessly with form libraries like react-hook-form:
```tsx
<Controller
  control={control}
  name="fieldName"
  render={({ field }) => (
    <Input {...field} placeholder="Controlled input" />
  )}
/>
```