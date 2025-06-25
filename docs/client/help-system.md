# Help System

## Overview
The help system provides context-sensitive assistance throughout the application, including tooltips, detailed documentation, and interactive guides. It integrates with the onboarding system and command palette for a comprehensive support experience.

## Architecture

### Core Components

#### HelpRegistry (`client/src/features/help/HelpRegistry.ts`)
Central registry managing all help content and categories.

```typescript
class HelpRegistry {
  registerCategory(category: HelpCategory): void
  registerContent(categoryId: string, content: HelpContent): void
  getCategory(categoryId: string): HelpCategory | undefined
  getContent(contentId: string): HelpContent | undefined
  searchHelp(query: string): HelpContent[]
}
```

#### Help Types (`client/src/features/help/types.ts`)
```typescript
interface HelpContent {
  id: string                    // Unique identifier
  title: string                 // Help topic title
  description: string           // Brief description
  detailedHelp?: string        // Full documentation (Markdown)
  examples?: string[]          // Usage examples
  relatedTopics?: string[]     // Related help IDs
  videoUrl?: string            // Tutorial video link
}

interface HelpCategory {
  id: string                    // Category identifier
  name: string                  // Display name
  description: string           // Category description
  items: HelpContent[]         // Help items in category
}

interface TooltipConfig {
  showDelay?: number           // Default: 500ms
  hideDelay?: number           // Default: 200ms
  showHelpIndicator?: boolean  // Show (?) icon
  theme?: 'light' | 'dark' | 'auto'
}
```

### User Interface Components

#### HelpProvider (`client/src/features/help/components/HelpProvider.tsx`)
React context provider managing help state and functionality.

```typescript
interface HelpContextValue {
  showHelp: (contentId: string) => void
  hideHelp: () => void
  searchHelp: (query: string) => HelpContent[]
  getHelpForElement: (elementId: string) => HelpContent | undefined
  registerElementHelp: (elementId: string, helpId: string) => void
  activeHelp: HelpContent | null
  isHelpOpen: boolean
}
```

#### HelpTooltip (`client/src/features/help/components/HelpTooltip.tsx`)
Contextual tooltip component for inline help.

```typescript
interface HelpTooltipProps {
  helpId: string               // Help content ID
  children: React.ReactNode    // Wrapped element
  position?: 'top' | 'bottom' | 'left' | 'right'
  showIndicator?: boolean      // Show (?) icon
  trigger?: 'hover' | 'click' | 'focus'
}
```

## Help Content Categories

### Getting Started
- Application overview
- Basic navigation
- Core concepts
- First steps guide

### Visualization
- Graph controls
- Camera navigation
- Node interactions
- Edge management
- Layout algorithms

### Settings
- Settings panel overview
- Visualization options
- Performance tuning
- Appearance customization
- Keyboard shortcuts

### XR/VR Features
- WebXR setup
- Hand tracking
- Gesture controls
- Safety guidelines
- Device compatibility

### AI Features
- Chat interface
- Voice commands
- AI service configuration
- API key management

### Advanced Topics
- Command palette usage
- Data import/export
- Performance optimization
- Troubleshooting guide

## Usage Examples

### Adding Contextual Help
```typescript
import { HelpTooltip } from '@/features/help';

function SettingControl() {
  return (
    <HelpTooltip 
      helpId="settings.node-size" 
      position="right"
      showIndicator
    >
      <Slider 
        label="Node Size"
        value={nodeSize}
        onChange={setNodeSize}
      />
    </HelpTooltip>
  );
}
```

### Registering Help Content
```typescript
import { useHelp } from '@/features/help';

function MyFeature() {
  const { registerElementHelp } = useHelp();
  
  useEffect(() => {
    registerElementHelp('my-feature-button', 'help.my-feature');
  }, []);
}
```

### Programmatic Help Display
```typescript
const { showHelp } = useHelp();

const handleComplexAction = () => {
  // Perform action
  showHelp('help.complex-action-complete');
};
```

## Settings Help Integration

### Dynamic Help Content (`client/src/features/help/settingsHelp.ts`)
Automatically generated help for all settings:

```typescript
const settingsHelp = generateSettingsHelp(settingsSchema);
// Creates help entries for each setting with:
// - Description from schema
// - Valid ranges/options
// - Default values
// - Performance implications
```

### Performance Impact Indicators
- 🟢 Low impact
- 🟡 Medium impact
- 🔴 High impact
- ⚡ GPU accelerated

## Search Functionality

### Full-Text Search
- Searches titles, descriptions, and detailed help
- Fuzzy matching for typos
- Weighted results (title > description > content)
- Recent searches cached

### Context-Aware Suggestions
- Suggests help based on current view
- Tracks frequently accessed help
- Machine learning recommendations (planned)

## Accessibility Features

### Screen Reader Support
- ARIA labels for all help elements
- Keyboard navigation
- Announce help content changes

### Visual Accommodations
- High contrast mode
- Adjustable font sizes
- Dyslexia-friendly fonts option

## Multi-Language Support

### Content Structure
```typescript
interface LocalizedHelpContent extends HelpContent {
  locale: string
  translations: {
    [locale: string]: {
      title: string
      description: string
      detailedHelp?: string
    }
  }
}
```

### Language Detection
- Browser language preference
- User setting override
- Fallback to English

## Integration Points

### Command Palette
- "Search Help" command
- Quick help shortcuts
- Help command category

### Onboarding System
- Links to detailed help from tours
- "Learn More" options in steps
- Post-onboarding help suggestions

### Error Messages
- Contextual help for errors
- Troubleshooting guides
- Recovery suggestions

## Analytics

### Usage Tracking
- Most viewed help topics
- Search queries
- Time spent reading help
- Help effectiveness metrics

### Improvement Feedback
- "Was this helpful?" prompts
- Suggestion collection
- Missing topic reporting

## Performance Optimization

### Lazy Loading
- Help content loaded on demand
- Images lazy loaded
- Video thumbnails only

### Caching Strategy
- IndexedDB for offline access
- Service worker integration
- Delta updates for content

## Best Practices

1. **Content Writing**: Clear, concise, action-oriented
2. **Examples**: Include practical, real-world examples
3. **Visuals**: Use screenshots and diagrams
4. **Updates**: Keep help synchronized with features
5. **Feedback**: Actively collect and incorporate user feedback

## Future Enhancements

- Interactive tutorials within help
- AI-powered help chat
- Community-contributed help content
- Video tutorial integration
- Augmented reality help overlays