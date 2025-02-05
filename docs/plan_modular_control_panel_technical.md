# Technical Implementation Details for Modular Control Panel

## 1. Component Architecture

### 1.1 Core Components
```typescript
// ModularControlPanel
interface ModularControlPanelProps {
  sections: ControlSection[];
  layout: LayoutConfig;
  onLayoutChange: (newLayout: LayoutConfig) => void;
}

// ControlSection
interface ControlSection {
  id: string;
  title: string;
  settings: Setting[];
  isDetached: boolean;
  position?: { x: number, y: number };
  size?: { width: number, height: number };
}

// Setting
interface Setting {
  id: string;
  type: 'slider' | 'toggle' | 'color' | 'select';
  value: any;
  metadata: SettingMetadata;
}

// SettingMetadata
interface SettingMetadata {
  label: string;
  description: string;
  category: 'basic' | 'advanced';
  visualAid?: string; // Path to preview image/animation
  dependencies?: string[]; // IDs of related settings
}
```

## 2. State Management

### 2.1 Layout Management
```typescript
interface LayoutConfig {
  sections: {
    [sectionId: string]: {
      position: { x: number, y: number };
      size: { width: number, height: number };
      isDetached: boolean;
      isCollapsed: boolean;
    };
  };
  userPreferences: {
    showAdvanced: boolean;
    activeFilters: string[];
    customOrder: string[]; // Section IDs in user-defined order
  };
}

class LayoutManager {
  private layout: LayoutConfig;
  private observers: Set<(layout: LayoutConfig) => void>;

  updateSectionPosition(sectionId: string, position: Position): void;
  updateSectionSize(sectionId: string, size: Size): void;
  toggleDetached(sectionId: string): void;
  saveLayout(): void;
  loadLayout(): void;
}
```

### 2.2 Real-time Preview Integration
```typescript
class PreviewManager {
  private previewTimeoutId: number | null = null;
  private readonly PREVIEW_DELAY = 16; // ~60fps

  // Debounced update to prevent overwhelming the 3D renderer
  updatePreview(setting: Setting, value: any): void {
    if (this.previewTimeoutId) {
      clearTimeout(this.previewTimeoutId);
    }
    
    this.previewTimeoutId = setTimeout(() => {
      this.applyPreview(setting, value);
      this.previewTimeoutId = null;
    }, this.PREVIEW_DELAY);
  }

  private applyPreview(setting: Setting, value: any): void {
    // Integration with existing visualization controller
    VisualizationController.getInstance().updateSetting(setting.id, value);
  }
}
```

## 3. Drag and Drop Implementation

### 3.1 Custom DragManager
```typescript
class DragManager {
  private draggingElement: HTMLElement | null = null;
  private offset: { x: number, y: number } = { x: 0, y: 0 };

  initialize(container: HTMLElement): void {
    container.addEventListener('mousedown', this.handleDragStart);
    document.addEventListener('mousemove', this.handleDrag);
    document.addEventListener('mouseup', this.handleDragEnd);
  }

  private handleDragStart = (e: MouseEvent): void => {
    if (!(e.target as HTMLElement).classList.contains('draggable')) return;
    
    this.draggingElement = e.target as HTMLElement;
    const rect = this.draggingElement.getBoundingClientRect();
    this.offset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  private handleDrag = (e: MouseEvent): void => {
    if (!this.draggingElement) return;

    const x = e.clientX - this.offset.x;
    const y = e.clientY - this.offset.y;
    
    this.draggingElement.style.transform = `translate(${x}px, ${y}px)`;
  };
}
```

## 4. Dynamic Tooltips

### 4.1 Tooltip Component
```typescript
interface TooltipProps {
  setting: Setting;
  position: { x: number, y: number };
}

class TooltipManager {
  private tooltipElement: HTMLElement;
  private currentSetting: Setting | null = null;

  showTooltip(props: TooltipProps): void {
    this.tooltipElement.innerHTML = this.generateTooltipContent(props.setting);
    this.position(props.position);
    this.tooltipElement.classList.add('visible');
  }

  private generateTooltipContent(setting: Setting): string {
    return `
      <div class="tooltip-content">
        <h4>${setting.metadata.label}</h4>
        <p>${setting.metadata.description}</p>
        ${this.getVisualAidHTML(setting)}
        ${this.getDependenciesHTML(setting)}
      </div>
    `;
  }
}
```

## 5. Performance Considerations

### 5.1 Optimization Strategies
- Use ResizeObserver for efficient size tracking
- Implement virtual scrolling for large setting lists
- Debounce real-time preview updates
- Use CSS transforms for smooth animations
- Implement lazy loading for visual aids

### 5.2 Memory Management
```typescript
class ControlPanelManager {
  private readonly observers: WeakMap<HTMLElement, ResizeObserver>;
  private readonly eventListeners: WeakMap<HTMLElement, Function[]>;

  dispose(): void {
    this.observers.forEach((observer) => observer.disconnect());
    this.eventListeners.forEach((listeners, element) => {
      listeners.forEach(listener => element.removeEventListener('click', listener));
    });
  }
}
```

## 6. Integration Points

### 6.1 Required Changes
1. Modify `SettingsStore` to support real-time updates
2. Update `VisualizationController` to handle immediate preview rendering
3. Extend `SettingsObserver` to track UI-specific state
4. Add persistence layer for layout configurations

### 6.2 Migration Strategy
1. Implement new components alongside existing ones
2. Gradually transition settings to new system
3. Add feature flags for incremental rollout
4. Maintain backwards compatibility during transition