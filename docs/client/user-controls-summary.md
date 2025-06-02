# User Controls Summary - Settings Panel

## Intuitive Features Implemented

### 1. **Smart Control Type Selection**
The system automatically selects the most appropriate control type based on the `controlType` specified in the `UISettingDefinition` (from [`client/src/features/settings/config/settingsUIDefinition.ts`](../../client/src/features/settings/config/settingsUIDefinition.ts)) and the data type of the setting. Key control types rendered by [`SettingControlComponent.tsx`](../../client/src/features/settings/components/SettingControlComponent.tsx) include:

-   **`toggle`**: For boolean values (on/off settings) -> Renders a Switch.
-   **`slider`**: For numeric values with defined `min`, `max`, and `step` -> Renders a Slider.
-   **`numberInput`**: For general numeric values -> Renders a Number Input.
-   **`textInput`**: For string values.
-   **`passwordInput`**: A variant of `textInput` for sensitive string fields (API keys, secrets), providing masking.
-   **`select`**: For predefined options (enum-like strings) defined in `options` array in `UISettingDefinition` -> Renders a Select Dropdown.
-   **`colorPicker`**: For single string color values -> Renders a Color Picker with hex input.
-   **`rangeSlider`**: For `[number, number]` array values, representing a min/max range -> Renders a specialized Range Slider.
-   **`dualColorPicker`**: For `[string, string]` array values, representing two colors (e.g., for gradients) -> Renders two Color Pickers.
-   **`radioGroup`**: For selecting one option from a list (mutually exclusive choices) defined in `options` -> Renders a Radio Group.
-   **`buttonAction`**: For triggering an action (e.g., reset a section, trigger a backend process) -> Renders a Button. The action is defined by `actionId` in `UISettingDefinition`.

### 2. **User Experience Enhancements**

#### Visual Feedback
- **Live Value Display** - Shows current value next to sliders with appropriate decimal places
- **Unit Display** - Shows units (px, ms, etc.) where applicable
- **Hover Effects** - Subtle background highlight on hover for better interactivity
- **Tooltips** - Info icons with descriptions for complex settings

#### Input Handling
- **Debounced Inputs** - 300ms delay prevents excessive updates while typing
- **Validation** - Color inputs validate hex format and auto-correct invalid entries
- **Password Visibility Toggle** - Eye icon to show/hide sensitive values
- **Placeholder Text** - Contextual hints for input fields

#### Layout & Styling
- **Responsive Design** - Controls adapt to available space
- **Consistent Spacing** - Proper padding and margins for readability
- **Visual Hierarchy** - Clear label/control separation
- **Smooth Transitions** - CSS transitions for hover states

### 3. **Task-Appropriate Features**

#### For Visualization Settings
- **Real-time Updates** - Changes to visualization settings update the viewport immediately
- **Slider Preference** - Numeric inputs with ranges automatically use sliders for easier adjustment
- **Precise Control** - Step values configured appropriately (0.01 for decimals, 1 for integers)

#### For Security/Authentication
- **Automatic Masking** - API keys, secrets, and tokens are masked by default
- **Secure Placeholders** - "Enter secure value" for sensitive fields
- **Power User Gating** - Advanced settings only visible to authenticated power users

#### For Color Settings
- **Dual Input** - Both visual picker and text input for flexibility
- **Validation** - Ensures only valid hex colors are saved
- **Fallback Values** - Defaults to black (#000000) if invalid

### 4. **Accessibility Features**
- **Proper Labels** - All controls have associated labels
- **Keyboard Navigation** - Full keyboard support for all controls
- **ARIA Attributes** - Proper IDs and relationships
- **Focus Indicators** - Clear focus states for keyboard users

## Control Types by Use Case

### Basic Settings
- Enable/Disable features → **Toggle Switch**
- Adjust sizes/distances → **Slider with value display**
- Enter text/names → **Text Input with placeholder**

### Advanced Settings
- API Configuration → **Password Input with visibility toggle**
- Color Themes → **Color Picker with hex validation**
- Performance Ranges → **Range Slider for min/max**
- Display Modes → **Select Dropdown**

### Power User Settings
- Debug Options → **Hidden unless authenticated**
- Advanced XR Settings → **Gated by Nostr auth**
- AI Model Parameters → **Only visible to power users**

## Implementation Details

The controls are implemented in [`SettingControlComponent.tsx`](../../client/src/features/settings/components/SettingControlComponent.tsx) with:
- React hooks for state management (getting/setting values via `useSettingsStore`).
- Logic to determine the appropriate UI control based on `UISettingDefinition`.
- Custom debounce hook for input optimization on text/number inputs.
- TypeScript for type safety.
- Tailwind CSS for consistent styling.
- Lucide React icons for visual elements (e.g., tooltips, password visibility).

All controls follow the same pattern:
1. Receive value and onChange from parent
2. Manage local state for debouncing if needed
3. Validate input before calling onChange
4. Provide appropriate visual feedback