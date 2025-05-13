import React, { CSSProperties, useState, createContext, useContext } from 'react';
import { settingsUIDefinition, UICategoryDefinition } from '../../features/settings/config/settingsUIDefinition';
import { SettingsSection } from '../../features/settings/components/SettingsSection';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/ui/Collapsible';
import { Button } from '@/ui/Button';
import { Eye, Settings as SettingsIcon, Smartphone, Brain, ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react'; // Added ShieldCheck for Auth
import NostrAuthSection from '../../features/auth/components/NostrAuthSection';


// Simplified Context for advancedMode, similar to control-panel-context.tsx
interface ControlPanelContextType {
  advancedMode: boolean;
  toggleAdvancedMode: () => void;
  // Add other context values if SettingsSection depends on them, e.g., detachedSections
}
const ControlPanelContext = createContext<ControlPanelContextType | undefined>(undefined);

export const useControlPanelContext = () => {
  const context = useContext(ControlPanelContext);
  if (!context) {
    throw new Error('useControlPanelContext must be used within a ControlPanelProvider');
  }
  return context;
};

// Map icon names from settingsUIDefinition to Lucide components
const iconMap: { [key: string]: React.ElementType } = {
  Eye: Eye,
  Settings: SettingsIcon,
  Smartphone: Smartphone,
  Brain: Brain,
  ShieldCheck: ShieldCheck, // Added for Auth
  // Add other icons as needed
};


const RightPaneControlPanel: React.FC = () => {
  const [advancedMode, setAdvancedMode] = useState(false);
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({
    visualisation: true, // Default Visualisation to open
  });

  const toggleAdvancedMode = () => setAdvancedMode(prev => !prev);

  const toggleCategory = (categoryKey: string) => {
    setOpenCategories(prev => ({ ...prev, [categoryKey]: !prev[categoryKey] }));
  };

  const panelStyle: CSSProperties = {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
    padding: '0px', // Padding will be handled by inner elements or sections
    boxSizing: 'border-box',
    backgroundColor: '#ffffff', // Main panel background
  };

  const categoryHeaderStyle: CSSProperties = {
    padding: '8px 12px',
    borderBottom: '1px solid #e5e7eb', // tailwind gray-200
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f9fafb', // tailwind gray-50
  };

  const categoryTitleStyle: CSSProperties = {
    fontSize: '1em',
    fontWeight: '600', // semibold
    color: '#1f2937', // tailwind gray-800
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const categoryContentStyle: CSSProperties = {
    padding: '12px', // Padding for content within a category
    borderBottom: '1px solid #e5e7eb',
  };
  
  const authSectionStyle: CSSProperties = {
    padding: '12px',
    borderBottom: '1px solid #e5e7eb',
  };


  return (
    <ControlPanelContext.Provider value={{ advancedMode, toggleAdvancedMode }}>
      <div style={panelStyle} className="custom-scrollbar">
        {/* Auth Section - Always visible at the top */}
        <div style={authSectionStyle}>
            <NostrAuthSection />
        </div>

        {/* Advanced Mode Toggle - Placed strategically, e.g., at the top or bottom */}
        <div style={{ padding: '12px', borderBottom: '1px solid #e5e7eb', backgroundColor: '#f9fafb' }}>
            <label htmlFor="advancedModeToggle" className="flex items-center cursor-pointer">
                <input
                    type="checkbox"
                    id="advancedModeToggle"
                    checked={advancedMode}
                    onChange={toggleAdvancedMode}
                    className="mr-2 h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm font-medium text-gray-700">Show Advanced Settings</span>
            </label>
        </div>

        {Object.entries(settingsUIDefinition).map(([categoryKey, categoryDef]) => {
          const IconComponent = categoryDef.icon ? iconMap[categoryDef.icon] : SettingsIcon; // Default icon
          const isCategoryOpen = openCategories[categoryKey] ?? false;

          return (
            <Collapsible key={categoryKey} open={isCategoryOpen} onOpenChange={() => toggleCategory(categoryKey)} className="w-full">
              <CollapsibleTrigger asChild>
                <div style={categoryHeaderStyle} role="button" tabIndex={0} aria-expanded={isCategoryOpen}>
                  <span style={categoryTitleStyle}>
                    {IconComponent && <IconComponent size={16} />}
                    {categoryDef.label}
                  </span>
                  {isCategoryOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div style={categoryContentStyle}>
                  {Object.entries(categoryDef.subsections).map(([subsectionKey, subsectionDef]) => (
                    <SettingsSection
                      key={subsectionKey}
                      id={`settings-${categoryKey}-${subsectionKey}`}
                      title={subsectionDef.label}
                      subsectionSettings={subsectionDef.settings}
                    />
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          );
        })}
      </div>
    </ControlPanelContext.Provider>
  );
};

export default RightPaneControlPanel;