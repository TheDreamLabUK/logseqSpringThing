import React, { useState, ReactNode } from 'react';
import { cn } from '../utils/cn'; // Assuming you have a utility for class names

interface Tab {
  label: string;
  content: ReactNode;
  icon?: ReactNode; // Optional icon for the tab
}

interface TabsProps {
  tabs: Tab[];
  initialTab?: number;
  className?: string;
  tabListClassName?: string;
  tabButtonClassName?: string;
  activeTabButtonClassName?: string;
  tabContentClassName?: string;
}

const Tabs: React.FC<TabsProps> = ({
  tabs,
  initialTab = 0,
  className,
  tabListClassName,
  tabButtonClassName,
  activeTabButtonClassName,
  tabContentClassName,
}) => {
  const [activeTab, setActiveTab] = useState(initialTab);

  if (!tabs || tabs.length === 0) {
    return null; // Don't render anything if no tabs are provided
  }

  return (
    // Added default dark theme classes
    <div className={cn('flex flex-col h-full bg-card text-card-foreground dark:bg-gray-900 dark:text-gray-100', className)}>
      {/* Tab List */}
      <div
        className={cn(
          'flex border-b border-border overflow-x-auto',
          tabListClassName
        )}
      >
        {tabs.map((tab, index) => (
          <button
            key={index}
            onClick={() => setActiveTab(index)}
            className={cn(
              // Base styles for dark theme button appearance
              'appearance-none border-none bg-transparent', // Remove default browser styling
              'flex items-center px-4 py-2', // Adjusted padding
              'text-sm font-medium', // Consistent font styling
              'text-muted-foreground hover:text-foreground', // Text colors
              'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background', // Focus ring for accessibility
              'whitespace-nowrap transition-colors duration-150 ease-in-out', // Smooth transition
              // Remove default bottom border, apply only to active
              tabButtonClassName, // Allow overrides
              // Active tab styles
              activeTab === index && 'border-b-2 border-primary text-foreground',
              activeTab === index && activeTabButtonClassName // Allow overrides for active state
            )}
            aria-selected={activeTab === index}
            role="tab"
          >
            {tab.icon && <span className="mr-2">{tab.icon}</span>}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content - Added default dark theme classes */}
      <div
        className={cn('flex-1 min-h-0 overflow-y-auto p-4 space-y-6 bg-card text-card-foreground dark:bg-gray-900 dark:text-gray-100', tabContentClassName)}
        role="tabpanel"
      >
        {tabs[activeTab]?.content}
      </div>
    </div>
  );
};

export default Tabs;
