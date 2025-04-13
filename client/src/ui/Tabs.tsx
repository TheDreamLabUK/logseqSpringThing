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
    <div className={cn('flex flex-col h-full', className)}>
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
              'flex items-center px-3 py-2 text-muted-foreground hover:text-foreground focus:outline-none whitespace-nowrap',
              tabButtonClassName,
              activeTab === index &&
                'border-b-2 border-primary font-medium text-foreground',
              activeTab === index && activeTabButtonClassName
            )}
            aria-selected={activeTab === index}
            role="tab"
          >
            {tab.icon && <span className="mr-2">{tab.icon}</span>}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div
        className={cn('flex-1 min-h-0 overflow-y-auto p-4 space-y-6', tabContentClassName)}
        role="tabpanel"
      >
        {tabs[activeTab]?.content}
      </div>
    </div>
  );
};

export default Tabs;
