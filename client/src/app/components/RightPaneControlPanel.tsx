import React from 'react';
import { SettingsPanelRedesign } from '../../features/settings/components/panels/SettingsPanelRedesign';
import NostrAuthSection from '../../features/auth/components/NostrAuthSection';

const RightPaneControlPanel: React.FC = () => {
  return (
    <div className="h-full flex flex-col bg-background">
      {/* Auth Section */}
      <div className="p-4 border-b">
        <NostrAuthSection />
      </div>
      
      {/* Settings Panel */}
      <div className="flex-1 overflow-hidden">
        <SettingsPanelRedesign />
      </div>
    </div>
  );
};

export default RightPaneControlPanel;