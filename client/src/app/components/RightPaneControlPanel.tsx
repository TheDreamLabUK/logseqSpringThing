import React from 'react';
import { SettingsPanelRedesign } from '../../features/settings/components/panels/SettingsPanelRedesign';
import NostrAuthSection from '../../features/auth/components/NostrAuthSection';

interface RightPaneControlPanelProps {
  toggleLowerRightPaneDock: () => void;
  isLowerRightPaneDocked: boolean;
}

const RightPaneControlPanel: React.FC<RightPaneControlPanelProps> = ({ toggleLowerRightPaneDock, isLowerRightPaneDocked }) => {
  return (
    <div className="h-full flex flex-col bg-background">
      {/* Auth Section */}
      <div className="p-4 border-b">
        <NostrAuthSection />
      </div>
      
      {/* Settings Panel */}
      <div className="flex-1 overflow-hidden">
        <SettingsPanelRedesign toggleLowerRightPaneDock={toggleLowerRightPaneDock} isLowerRightPaneDocked={isLowerRightPaneDocked} />
      </div>
    </div>
  );
};

export default RightPaneControlPanel;