import { type ReactNode } from 'react';
import { createLogger } from '../../utils/logger';

const logger = createLogger('MainLayout');

interface MainLayoutProps {
  viewportContent: ReactNode;
  topDockContent?: ReactNode;
  rightDockContent?: ReactNode;
  overlays?: ReactNode; // Add overlays prop
}

const MainLayout = ({
  viewportContent,
  topDockContent,
  rightDockContent,
  overlays, // Destructure overlays prop
}: MainLayoutProps) => {
  return (
    <div 
      className="main-layout"
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        height: '100%',
        minHeight: '0', // Important for Firefox
        overflow: 'hidden'
      }}
    >
      {/* Top Dock */}
      {topDockContent}
      
      {/* Main Content Area (Viewport + Right Dock) */}
      <div 
        style={{
           display: 'flex',
           flex: '1 1 auto',
           minHeight: '0', // Important for Firefox
           // height: '100%', // Removed explicit height, rely on flex
           width: '100%',
           position: 'relative'
         }}
      >
         {/* Main Viewport */}
         <div
            style={{
              flex: '1 1 auto', // Grow horizontally
              minHeight: '0', // Important for Firefox
              position: 'relative',
              display: 'flex', // Restore display: flex
              flexDirection: 'column', // Restore flex-direction
              height: '100%' // Keep height 100%
            }}
          >
            {viewportContent}
          </div>

          {/* Render Overlays here, positioned absolutely relative to this container */}
          {overlays}

          {/* Right Dock */}
          {rightDockContent}
        </div>
      </div>
  );
};

export default MainLayout;
