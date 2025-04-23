import React, { useEffect } from 'react';
import Tabs from '../../ui/Tabs';
import NostrAuthSection from '../../features/auth/components/NostrAuthSection';
import SystemPanel from '../../features/settings/components/panels/SystemPanel';
import VisualizationPanel from '../../features/settings/components/panels/VisualizationPanel';
import XRPanel from '../../features/settings/components/panels/XRPanel';
import AIPanel from '../../features/settings/components/panels/AIPanel';
import MarkdownRenderer from '../../ui/markdown/MarkdownRenderer'; // Assuming path is correct
import { Button } from '../../ui/Button'; // Assuming path is correct
// Lucide icons for tabs
import { Settings, Eye, Anchor, Send } from 'lucide-react'; // Using icons declared in types/lucide-react.d.ts
// Note: 'File' icon is not declared, using 'Settings' as placeholder for Markdown tab.

// Placeholder markdown content
const placeholderMarkdown = `
# Markdown Tab

This tab uses the \`MarkdownRenderer\` component.

*   Supports standard Markdown.
*   Includes syntax highlighting for code blocks.

\`\`\`javascript
// Example code block
function greet(name) {
  console.log(\`Hello, \${name}!\`);
}
greet('World');
\`\`\`

Visit [Narrative Gold Mine](https://narrativegoldmine.com).
`;

const LowerControlPanel: React.FC = () => {
  // Add custom scrollbar styles for WebKit browsers
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .custom-scrollbar::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }
      .custom-scrollbar::-webkit-scrollbar-track {
        background: #1F2937;
      }
      .custom-scrollbar::-webkit-scrollbar-thumb {
        background-color: #4B5563;
        border-radius: 4px;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);
  return (
    // Outer container for padding. Background color applied here.
    // Height/Scrolling is managed by the parent container in AppPage.
    // Removed h-full, parent container in AppPage manages height/scrolling. Keep flex flex-col for header + panes.
    <div className="container mx-auto px-4 py-6 bg-gray-900 text-white flex flex-col" style={{ backgroundColor: '#111827', color: 'white' }}>
      {/* Header */}
      <div className="mb-6 text-center flex-shrink-0"> {/* Prevent header from shrinking */}
        <h2 className="text-3xl font-bold mb-2">Control Panel</h2>
        <div className="h-1 w-24 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto"></div>
      </div>

      {/* Two-Pane Container */}
      {/* Uses flex row, card styling, and flex-1 to fill available vertical space */}
      {/* Ensure flex-1 is present so this container grows vertically */}
      <div className="flex flex-row bg-gray-800 rounded-lg overflow-hidden shadow-xl border border-gray-700 min-h-[300px] flex-1" style={{ display: 'flex', flexDirection: 'row', backgroundColor: '#1f2937', color: 'white' }}>

        {/* Left Pane: Settings Tabs */}
        {/* Uses flex-col to stack tab list and content */}
        <div className="w-1/2 h-full border-r border-gray-700 flex flex-col overflow-y-auto custom-scrollbar" style={{
          width: '50%',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '100%',
          overflowY: 'auto',
          scrollbarWidth: 'thin',
          scrollbarColor: '#4B5563 #1F2937'
        }}>
          <Tabs
            tabs={[
              { label: 'Auth', icon: <Settings className="h-4 w-4" />, content: <NostrAuthSection /> },
              { label: 'System', icon: <Settings className="h-4 w-4" />, content: <SystemPanel panelId="main-settings-system" /> },
              { label: 'Visualization', icon: <Eye className="h-4 w-4" />, content: <VisualizationPanel /> },
              { label: 'XR', icon: <Settings className="h-4 w-4" />, content: <XRPanel panelId="main-settings-xr" /> },
              { label: 'AI Services', icon: <Settings className="h-4 w-4" />, content: <AIPanel /> },
            ]}
            // Prevent tab list shrinking, add border
            tabListClassName="bg-gray-800 px-4 flex-shrink-0 border-b border-gray-700"
            tabButtonClassName="py-3"
            // Allow tab content to grow and scroll internally, add padding
            tabContentClassName="bg-gray-800 text-white flex-1 overflow-y-auto p-4 custom-scrollbar"
          />
        </div>

        {/* Right Pane: New Feature Tabs */}
        {/* Uses flex-col to stack tab list and content */}
        <div className="w-1/2 h-full flex flex-col bg-gray-800 overflow-y-auto custom-scrollbar" style={{
          width: '50%',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '100%',
          backgroundColor: '#1f2937',
          overflowY: 'auto',
          color: 'white',
          scrollbarWidth: 'thin',
          scrollbarColor: '#4B5563 #1F2937'
        }}>
          <Tabs
            tabs={[
              {
                label: 'Narrative Gold Mine',
                icon: <Anchor className="h-4 w-4" />, // Use Anchor icon (declared)
                content: (
                  // Container ensures iframe takes full height of the tab content area
                  <div className="w-full h-full flex flex-col">
                    <iframe
                      src="https://narrativegoldmine.com"
                      className="flex-1 border-none" // Simplified: flex-1 handles filling space
                      title="Narrative Gold Mine"
                      sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
                      loading="lazy"
                      referrerPolicy="no-referrer"
                    />
                  </div>
                ),
              },
              {
                label: 'Markdown',
                icon: <Settings className="h-4 w-4" />, // Use Settings icon (placeholder, File not declared)
                // Pass content, add padding via className to the renderer's wrapper
                content: <MarkdownRenderer content={placeholderMarkdown} className="p-4" />,
              },
              {
                label: 'LLM Query',
                icon: <Send className="h-4 w-4" />, // Use Send icon (declared)
                content: (
                  // Container ensures elements stack correctly and use full height
                  <div className="p-4 flex flex-col h-full">
                    {/* Using standard HTML textarea with Tailwind classes */}
                    <textarea
                      className="flex-1 mb-2 p-2 border border-border rounded bg-input text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary"
                      placeholder="Enter your query..."
                    />
                    <Button className="self-end">Send Query</Button>
                  </div>
                ),
              },
            ]}
            // Prevent tab list shrinking, add border
            tabListClassName="bg-gray-800 px-4 flex-shrink-0 border-b border-gray-700"
            tabButtonClassName="py-3"
            // Allow tab content to grow and scroll internally. Padding handled by individual content wrappers.
            tabContentClassName="bg-gray-800 text-white flex-1 overflow-y-auto custom-scrollbar"
          />
        </div>
      </div>
    </div>
  );
};

export default LowerControlPanel;