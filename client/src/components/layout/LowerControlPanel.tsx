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

Visit [Narrative Gold Mine](https://narrativegoldmine.com/#/graph).

## Extra Content to Force Scrolling

${'- This is a repeated line to force scrolling\n'.repeat(50)}

### More Content

${'- Another repeated line with different text\n'.repeat(50)}
`;

const LowerControlPanel: React.FC = () => {
  // Add custom scrollbar styles and smooth scrolling behavior
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      /* Custom scrollbar styling */
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

      /* Improve scrolling behavior */
      .custom-scrollbar {
        scroll-behavior: smooth;
        scrollbar-gutter: stable;
      }

      /* Ensure tab content areas are scrollable */
      .left-pane-content, .right-pane-content {
        overflow-y: auto;
        max-height: 100%;
        height: 100%;
      }

      /* Ensure content fills available space */
      .tab-content-container {
        display: flex;
        flex-direction: column;
        height: 100%;
        padding: 16px;
      }

      /* Special styling for the iframe container */
      .iframe-container {
        padding: 0;
        height: 100%;
        overflow-y: auto;
      }

      /* Style for iframes */
      iframe {
        border: none;
        border-radius: 0; /* Full width, no rounded corners */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
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
      <div className="flex flex-row bg-gray-800 rounded-lg overflow-hidden shadow-xl border border-gray-700 min-h-[300px]" style={{ display: 'flex', flexDirection: 'row', backgroundColor: '#1f2937', color: 'white' }}>

        {/* Left Pane: Settings Tabs - 35% width */}
        {/* Uses flex-col to stack tab list and content */}
        <div className="w-[35%] border-r border-gray-700 flex flex-col overflow-y-auto custom-scrollbar" style={{
          width: '35%',
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
          scrollbarWidth: 'thin',
          scrollbarColor: '#4B5563 #1F2937'
        }}>
          {/* Removed manual scroll buttons - using mouse wheel instead */}
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
            // Allow tab content to grow and scroll internally. Padding handled by tab-content-container class.
            tabContentClassName="bg-gray-800 text-white custom-scrollbar left-pane-content overflow-y-auto"
          />
        </div>

        {/* Right Pane: New Feature Tabs - 65% width */}
        {/* Uses flex-col to stack tab list and content */}
        <div className="w-[65%] flex flex-col bg-gray-800 overflow-y-auto custom-scrollbar" style={{
          width: '65%',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#1f2937',
          overflowY: 'auto',
          color: 'white',
          scrollbarWidth: 'thin',
          scrollbarColor: '#4B5563 #1F2937'
        }}>
          {/* Removed manual scroll buttons - using mouse wheel instead */}
          <Tabs
            tabs={[
              {
                label: 'Narrative Gold Mine',
                icon: <Anchor className="h-4 w-4" />, // Use Anchor icon (declared)
                content: (
                  // Container ensures iframe takes full height of the tab content area
                  <div className="w-full h-full flex flex-col tab-content-container iframe-container"> {/* Using special iframe container class */}
                    {/* Full-width iframe that renders off the bottom with scrolling */}
                    <iframe
                      src="https://narrativegoldmine.com"
                      className="w-full h-full border-none"
                      title="Narrative Gold Mine"
                      sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
                      loading="lazy"
                      referrerPolicy="no-referrer"
                      style={{ height: '1200px' }} /* Tall enough to ensure scrolling is needed */
                    />
                  </div>
                ),
              },
              {
                label: 'Markdown',
                icon: <Settings className="h-4 w-4" />, // Use Settings icon (placeholder, File not declared)
                // Pass content, add padding via className to the renderer's wrapper
                content: <div className="tab-content-container"><MarkdownRenderer content={placeholderMarkdown} className="" /></div>,
              },
              {
                label: 'LLM Query',
                icon: <Send className="h-4 w-4" />, // Use Send icon (declared)
                content: (
                  // Container ensures elements stack correctly and use full height
                  <div className="flex flex-col h-full tab-content-container">
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
            // Allow tab content to grow and scroll internally. Padding handled by tab-content-container class.
            tabContentClassName="bg-gray-800 text-white custom-scrollbar right-pane-content overflow-y-auto"
          />
        </div>
      </div>
    </div>
  );
};

export default LowerControlPanel;