import React from 'react';
import Tabs from '../../ui/Tabs';
import NostrAuthSection from '../../features/auth/components/NostrAuthSection';
import SystemPanel from '../../features/settings/components/panels/SystemPanel';
import VisualisationPanel from '../../features/settings/components/panels/VisualisationPanel';
import XRPanel from '../../features/settings/components/panels/XRPanel';
import AIPanel from '../../features/settings/components/panels/AIPanel';
import MarkdownRenderer from '../../ui/markdown/MarkdownRenderer';
import { Button } from '../../ui/Button';
import { Settings, Eye, Smartphone, Send, Anchor } from 'lucide-react'; // Using Settings as placeholder for missing icons
import { settingsUIDefinition } from '../../features/settings/config/settingsUIDefinition';

// Placeholder markdown content (can be moved to a separate file or constant)
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

${'- This is a repeated line to force scrolling\n'.repeat(20)}

### More Content

${'- Another repeated line with different text\n'.repeat(20)}
`;

const LowerControlPanel: React.FC = () => {
  return (
    <div className="flex flex-col w-full h-full bg-card text-card-foreground overflow-hidden">
      <div className="flex flex-1 min-h-0"> {/* This flex-1 and min-h-0 is key for scrolling */}
        {/* Left Pane: Settings - 40% width */}
        <div className="w-[40%] border-r border-border flex flex-col overflow-hidden">
          <Tabs
            tabs={[
              { label: 'Auth', icon: <Settings size={16} />, content: <div className="p-4 overflow-y-auto h-full custom-scrollbar"><NostrAuthSection /></div> }, // Placeholder icon
              { label: 'Visualisation', icon: <Eye size={16} />, content: <VisualisationPanel settingsDef={settingsUIDefinition.visualisation} /> },
              { label: 'System', icon: <Settings size={16} />, content: <SystemPanel settingsDef={settingsUIDefinition.system} /> },
              { label: 'XR', icon: <Smartphone size={16} />, content: <XRPanel settingsDef={settingsUIDefinition.xr} /> },
              { label: 'AI', icon: <Settings size={16} />, content: <AIPanel settingsDef={settingsUIDefinition.ai} /> }, // Placeholder icon
            ]}
            tabListClassName="flex-shrink-0 bg-background border-b border-border" // Ensure tab list doesn't shrink
            tabContentClassName="flex-grow overflow-y-auto custom-scrollbar" // Ensure content area scrolls
          />
        </div>

        {/* Right Pane: Tools - 60% width */}
        <div className="w-[60%] flex flex-col overflow-hidden">
          <Tabs
            tabs={[
              { label: 'Narrative Gold Mine', icon: <Anchor size={16} />, content: (
                  <div className="w-full h-full overflow-hidden">
                    <iframe
                      src="https://narrativegoldmine.com"
                      className="w-full h-full border-none"
                      title="Narrative Gold Mine"
                      sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
                      loading="lazy"
                      referrerPolicy="no-referrer"
                    />
                  </div>
                )
              },
              { label: 'Markdown', icon: <Settings size={16} />, content: <div className="p-4 overflow-y-auto h-full custom-scrollbar"><MarkdownRenderer content={placeholderMarkdown} className="" /></div> }, // Placeholder icon
              { label: 'LLM Query', icon: <Send size={16} />, content: (
                  <div className="p-4 flex flex-col h-full">
                    <textarea
                      className="flex-1 mb-2 p-2 border border-border rounded bg-input text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary custom-scrollbar"
                      placeholder="Enter your query..."
                    />
                    <Button className="self-end">Send Query</Button>
                  </div>
                )
              },
            ]}
            tabListClassName="flex-shrink-0 bg-background border-b border-border"
            tabContentClassName="flex-grow overflow-y-auto custom-scrollbar"
          />
        </div>
      </div>
    </div>
  );
};

export default LowerControlPanel;