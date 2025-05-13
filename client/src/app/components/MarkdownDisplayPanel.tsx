import React, { CSSProperties } from 'react';
import MarkdownRenderer from '../../ui/markdown/MarkdownRenderer'; // Adjusted path

const sampleMarkdownContent = `
# Markdown Display Panel

This panel uses the existing \`MarkdownRenderer\` component.

## Features
*   Renders Markdown text.
*   Supports GitHub Flavored Markdown (GFM).
*   Includes syntax highlighting for code blocks.

\`\`\`javascript
// Example JavaScript code
function helloWorld() {
  console.log("Hello, from the Markdown Panel!");
}
helloWorld();
\`\`\`

### More Content
You can add more Markdown content here to test scrolling and layout.
This panel is designed to fit within one of the resizable sub-panes.

${'- List item to add more content for scrolling.\n'.repeat(15)}

End of sample content.
`;

const MarkdownDisplayPanel: React.FC = () => {
  const panelStyle: CSSProperties = {
    width: '100%',
    height: '100%',
    overflowY: 'auto', // Enable vertical scroll for overflow within this panel
    padding: '10px', // Inner padding for the content area
    boxSizing: 'border-box',
    backgroundColor: '#fff', // White background for the markdown content area
  };

  return (
    <div style={panelStyle} className="custom-scrollbar">
      <MarkdownRenderer content={sampleMarkdownContent} className="" />
    </div>
  );
};

export default MarkdownDisplayPanel;