import React, { useState, useCallback, CSSProperties } from 'react';
import GraphViewport from '../features/graph/components/GraphViewport'; // Added import

const TwoPaneLayout: React.FC = () => {
  const [leftPaneWidth, setLeftPaneWidth] = useState<number>(300); // Initial width of the left pane
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isRightPaneDocked, setIsRightPaneDocked] = useState<boolean>(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    // Prevent text selection while dragging
    e.preventDefault();
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDragging && !isRightPaneDocked) {
        // Calculate new width based on mouse position
        // Adjust as needed based on parent container's offset
        const newWidth = e.clientX;
        // Add constraints for min/max width if necessary
        if (newWidth > 50 && newWidth < window.innerWidth - 50) { // Basic constraints
          setLeftPaneWidth(newWidth);
        }
      }
    },
    [isDragging, isRightPaneDocked]
  );

  const toggleRightPaneDock = () => {
    setIsRightPaneDocked(!isRightPaneDocked);
    if (!isRightPaneDocked) {
      // Optionally, reset left pane width or set to a specific value when docking
    } else {
      // Optionally, restore left pane width or set to a specific value when undocking
      // For now, it will just expand based on leftPaneWidth
    }
  };

  // Add and remove mouse move/up listeners on the window
  React.useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const containerStyle: CSSProperties = {
    display: 'flex',
    height: '100vh', // Full viewport height
    overflow: 'hidden', // Prevent scrollbars on the container itself
  };

  const leftPaneStyle: CSSProperties = {
    width: isRightPaneDocked ? '100%' : `${leftPaneWidth}px`,
    minWidth: '50px', // Minimum width for the left pane
    // backgroundColor: '#f0f0f0', // Removed, canvas has its own
    // padding: '20px', // Removed, GraphViewport handles its own layout
    // overflow: 'auto', // Removed, GraphViewport handles its own overflow/scroll
    height: '100%', // Ensure left pane takes full height for the canvas
    position: 'relative', // For potential absolute positioned elements within GraphViewport
    transition: 'width 0.3s ease', // Smooth transition for docking
  };

  const dividerStyle: CSSProperties = {
    width: '10px',
    cursor: isRightPaneDocked ? 'default' : 'ew-resize', // Change cursor when docked
    backgroundColor: '#cccccc',
    display: isRightPaneDocked ? 'none' : 'flex', // Hide divider when docked
    alignItems: 'center',
    justifyContent: 'center',
    userSelect: 'none',
  };

  const rightPaneStyle: CSSProperties = {
    flexGrow: 1, // Right pane takes remaining space
    backgroundColor: '#e0e0e0',
    padding: '20px',
    display: isRightPaneDocked ? 'none' : 'block', // Hide right pane when docked
    overflow: 'auto', // Add scroll if content overflows
  };

  const dockButtonStyle: CSSProperties = {
    position: 'absolute',
    top: '10px',
    right: isRightPaneDocked ? '10px' : `${10 + (isRightPaneDocked ? 0 : 0)}px`, // Adjust button position
    zIndex: 100,
    padding: '5px 10px',
    cursor: 'pointer',
  };


  return (
    <div style={containerStyle}>
      <div style={leftPaneStyle}>
        <GraphViewport />
      </div>
      <div
        style={dividerStyle}
        onMouseDown={!isRightPaneDocked ? handleMouseDown : undefined}
        title={isRightPaneDocked ? "" : "Drag to resize"}
      >
        ||
      </div>
      <div style={rightPaneStyle}>
        <h2>Right Pane</h2>
        <p>Right Pane Placeholder</p>
      </div>
      <button onClick={toggleRightPaneDock} style={dockButtonStyle} title={isRightPaneDocked ? "Expand Right Pane" : "Collapse Right Pane"}>
        {isRightPaneDocked ? '>' : '<'}
      </button>
    </div>
  );
};

export default TwoPaneLayout;