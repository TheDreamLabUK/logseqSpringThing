import React, { useState, useCallback, CSSProperties } from 'react';
import GraphViewport from '../features/graph/components/GraphViewport';
import RightPaneControlPanel from './components/RightPaneControlPanel'; // Added import

const TwoPaneLayout: React.FC = () => {
  const [leftPaneWidth, setLeftPaneWidth] = useState<number>(300); // Initial width of the left pane
  const [isDraggingVertical, setIsDraggingVertical] = useState<boolean>(false); // Renamed for clarity
  const [isRightPaneDocked, setIsRightPaneDocked] = useState<boolean>(false);

  // State for horizontal splitter in right pane
  const [rightPaneTopHeight, setRightPaneTopHeight] = useState<number>(200); // Initial height for top section
  const [isDraggingHorizontal, setIsDraggingHorizontal] = useState<boolean>(false);


  const handleVerticalMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDraggingVertical(true);
    // Prevent text selection while dragging
    e.preventDefault();
  }, []);

  const handleVerticalMouseUp = useCallback(() => {
    setIsDraggingVertical(false);
  }, []);

  const handleVerticalMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDraggingVertical && !isRightPaneDocked) {
        const newWidth = e.clientX;
        if (newWidth > 50 && newWidth < window.innerWidth - 50) {
          setLeftPaneWidth(newWidth);
        }
      }
    },
    [isDraggingVertical, isRightPaneDocked]
  );

  // Event handlers for horizontal splitter
  const handleHorizontalMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDraggingHorizontal(true);
    e.preventDefault();
  }, []);

  const handleHorizontalMouseUp = useCallback(() => {
    setIsDraggingHorizontal(false);
  }, []);

  const handleHorizontalMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDraggingHorizontal) {
        // Assuming the right pane is relative to the viewport top for clientY
        // This might need adjustment if the right pane has a different offset parent
        const newHeight = e.clientY;
        const rightPaneElement = document.getElementById('right-pane-container'); // Need an ID for the right pane
        if (rightPaneElement) {
            const rect = rightPaneElement.getBoundingClientRect();
            const relativeNewHeight = e.clientY - rect.top;

            // Min height 50px for top, and bottom pane also needs at least 50px + divider height (10px)
            if (relativeNewHeight > 50 && relativeNewHeight < rect.height - 50 - 10) {
                 setRightPaneTopHeight(relativeNewHeight);
            }
        }
      }
    },
    [isDraggingHorizontal]
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
    const handleGlobalMouseMove = (e: MouseEvent) => {
      handleVerticalMouseMove(e);
      handleHorizontalMouseMove(e);
    };

    const handleGlobalMouseUp = () => {
      handleVerticalMouseUp();
      handleHorizontalMouseUp();
    };

    if (isDraggingVertical || isDraggingHorizontal) {
      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
    } else {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isDraggingVertical, isDraggingHorizontal, handleVerticalMouseMove, handleHorizontalMouseMove, handleVerticalMouseUp, handleHorizontalMouseUp]);

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

  const rightPaneContainerStyle: CSSProperties = { // Renamed for clarity
    flexGrow: 1,
    display: isRightPaneDocked ? 'none' : 'flex', // Use flex column for top/bottom sections
    flexDirection: 'column',
    overflow: 'hidden', // Containing div handles overflow
    height: '100vh', // Ensure it takes full viewport height
  };

  const rightPaneTopStyle: CSSProperties = {
    height: `${rightPaneTopHeight}px`,
    minHeight: '50px', // Min height for top section
    // backgroundColor: '#e0e0e0', // Removed, panel has its own
    // padding: '10px', // Removed, panel has its own
    overflowY: 'hidden', // Panel itself will scroll its content
    position: 'relative',
  };

  const horizontalDividerStyle: CSSProperties = {
    height: '10px',
    cursor: 'ns-resize',
    backgroundColor: '#b0b0b0', // Darker for grab handle appearance
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    userSelect: 'none',
    borderTop: '1px solid #999999',
    borderBottom: '1px solid #999999',
  };

  const rightPaneBottomStyle: CSSProperties = {
    flexGrow: 1, // Bottom section takes remaining space
    minHeight: '50px', // Min height for bottom section
    backgroundColor: '#d0d0d0',
    padding: '10px', // Reduced padding
    overflowY: 'auto', // Scroll for content overflow
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
        onMouseDown={!isRightPaneDocked ? handleVerticalMouseDown : undefined}
        title={isRightPaneDocked ? "" : "Drag to resize"}
      >
        ||
      </div>
      <div id="right-pane-container" style={rightPaneContainerStyle}>
        {!isRightPaneDocked && (
          <>
            <div style={rightPaneTopStyle}>
              <RightPaneControlPanel />
            </div>
            <div
              style={horizontalDividerStyle}
              onMouseDown={handleHorizontalMouseDown}
              title="Drag to resize sections"
            >
              ══
            </div>
            <div style={rightPaneBottomStyle}>
              <h2>Bottom Right Section</h2>
              <p>Placeholder for other content</p>
            </div>
          </>
        )}
      </div>
      <button onClick={toggleRightPaneDock} style={dockButtonStyle} title={isRightPaneDocked ? "Expand Right Pane" : "Collapse Right Pane"}>
        {isRightPaneDocked ? '>' : '<'}
      </button>
    </div>
  );
};

export default TwoPaneLayout;