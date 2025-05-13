import React, { useState, useCallback, CSSProperties } from 'react';
import GraphViewport from '../features/graph/components/GraphViewport';
import RightPaneControlPanel from './components/RightPaneControlPanel';
// import MarkdownDisplayPanel from './components/MarkdownDisplayPanel'; // Remove this
import ConversationPane from './components/ConversationPane'; // Add this
import NarrativeGoldminePanel from './components/NarrativeGoldminePanel';

const TwoPaneLayout: React.FC = () => {
  const [leftPaneWidth, setLeftPaneWidth] = useState<number>(300); // Initial width of the left pane
  const [isDraggingVertical, setIsDraggingVertical] = useState<boolean>(false);
  const [isRightPaneDocked, setIsRightPaneDocked] = useState<boolean>(false);

  // State for TOP horizontal splitter in right pane (dividing Control Panel and the rest)
  const [rightPaneTopHeight, setRightPaneTopHeight] = useState<number>(300); // Adjusted initial height
  const [isDraggingHorizontalTop, setIsDraggingHorizontalTop] = useState<boolean>(false); // Renamed

  // State for the new BOTTOM horizontal splitter within the lower part of the right pane
  const [bottomRightUpperHeight, setBottomRightUpperHeight] = useState<number>(200); // Initial height for Markdown panel
  const [isDraggingHorizontalBottom, setIsDraggingHorizontalBottom] = useState<boolean>(false);


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

  // Event handlers for TOP horizontal splitter
  const handleHorizontalTopMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDraggingHorizontalTop(true);
    e.preventDefault();
  }, []);

  const handleHorizontalTopMouseUp = useCallback(() => {
    setIsDraggingHorizontalTop(false);
  }, []);

  const handleHorizontalTopMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDraggingHorizontalTop) {
        const rightPaneContainer = document.getElementById('right-pane-container');
        if (rightPaneContainer) {
            const rect = rightPaneContainer.getBoundingClientRect();
            const relativeNewHeight = e.clientY - rect.top;
            if (relativeNewHeight > 50 && relativeNewHeight < rect.height - 110) { // Min 50 for top, 110 for bottom container (50+10+50)
                 setRightPaneTopHeight(relativeNewHeight);
            }
        }
      }
    },
    [isDraggingHorizontalTop]
  );

  // Event handlers for BOTTOM horizontal splitter
  const handleHorizontalBottomMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDraggingHorizontalBottom(true);
    e.preventDefault();
  }, []);

  const handleHorizontalBottomMouseUp = useCallback(() => {
    setIsDraggingHorizontalBottom(false);
  }, []);

  const handleHorizontalBottomMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDraggingHorizontalBottom) {
        const bottomRightContainer = document.getElementById('right-pane-bottom-container');
        if (bottomRightContainer) {
            const rect = bottomRightContainer.getBoundingClientRect();
            const relativeNewHeight = e.clientY - rect.top;
            if (relativeNewHeight > 50 && relativeNewHeight < rect.height - 60) { // Min 50 for upper, 50 for lower + 10 for divider
                 setBottomRightUpperHeight(relativeNewHeight);
            }
        }
      }
    },
    [isDraggingHorizontalBottom]
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
      handleHorizontalTopMouseMove(e);
      handleHorizontalBottomMouseMove(e);
    };

    const handleGlobalMouseUp = () => {
      handleVerticalMouseUp();
      handleHorizontalTopMouseUp();
      handleHorizontalBottomMouseUp();
    };

    if (isDraggingVertical || isDraggingHorizontalTop || isDraggingHorizontalBottom) {
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
  }, [
      isDraggingVertical, isDraggingHorizontalTop, isDraggingHorizontalBottom,
      handleVerticalMouseMove, handleHorizontalTopMouseMove, handleHorizontalBottomMouseMove,
      handleVerticalMouseUp, handleHorizontalTopMouseUp, handleHorizontalBottomMouseUp
    ]);

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

  const horizontalTopDividerStyle: CSSProperties = { // Renamed
    height: '10px',
    cursor: 'ns-resize',
    backgroundColor: '#b0b0b0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    userSelect: 'none',
    borderTop: '1px solid #999999',
    borderBottom: '1px solid #999999',
    flexShrink: 0,
  };

  const rightPaneBottomContainerStyle: CSSProperties = { // New container for the split
    flexGrow: 1,
    minHeight: '110px', // 50 (upper) + 10 (divider) + 50 (lower)
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    backgroundColor: '#d0d0d0', // Base for this area
  };

  const bottomRightUpperStyle: CSSProperties = {
    height: `${bottomRightUpperHeight}px`,
    minHeight: '50px',
    // backgroundColor: '#d8d8d8', // Panel has its own bg
    padding: '0px', // Panel has its own padding
    overflowY: 'hidden', // Panel handles scroll
    position: 'relative',
  };

  const horizontalBottomDividerStyle: CSSProperties = { // New divider
    height: '10px',
    cursor: 'ns-resize',
    backgroundColor: '#a0a0a0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    userSelect: 'none',
    borderTop: '1px solid #888888',
    borderBottom: '1px solid #888888',
    flexShrink: 0,
  };

  const bottomRightLowerStyle: CSSProperties = {
    flexGrow: 1,
    minHeight: '50px',
    // backgroundColor: '#c8c8c8', // Panel has its own bg
    padding: '0px', // Panel has its own padding
    overflowY: 'hidden', // Panel handles scroll
    position: 'relative',
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
              style={horizontalTopDividerStyle}
              onMouseDown={handleHorizontalTopMouseDown}
              title="Drag to resize Control Panel / Lower Area"
            >
              ══
            </div>
            <div id="right-pane-bottom-container" style={rightPaneBottomContainerStyle}>
              <div style={bottomRightUpperStyle}>
                {/* <MarkdownDisplayPanel /> */} {/* Replace this */}
                <ConversationPane /> {/* With this */}
              </div>
              <div
                style={horizontalBottomDividerStyle}
                onMouseDown={handleHorizontalBottomMouseDown}
                title="Drag to resize Markdown / Narrative Goldmine"
              >
                ══
              </div>
              <div style={bottomRightLowerStyle}>
                <NarrativeGoldminePanel />
              </div>
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