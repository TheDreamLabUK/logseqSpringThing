import React, { Suspense, useEffect, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// Import graph components and managers with new paths
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import GraphManager from '../features/graph/components/GraphManager';

// Import layout components with new paths
import MainLayout from '../components/layout/MainLayout';
import ViewportContainer from '../components/layout/ViewportContainer';

// Camera controller to automatically position based on graph data
const CameraController = ({ center, size }: { center: [number, number, number], size: number }) => {
  const { camera } = useThree();
  
  useEffect(() => {
    console.log(`Setting camera to look at center: [${center}] with distance: ${size*1.5}`);
    
    // Position camera to look at the center from a distance
    // that's proportional to the graph size
    camera.position.set(center[0], center[1], center[2] + size*1.5);
    camera.lookAt(center[0], center[1], center[2]);
    camera.updateProjectionMatrix();
  }, [camera, center, size]);
  
  return null;
};

const SimpleGraphPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50); // Default size

  // Fetch initial graph data on mount
  useEffect(() => {
    const initializeGraph = async () => {
      try {
        console.log('SimpleGraphPage: Fetching initial graph data...');
        await graphDataManager.fetchInitialData();
        console.log('SimpleGraphPage: Initial graph data fetched successfully.');
        
        // Calculate graph extents and center
        const data = graphDataManager.getGraphData();
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        // Debug positions array directly
        console.log("Raw positions from first 5 nodes:");
        data.nodes.slice(0, 5).forEach((node, i) => {
          if (node.position) {
            console.log(`Node ${i} (${node.id}): x=${node.position.x}, y=${node.position.y}, z=${node.position.z}`);
            
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
            minZ = Math.min(minZ, node.position.z);
            maxZ = Math.max(maxZ, node.position.z);
          } else {
            console.log(`Node ${i} (${node.id}): POSITION IS NULL`);
          }
        });
        
        // Calculate center and dimensions
        const centerX = (maxX + minX) / 2;
        const centerY = (maxY + minY) / 2;
        const centerZ = (maxZ + minZ) / 2;
        
        const width = maxX - minX;
        const height = maxY - minY;
        const depth = maxZ - minZ;
        
        // Use the largest dimension as the "size" of the graph
        const maxDimension = Math.max(width, height, depth);
        
        console.log(`Graph extents: X [${minX.toFixed(2)} to ${maxX.toFixed(2)}], Y [${minY.toFixed(2)} to ${maxY.toFixed(2)}], Z [${minZ.toFixed(2)} to ${maxZ.toFixed(2)}]`);
        console.log(`Graph dimensions: Width=${width.toFixed(2)}, Height=${height.toFixed(2)}, Depth=${depth.toFixed(2)}`);
        console.log(`Graph center: X=${centerX.toFixed(2)}, Y=${centerY.toFixed(2)}, Z=${centerZ.toFixed(2)}`);
        console.log(`Max dimension: ${maxDimension.toFixed(2)}`);
        
        // Set state for camera positioning
        setGraphCenter([centerX, centerY, centerZ]);
        setGraphSize(maxDimension);
        setIsLoading(false);
      } catch (err) {
        console.error('SimpleGraphPage: Failed to fetch initial graph data:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred during data fetch.');
        setIsLoading(false);
      }
    };

    initializeGraph();
  }, []); // Empty dependency array ensures this runs only once on mount

  // Background color (using hex format for the color component)
  const blueBackgroundColor = '#0000cd';

  if (isLoading) {
    return <div style={{ padding: '2rem', color: 'white', backgroundColor: '#222' }}>Loading graph data...</div>;
  }

  if (error) {
    return <div style={{ padding: '2rem', color: 'red', backgroundColor: '#222' }}>Error loading graph data: {error}</div>;
  }

  return (
    <div className="app-container"> {/* Use app-container for base styles */}
      <MainLayout
        topDockContent={null}
        viewportContent={
          <ViewportContainer>
            <div className="relative w-full h-full"> {/* Ensure relative positioning for absolute canvas */}
              <div className="absolute inset-0"> {/* Container for Canvas */} 
                <Canvas
                  className="three-canvas" /* Added class for easier styling */
                  style={{ display: 'block' }}
                  camera={{ 
                    position: [0, 0, 50], // Initial position
                    fov: 50,
                    near: 0.1,
                    far: 2000
                  }}
                  gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
                  dpr={[1, 2]}
                  shadows
                >
                  <color attach="background" args={[blueBackgroundColor]} />
                  <CameraController center={graphCenter} size={graphSize} />
                  <ambientLight intensity={0.6} />
                  <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
                  <pointLight position={[-10, -10, -5]} intensity={0.5} />
                  <OrbitControls
                    makeDefault
                    enableDamping
                    dampingFactor={0.05}
                    minDistance={1}
                    maxDistance={2000}
                    target={[graphCenter[0], graphCenter[1], graphCenter[2]]}
                  />
                  <Suspense fallback={null}>
                    <GraphManager />
                  </Suspense>
                  {/* Debug axes helper */}
                  <axesHelper args={[2]} />
                  <Stats />
                </Canvas>
              </div>
            </div>
          </ViewportContainer>
        }
      />
      
      {/* Keep controls overlay */}
      <div style={{ 
        position: 'absolute', 
        bottom: '10px', 
        left: '10px', 
        color: 'white', 
        zIndex: 1000,
        background: 'rgba(0,0,0,0.5)',
        padding: '10px',
        borderRadius: '5px'
      }}>
        <p>Orbit: Left-click drag</p>
        <p>Pan: Right-click drag</p>
        <p>Zoom: Scroll wheel</p>
      </div>
    </div>
  );
};

export default SimpleGraphPage;
