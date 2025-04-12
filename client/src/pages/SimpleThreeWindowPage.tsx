import React, { Suspense, useEffect, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// Import graph components and managers
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import GraphManager from '../features/graph/components/GraphManager';

// Import styles
import '../styles/simple-three-window-page.css';
import '../styles/simple-graph-page.css'; // Import existing styles as well

// Camera controller to automatically position based on graph data
const CameraController = ({ center, size }: { center: [number, number, number], size: number }) => {
  const { camera } = useThree();

  useEffect(() => {
    // Position camera to look at the center from a distance
    // that's proportional to the graph size
    camera.position.set(center[0], center[1], center[2] + size*1.5);
    camera.lookAt(center[0], center[1], center[2]);
    camera.updateProjectionMatrix();
  }, [camera, center, size]);

  return null;
};

// Individual graph window component
interface GraphWindowProps {
  title: string;
  graphCenter: [number, number, number];
  graphSize: number;
  backgroundColor: string;
  className: string;
  showControls?: boolean;
}

const GraphWindow: React.FC<GraphWindowProps> = ({
  title,
  graphCenter,
  graphSize,
  backgroundColor,
  className,
  showControls = false
}) => {
  return (
    <div className={`graph-window ${className}`}>
      <div className="window-title">{title}</div>
      <Canvas
        className="three-canvas"
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
        <color attach="background" args={[backgroundColor]} />
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
        <axesHelper args={[2]} />
        <Stats />
      </Canvas>

      {showControls && (
        <div className="controls-overlay">
          <p>Orbit: Left-click drag</p>
          <p>Pan: Right-click drag</p>
          <p>Zoom: Scroll wheel</p>
        </div>
      )}
    </div>
  );
};

const SimpleThreeWindowPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50); // Default size

  // Fetch initial graph data on mount
  useEffect(() => {
    const initializeGraph = async () => {
      try {
        console.log('SimpleThreeWindowPage: Fetching initial graph data...');
        await graphDataManager.fetchInitialData();
        console.log('SimpleThreeWindowPage: Initial graph data fetched successfully.');

        // Calculate graph extents and center
        const data = graphDataManager.getGraphData();
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        // Process node positions to find extents
        data.nodes.forEach(node => {
          if (node.position) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
            minZ = Math.min(minZ, node.position.z);
            maxZ = Math.max(maxZ, node.position.z);
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

        // Set state for camera positioning
        setGraphCenter([centerX, centerY, centerZ]);
        setGraphSize(maxDimension);
        setIsLoading(false);
      } catch (err) {
        console.error('SimpleThreeWindowPage: Failed to fetch initial graph data:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred during data fetch.');
        setIsLoading(false);
      }
    };

    initializeGraph();
  }, []); // Empty dependency array ensures this runs only once on mount

  if (isLoading) {
    return <div style={{ padding: '2rem', color: 'white', backgroundColor: '#222' }}>Loading graph data...</div>;
  }

  if (error) {
    return <div style={{ padding: '2rem', color: 'red', backgroundColor: '#222' }}>Error loading graph data: {error}</div>;
  }

  return (
    <div className="app-container">
      <div className="three-window-container">
        <GraphWindow
          title="Top Left View"
          graphCenter={graphCenter}
          graphSize={graphSize}
          backgroundColor="#000066"
          className="graph-window-1"
        />
        <GraphWindow
          title="Top Right View"
          graphCenter={graphCenter}
          graphSize={graphSize}
          backgroundColor="#000066"
          className="graph-window-2"
        />
        <GraphWindow
          title="Bottom View"
          graphCenter={graphCenter}
          graphSize={graphSize}
          backgroundColor="#000066"
          className="graph-window-3"
          showControls={true}
        />
      </div>
    </div>
  );
};

export default SimpleThreeWindowPage;
