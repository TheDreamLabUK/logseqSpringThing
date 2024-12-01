interface NodeData {
  position: {
    x: number;
    y: number;
    z: number;
  };
  velocity: {
    x: number;
    y: number;
    z: number;
  };
}

// Debug utility function with enhanced websocket logging
const debugLog = (message: string, data?: any) => {
  // Always log in production for debugging
  const timestamp = new Date().toISOString();
  console.debug(`[WebsocketService ${timestamp}] ${message}`);
  
  if (data) {
    if (data instanceof ArrayBuffer) {
      // For binary data, show detailed header and content info
      const view = new DataView(data);
      const isInitial = view.getFloat32(0, true);
      const nodeCount = (data.byteLength - 4) / 24; // 24 bytes per node (6 float32s)
      
      // Show binary header info
      console.debug(`[WebSocket Binary Data] Header:
        Is Initial: ${isInitial}
        Node Count: ${nodeCount}
        Total Size: ${data.byteLength} bytes
        Content Type: ArrayBuffer
        Byte Length: ${data.byteLength}
        First 32 bytes: ${Array.from(new Uint8Array(data.slice(0, 32))).map(b => b.toString(16).padStart(2, '0')).join(' ')}`);

      // Show sample of node data if available
      if (nodeCount > 0) {
        const sampleNodeData: NodeData[] = [];
        for (let i = 0; i < Math.min(3, nodeCount); i++) {
          const offset = 4 + i * 24; // Skip header (4 bytes) and calculate node offset
          const nodeView = new DataView(data, offset, 24);
          sampleNodeData.push({
            position: {
              x: nodeView.getFloat32(0, true),
              y: nodeView.getFloat32(4, true),
              z: nodeView.getFloat32(8, true)
            },
            velocity: {
              x: nodeView.getFloat32(12, true),
              y: nodeView.getFloat32(16, true),
              z: nodeView.getFloat32(20, true)
            }
          });
        }
        console.debug('[WebSocket Binary Data] Sample Nodes:', sampleNodeData);
      }
    } else if (typeof data === 'object') {
      // For JSON data, show full structure with type information
      console.debug('[WebSocket JSON Data]:', {
        type: data.type || 'unknown',
        size: JSON.stringify(data).length,
        timestamp: new Date().toISOString(),
        content: data,
        stack: new Error().stack // Show call stack for debugging
      });
    } else {
      console.debug('[WebSocket Raw Data]:', {
        type: typeof data,
        value: data,
        timestamp: new Date().toISOString()
      });
    }
  }
};
