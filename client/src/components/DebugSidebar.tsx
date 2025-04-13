import React from 'react';

interface DebugSidebarProps {
  visible: boolean;
  onClose: () => void;
}

const DebugSidebar: React.FC<DebugSidebarProps> = ({ visible, onClose }) => {
  if (!visible) return null;

  return (
    <div 
      className="fixed top-0 right-0 w-80 h-full bg-black text-white z-[9999] overflow-y-auto"
      style={{ 
        boxShadow: '-5px 0 20px rgba(255,0,0,0.5)',
        border: '4px solid red'
      }}
    >
      <div className="p-4">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold">Debug Sidebar</h2>
          <button 
            onClick={onClose}
            className="bg-red-500 text-white p-2 rounded-full"
          >
            X
          </button>
        </div>
        
        <div className="space-y-4">
          <div className="border border-gray-700 p-3 rounded">
            <h3 className="text-lg font-semibold mb-2">Authentication</h3>
            <p>Status: Authenticated</p>
            <button className="mt-2 bg-red-600 text-white px-3 py-1 rounded">
              Logout
            </button>
          </div>
          
          <div className="border border-gray-700 p-3 rounded">
            <h3 className="text-lg font-semibold mb-2">Settings</h3>
            <div className="space-y-2">
              <div>
                <label className="block text-sm">Debug Mode</label>
                <div className="flex items-center mt-1">
                  <input type="checkbox" className="mr-2" />
                  <span>Enabled</span>
                </div>
              </div>
              
              <div>
                <label className="block text-sm">Rendering Quality</label>
                <select className="w-full bg-gray-800 p-1 rounded mt-1">
                  <option>Low</option>
                  <option>Medium</option>
                  <option>High</option>
                </select>
              </div>
            </div>
          </div>
          
          <div className="border border-gray-700 p-3 rounded">
            <h3 className="text-lg font-semibold mb-2">Visualization</h3>
            <div className="space-y-2">
              <div>
                <label className="block text-sm">Node Size</label>
                <input 
                  type="range" 
                  min="1" 
                  max="100" 
                  className="w-full mt-1" 
                />
              </div>
              
              <div>
                <label className="block text-sm">Edge Thickness</label>
                <input 
                  type="range" 
                  min="1" 
                  max="10" 
                  className="w-full mt-1" 
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DebugSidebar;
