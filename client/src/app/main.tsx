import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import '../styles/tokens.css'; // Use relative path
// Removed import for layout.css as the file was deleted
import '../styles/globals.css'; // Use relative path

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
