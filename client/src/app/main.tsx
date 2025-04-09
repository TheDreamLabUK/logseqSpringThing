import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import '../styles/tokens.css'; // Use relative path
import '../styles/layout.css'; // Use relative path
import '../styles/globals.css'; // Use relative path

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
