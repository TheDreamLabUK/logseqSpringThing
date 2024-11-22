import { App } from './app.js';

console.log('Script loading...');

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, checking elements:');
    console.log('app element:', document.getElementById('app'));
    console.log('scene-container:', document.getElementById('scene-container'));
    console.log('connection-status:', document.getElementById('connection-status'));

    try {
        console.log('Creating App instance');
        const app = new App();
        console.log('Starting App');
        app.start();
    } catch (error) {
        console.error('Failed to initialize app:', error);
        console.error('Error stack:', error.stack);
        const debugInfo = document.getElementById('debug-info');
        if (debugInfo) {
            debugInfo.innerHTML += `<div>Init Error: ${error.message}</div>`;
            debugInfo.innerHTML += `<div>Stack: ${error.stack}</div>`;
        }
    }
});
