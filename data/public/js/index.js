// data/public/js/index.js

import { App } from './app.js';

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.start().catch(error => {
        console.error('Failed to start application:', error);
    });

    // Handle cleanup on page unload
    window.addEventListener('beforeunload', () => {
        app.stop();
    });
});
