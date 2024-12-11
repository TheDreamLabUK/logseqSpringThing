// Force-directed initialization must come first
import './init/forceDirected';

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './components/App.vue'
import { errorTracking } from './services/errorTracking'

// Disable Vue devtools
window.__VUE_PROD_DEVTOOLS__ = false

// Create Vue application
const app = createApp(App)

// Configure global error handler for Vue
app.config.errorHandler = (err, instance, info) => {
  // Track error with our service
  errorTracking.trackError(err, {
    context: 'Vue Error Handler',
    component: (instance as any)?.$options?.name || 'Unknown Component',
    additional: { info }
  })

  // Log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.error('Vue Error:', err)
    console.error('Component:', instance)
    console.error('Error Info:', info)
  }
}

// Create and use Pinia
const pinia = createPinia()
app.use(pinia)

// Add error tracking to Pinia
pinia.use(() => {
  return {
    error: (error: Error) => {
      errorTracking.trackError(error, {
        context: 'Pinia Store',
        additional: { store: error?.cause }
      })
    }
  }
})

// Mount the app
app.mount('#app')

// Log successful initialization
console.info('Application initialized', {
  context: 'App Initialization',
  environment: process.env.NODE_ENV,
  forceDirected: false // Log that force-directed is disabled
})
