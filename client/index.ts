import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './components/App.vue'
import { errorTracking } from './services/errorTracking'

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

// Configure performance tracking in development
if (process.env.NODE_ENV === 'development') {
  app.config.performance = true
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
  debug: window.location.search.includes('debug')
})

// Add keyboard shortcut for error log download
document.addEventListener('keydown', (event) => {
  // Ctrl/Cmd + Shift + E to download error log
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'E') {
    errorTracking.downloadErrorLog()
  }
})
