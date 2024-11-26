import { createApp } from 'vue'
import VueThreejs from 'vue-threejs'
import App from './components/App.vue'

// Create and mount the Vue application
const app = createApp(App)

// Use VueThreejs
app.use(VueThreejs)

// Mount the app
app.mount('#app')
