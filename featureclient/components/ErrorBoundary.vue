<template>
  <div>
    <div v-if="error" class="error-boundary">
      <div class="error-content">
        <h3>Component Error</h3>
        <div class="error-message">{{ error.message }}</div>
        <div class="error-info">
          <div class="error-component">Component: {{ error.component }}</div>
          <div class="error-stack">{{ error.stack }}</div>
        </div>
        <button @click="handleError" class="retry-button">
          Retry
        </button>
      </div>
    </div>
    <slot v-else></slot>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onErrorCaptured, ComponentPublicInstance } from 'vue'

export default defineComponent({
  name: 'ErrorBoundary',
  
  setup() {
    const error = ref<{
      message: string;
      component?: string;
      stack?: string;
    } | null>(null)

    onErrorCaptured((err, instance: ComponentPublicInstance | null, info) => {
      // Get component name
      const componentName = (instance as any)?.$options?.name || 'Unknown Component'
      
      // Format error for display
      error.value = {
        message: err.message || String(err),
        component: componentName,
        stack: err.stack
      }

      // Log to debug console
      console.error('Component Error:', {
        message: err.message,
        component: componentName,
        stack: err.stack,
        context: `Vue Component: ${componentName}`
      })

      // Prevent error from propagating
      return false
    })

    const handleError = () => {
      error.value = null
    }

    return {
      error,
      handleError
    }
  }
})
</script>

<style scoped>
.error-boundary {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.85);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
}

.error-content {
  background-color: #2c3e50;
  padding: 20px;
  border-radius: 8px;
  max-width: 80%;
  max-height: 80%;
  overflow-y: auto;
  color: white;
  font-family: monospace;
}

.error-content h3 {
  color: #e74c3c;
  margin-top: 0;
}

.error-message {
  color: #f1c40f;
  margin: 10px 0;
  font-size: 16px;
}

.error-info {
  margin: 15px 0;
  padding: 10px;
  background-color: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
}

.error-component {
  color: #3498db;
  margin-bottom: 8px;
}

.error-stack {
  color: #95a5a6;
  font-size: 12px;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.retry-button {
  background-color: #2ecc71;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  margin-top: 15px;
}

.retry-button:hover {
  background-color: #27ae60;
}
</style>
