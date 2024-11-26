<template>
  <div class="debug-panel" :class="{ 'debug-panel--collapsed': isCollapsed }">
    <div class="debug-panel__header" @click="toggleCollapse">
      <span>Debug Panel</span>
      <div class="debug-panel__controls">
        <button @click.stop="clearLogs" title="Clear logs">🗑️</button>
        <button @click.stop="downloadLogs" title="Download logs">💾</button>
        <button @click.stop="toggleCollapse" :title="isCollapsed ? 'Expand' : 'Collapse'">
          {{ isCollapsed ? '⬆️' : '⬇️' }}
        </button>
      </div>
    </div>
    
    <div v-if="!isCollapsed" class="debug-panel__content">
      <div class="debug-panel__summary">
        <div class="summary-item">
          <strong>Total Errors:</strong> {{ errorSummary.total }}
        </div>
        <div class="summary-item">
          <strong>Error Types:</strong>
          <ul>
            <li v-for="(count, type) in errorSummary.types" :key="type">
              {{ type }}: {{ count }}
            </li>
          </ul>
        </div>
        <div class="summary-item">
          <strong>Components:</strong>
          <ul>
            <li v-for="(count, component) in errorSummary.components" :key="component">
              {{ component }}: {{ count }}
            </li>
          </ul>
        </div>
      </div>

      <div class="debug-panel__logs">
        <div v-for="error in errors" :key="error.timestamp" class="error-entry">
          <div class="error-entry__header">
            <span class="error-entry__timestamp">
              {{ new Date(error.timestamp).toLocaleString() }}
            </span>
            <span class="error-entry__context" :title="error.context">
              {{ error.context || 'Unknown Context' }}
            </span>
          </div>
          <div class="error-entry__message">{{ error.message }}</div>
          <div v-if="error.component" class="error-entry__component">
            Component: {{ error.component }}
          </div>
          <pre v-if="error.stack" class="error-entry__stack">{{ error.stack }}</pre>
          <div v-if="error.additional" class="error-entry__additional">
            <strong>Additional Info:</strong>
            <pre>{{ JSON.stringify(error.additional, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, onUnmounted } from 'vue'
import { errorTracking } from '../services/errorTracking'

export default defineComponent({
  name: 'DebugPanel',

  setup() {
    const isCollapsed = ref(true)
    const errors = computed(() => errorTracking.getErrors())
    const errorSummary = computed(() => errorTracking.getErrorSummary())

    const toggleCollapse = () => {
      isCollapsed.value = !isCollapsed.value
    }

    const clearLogs = () => {
      errorTracking.clearErrors()
    }

    const downloadLogs = () => {
      errorTracking.downloadErrorLog()
    }

    // Update when new errors are tracked
    const handleErrorTracked = () => {
      // Force reactivity update
      errors.value
    }

    onMounted(() => {
      window.addEventListener('error-tracked', handleErrorTracked)
    })

    onUnmounted(() => {
      window.removeEventListener('error-tracked', handleErrorTracked)
    })

    return {
      isCollapsed,
      errors,
      errorSummary,
      toggleCollapse,
      clearLogs,
      downloadLogs
    }
  }
})
</script>

<style scoped>
.debug-panel {
  position: fixed;
  bottom: 0;
  right: 0;
  width: 400px;
  max-height: 80vh;
  background-color: rgba(0, 0, 0, 0.9);
  color: #fff;
  font-family: monospace;
  border-top-left-radius: 8px;
  z-index: 9999;
  transition: transform 0.3s ease;
}

.debug-panel--collapsed {
  transform: translateY(calc(100% - 40px));
}

.debug-panel__header {
  padding: 10px;
  background-color: #2c3e50;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-top-left-radius: 8px;
}

.debug-panel__controls {
  display: flex;
  gap: 8px;
}

.debug-panel__controls button {
  background: none;
  border: none;
  cursor: pointer;
  padding: 2px 6px;
  font-size: 14px;
}

.debug-panel__content {
  padding: 10px;
  overflow-y: auto;
  max-height: calc(80vh - 40px);
}

.debug-panel__summary {
  margin-bottom: 15px;
  padding: 10px;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

.summary-item {
  margin-bottom: 10px;
}

.summary-item ul {
  margin: 5px 0;
  padding-left: 20px;
}

.error-entry {
  margin-bottom: 15px;
  padding: 10px;
  background-color: rgba(255, 0, 0, 0.1);
  border-radius: 4px;
}

.error-entry__header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 5px;
}

.error-entry__timestamp {
  color: #95a5a6;
  font-size: 12px;
}

.error-entry__context {
  color: #3498db;
  font-size: 12px;
}

.error-entry__message {
  color: #e74c3c;
  margin: 5px 0;
}

.error-entry__component {
  color: #2ecc71;
  font-size: 12px;
  margin: 5px 0;
}

.error-entry__stack {
  font-size: 11px;
  color: #95a5a6;
  margin: 5px 0;
  padding: 5px;
  background-color: rgba(0, 0, 0, 0.3);
  border-radius: 2px;
  overflow-x: auto;
  white-space: pre-wrap;
}

.error-entry__additional {
  font-size: 11px;
  margin-top: 5px;
}

.error-entry__additional pre {
  margin: 5px 0;
  padding: 5px;
  background-color: rgba(0, 0, 0, 0.3);
  border-radius: 2px;
  overflow-x: auto;
}
</style>
