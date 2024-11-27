<!-- Debug panel disabled for performance optimization -->
<template>
  <div v-if="false" class="debug-panel"></div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
  name: 'DebugPanel'
})
</script>
