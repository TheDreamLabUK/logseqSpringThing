<template>
  <div id="app">
    <div id="scene-container"></div>
    <ControlPanel />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import ControlPanel from '@components/ControlPanel.vue'

export default defineComponent({
  name: 'App',
  components: {
    ControlPanel
  }
})
</script>

<style>
body, html {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
  background: #000000;
}

#scene-container {
  width: 100%;
  height: 100%;
  position: fixed;
  top: 0;
  left: 0;
  z-index: 0;
  background: #000000;
  touch-action: none;
}

#app {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
  pointer-events: none;
}

#app > * {
  pointer-events: auto;
}
</style>
