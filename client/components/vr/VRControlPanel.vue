<template>
  <!-- This is a 3D component, so no template needed -->
</template>

<script lang="ts">
import { defineComponent, onMounted, onBeforeUnmount, ref } from 'vue';
import * as THREE from 'three';
import { useVisualizationStore } from '../../stores/visualization';
import { useSettingsStore } from '../../stores/settings';

interface Props {
  scene: THREE.Scene;
  camera: THREE.Camera;
}

interface ControlChange {
  name: string;
  value: number | string;
}

interface Control {
  group: THREE.Group;
  min?: number;
  max?: number;
  value: number | string;
}

export default defineComponent({
  name: 'VRControlPanel',

  props: {
    scene: {
      type: Object as () => THREE.Scene,
      required: true
    },
    camera: {
      type: Object as () => THREE.Camera,
      required: true
    }
  },

  emits: {
    controlChange: (payload: ControlChange) => true
  },

  setup(props: Props, { emit }: { emit: (event: 'controlChange', payload: ControlChange) => void }) {
    const panel = ref<THREE.Group>(new THREE.Group());
    const controls = ref<Map<string, Control>>(new Map());
    const visualizationStore = useVisualizationStore();
    const settingsStore = useSettingsStore();

    const createTextTexture = (text: string): THREE.CanvasTexture => {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Could not get 2D context');

      canvas.width = 256;
      canvas.height = 64;
      context.font = '48px Arial';
      context.fillStyle = 'white';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(text, 128, 32);
      
      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      return texture;
    };

    const mapValue = (value: number, inMin: number, inMax: number, outMin: number, outMax: number): number => {
      return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    };

    const initPanel = () => {
      // Create background panel
      const panelGeometry = new THREE.PlaneGeometry(1, 1.5);
      const panelMaterial = new THREE.MeshBasicMaterial({ 
        color: 0x202020, 
        transparent: true, 
        opacity: 0.7 
      });
      const panelMesh = new THREE.Mesh(panelGeometry, panelMaterial);
      panel.value.add(panelMesh);

      // Position panel in front of camera
      panel.value.position.set(0, 0, -2);
      panel.value.lookAt(props.camera.position);

      props.scene.add(panel.value);
    };

    const createSlider = (name: string, min: number, max: number, value: number, y: number) => {
      const sliderGroup = new THREE.Group();
      sliderGroup.name = name;

      // Create slider track
      const trackGeometry = new THREE.PlaneGeometry(0.8, 0.05);
      const trackMaterial = new THREE.MeshBasicMaterial({ color: 0x505050 });
      const trackMesh = new THREE.Mesh(trackGeometry, trackMaterial);
      sliderGroup.add(trackMesh);

      // Create slider handle
      const handleGeometry = new THREE.SphereGeometry(0.03);
      const handleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const handleMesh = new THREE.Mesh(handleGeometry, handleMaterial);
      handleMesh.position.x = mapValue(value, min, max, -0.4, 0.4);
      sliderGroup.add(handleMesh);

      // Create label
      const labelGeometry = new THREE.PlaneGeometry(0.4, 0.1);
      const labelMaterial = new THREE.MeshBasicMaterial({ 
        map: createTextTexture(name) 
      });
      const labelMesh = new THREE.Mesh(labelGeometry, labelMaterial);
      labelMesh.position.set(-0.6, 0, 0);
      sliderGroup.add(labelMesh);

      sliderGroup.position.set(0, y, 0.01);
      panel.value.add(sliderGroup);
      controls.value.set(name, { group: sliderGroup, min, max, value });
    };

    const createColorPicker = (name: string, value: string, y: number) => {
      const pickerGroup = new THREE.Group();
      pickerGroup.name = name;

      // Create color swatch
      const swatchGeometry = new THREE.PlaneGeometry(0.1, 0.1);
      const swatchMaterial = new THREE.MeshBasicMaterial({ 
        color: new THREE.Color(value) 
      });
      const swatchMesh = new THREE.Mesh(swatchGeometry, swatchMaterial);
      pickerGroup.add(swatchMesh);

      // Create label
      const labelGeometry = new THREE.PlaneGeometry(0.4, 0.1);
      const labelMaterial = new THREE.MeshBasicMaterial({ 
        map: createTextTexture(name) 
      });
      const labelMesh = new THREE.Mesh(labelGeometry, labelMaterial);
      labelMesh.position.set(-0.3, 0, 0);
      pickerGroup.add(labelMesh);

      pickerGroup.position.set(0, y, 0.01);
      panel.value.add(pickerGroup);
      controls.value.set(name, { group: pickerGroup, value });
    };

    const updateControl = (name: string, value: number | string) => {
      const control = controls.value.get(name);
      if (!control) return;

      if ('min' in control && 'max' in control && typeof value === 'number') {
        // Slider
        const handle = control.group.children[1];
        handle.position.x = mapValue(value, control.min!, control.max!, -0.4, 0.4);
      } else if (typeof value === 'string') {
        // Color picker
        const swatch = control.group.children[0];
        (swatch.material as THREE.MeshBasicMaterial).color.set(value);
      }
      control.value = value;
    };

    const handleInteraction = (intersection: THREE.Intersection) => {
      const controlName = intersection.object.parent?.name;
      if (!controlName) return null;

      const control = controls.value.get(controlName);
      if (!control) return null;

      if ('min' in control && 'max' in control) {
        // Slider
        const newValue = mapValue(intersection.point.x, -0.4, 0.4, control.min!, control.max!);
        updateControl(controlName, newValue);
        emit('controlChange', { name: controlName, value: newValue });
        return { name: controlName, value: newValue };
      } else {
        // Color picker
        const colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff];
        const currentIndex = colors.indexOf(parseInt(control.value as string));
        const newValue = colors[(currentIndex + 1) % colors.length];
        const hexValue = '#' + newValue.toString(16).padStart(6, '0');
        updateControl(controlName, hexValue);
        emit('controlChange', { name: controlName, value: hexValue });
        return { name: controlName, value: hexValue };
      }
    };

    onMounted(() => {
      initPanel();

      // Initialize controls based on current settings
      const settings = settingsStore.getVisualizationSettings;
      let yPosition = 0.6;

      // Add sliders
      createSlider('Scale', 0.1, 2.0, settings.nodeScale || 1.0, yPosition);
      yPosition -= 0.2;
      createSlider('Opacity', 0, 1, settings.nodeOpacity || 1.0, yPosition);
      yPosition -= 0.2;

      // Add color pickers
      createColorPicker('Node Color', settings.nodeColor || '#ffffff', yPosition);
      yPosition -= 0.2;
      createColorPicker('Edge Color', settings.edgeColor || '#ffffff', yPosition);
    });

    onBeforeUnmount(() => {
      // Cleanup Three.js objects
      controls.value.forEach((control: Control) => {
        control.group.traverse((obj: THREE.Object3D) => {
          if (obj instanceof THREE.Mesh) {
            obj.geometry.dispose();
            if (Array.isArray(obj.material)) {
              obj.material.forEach(m => m.dispose());
            } else {
              obj.material.dispose();
            }
          }
        });
      });

      if (panel.value.parent) {
        panel.value.parent.remove(panel.value);
      }
    });

    return {
      panel,
      controls,
      handleInteraction,
      updateControl
    };
  }
});
</script>
