<template>
  <Transition name="fade">
    <div 
      v-if="error"
      class="error-message"
      @click="dismiss"
    >
      {{ error }}
    </div>
  </Transition>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, onBeforeUnmount } from 'vue';

interface Props {
  message: string;
  duration: number;
  dismissible: boolean;
}

export default defineComponent({
  name: 'ErrorMessage',
  
  props: {
    message: {
      type: String,
      required: true
    },
    duration: {
      type: Number,
      default: 5000
    },
    dismissible: {
      type: Boolean,
      default: true
    }
  },

  emits: ['dismiss'],

  setup(props: Props, { emit }: { emit: (event: 'dismiss') => void }) {
    const error = ref<string | null>(props.message);
    let timeout: number | null = null;

    const dismiss = () => {
      if (props.dismissible) {
        error.value = null;
        emit('dismiss');
      }
    };

    onMounted(() => {
      if (props.duration > 0) {
        timeout = window.setTimeout(() => {
          error.value = null;
          emit('dismiss');
        }, props.duration);
      }
    });

    onBeforeUnmount(() => {
      if (timeout) {
        clearTimeout(timeout);
      }
    });

    return {
      error,
      dismiss
    };
  }
});
</script>

<style scoped>
.error-message {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: rgba(255, 0, 0, 0.85);
  color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  cursor: pointer;
  max-width: 80vw;
  text-align: center;
  font-weight: 500;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
