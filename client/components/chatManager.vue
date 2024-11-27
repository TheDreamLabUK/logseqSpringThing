<template>
  <div class="chat-manager">
    <div class="chat-messages" ref="messagesContainer">
      <div v-for="(message, index) in messages" :key="index" class="message">
        {{ message }}
      </div>
    </div>
    <div class="chat-input">
      <input
        v-model="chatInput"
        @keyup.enter="sendMessage"
        placeholder="Type your message..."
        :disabled="!websocketService"
      />
      <div class="button-group">
        <button @click="sendMessage" :disabled="!websocketService || !chatInput.trim()">
          Send
        </button>
        <button 
          @click="toggleTTS" 
          :disabled="!websocketService"
          :class="{ active: useOpenAI }"
          :title="useOpenAI ? 'Using OpenAI for TTS' : 'Using Sonata for TTS'"
        >
          TTS: {{ useOpenAI ? 'OpenAI' : 'Sonata' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, PropType, onMounted, onBeforeUnmount, nextTick } from 'vue'
import type WebsocketService from '../services/websocketService'

export default defineComponent({
  name: 'ChatManager',
  
  props: {
    websocketService: {
      type: Object as PropType<WebsocketService>,
      required: true
    }
  },

  setup(props) {
    const chatInput = ref('')
    const messages = ref<string[]>([])
    const useOpenAI = ref(false)
    const messagesContainer = ref<HTMLElement | null>(null)

    const scrollToBottom = async () => {
      await nextTick()
      if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
      }
    }

    const handleRagflowAnswer = (answer: string) => {
      messages.value.push(`AI: ${answer}`)
      scrollToBottom()
    }

    const handleOpenAIResponse = (response: string) => {
      messages.value.push(`OpenAI: ${response}`)
      scrollToBottom()
    }

    const sendMessage = () => {
      if (chatInput.value.trim()) {
        messages.value.push(`You: ${chatInput.value}`)
        props.websocketService.send({
          type: 'chatMessage',
          message: chatInput.value,
          useOpenAI: useOpenAI.value
        })
        chatInput.value = ''
        scrollToBottom()
      }
    }

    const toggleTTS = () => {
      useOpenAI.value = !useOpenAI.value
      props.websocketService.send({
        type: 'setTTSMethod',
        useOpenAI: useOpenAI.value
      })
      console.log(`TTS method set to: ${useOpenAI.value ? 'OpenAI' : 'Sonata'}`)
    }

    onMounted(() => {
      props.websocketService.on('ragflowAnswer', handleRagflowAnswer)
      props.websocketService.on('openaiResponse', handleOpenAIResponse)
    })

    onBeforeUnmount(() => {
      props.websocketService.off('ragflowAnswer', handleRagflowAnswer)
      props.websocketService.off('openaiResponse', handleOpenAIResponse)
    })

    return {
      chatInput,
      messages,
      useOpenAI,
      messagesContainer,
      sendMessage,
      toggleTTS
    }
  }
})
</script>

<style scoped>
.chat-manager {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 300px;
  height: 400px;
  background: rgba(0, 0, 0, 0.8);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  padding: 10px;
  z-index: 1000;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.chat-messages {
  flex-grow: 1;
  overflow-y: auto;
  margin-bottom: 10px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.message {
  margin: 8px 0;
  padding: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  color: #fff;
  word-wrap: break-word;
}

.chat-input {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chat-input input {
  width: 100%;
  padding: 8px;
  border: none;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 14px;
}

.chat-input input::placeholder {
  color: rgba(255, 255, 255, 0.5);
}

.chat-input input:focus {
  outline: none;
  background: rgba(255, 255, 255, 0.15);
}

.button-group {
  display: flex;
  gap: 8px;
}

button {
  flex: 1;
  padding: 8px;
  border: none;
  border-radius: 4px;
  background: #2196f3;
  color: white;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:hover:not(:disabled) {
  background: #1976d2;
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
  opacity: 0.7;
}

button.active {
  background: #4caf50;
}

button.active:hover:not(:disabled) {
  background: #388e3c;
}

/* Scrollbar styling */
.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}
</style>
