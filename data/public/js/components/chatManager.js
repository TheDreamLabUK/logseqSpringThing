// public/js/components/chatManager.js

import Vue from 'vue';
import { WebsocketService } from '../services/websocketService.js';

export default Vue.component('ChatManager', {
    props: ['websocketService'],
    data() {
        return {
            chatInput: '',
            chatMessages: [],
            useOpenAI: false, // State for TTS toggle
        };
    },
    methods: {
        sendMessage() {
            if (this.chatInput.trim()) {
                this.websocketService.sendChatMessage(this.chatInput, this.useOpenAI);
                this.chatMessages.push({ sender: 'user', message: this.chatInput });
                this.chatInput = '';
            }
        },
        toggleTTS() {
            this.websocketService.toggleTTS(this.useOpenAI);
            console.log(`TTS method set to: ${this.useOpenAI ? 'OpenAI' : 'Sonata'}`);
        },
        receiveMessage(message) {
            this.chatMessages.push({ sender: 'bot', message });
        }
    },
    mounted() {
        this.websocketService.on('message', this.receiveMessage);
    },
    template: `
        <div id="chat-container">
            <div id="chat-messages">
                <div v-for="(msg, index) in chatMessages" :key="index">
                    <strong>{{ msg.sender }}:</strong> {{ msg.message }}
                </div>
            </div>
            <div id="chat-input-area">
                <input 
                    type="text" 
                    v-model="chatInput" 
                    placeholder="Ask a question..." 
                    @keypress.enter="sendMessage"
                >
                <button id="send-button" @click="sendMessage">Send</button>
            </div>
            <!-- Toggle Switch for TTS -->
            <div class="tts-toggle">
                <label for="tts-switch">Enable OpenAI TTS</label>
                <input 
                    type="checkbox" 
                    id="tts-switch" 
                    v-model="useOpenAI" 
                    @change="toggleTTS" 
                />
            </div>
        </div>
    `
});