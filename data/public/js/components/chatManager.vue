   <script>
   import { defineComponent, onBeforeUnmount } from 'vue';

   export default defineComponent({
       name: 'ChatManager',
       props: {
           websocketService: {
               type: Object,
               required: true
           }
       },
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
                   this.websocketService.sendChatMessage({
                       message: this.chatInput,
                       useOpenAI: this.useOpenAI
                   });
                   this.chatMessages.push({ sender: 'You', message: this.chatInput });
                   this.chatInput = '';
               }
           },
           toggleTTS() {
               this.websocketService.toggleTTS(this.useOpenAI);
               console.log(`TTS method set to: ${this.useOpenAI ? 'OpenAI' : 'Sonata'}`);
           },
           receiveMessage(message) {
               this.chatMessages.push({ sender: 'AI', message });
           }
       },
       mounted() {
           this.websocketService.on('message', this.receiveMessage);
       },
       beforeUnmount() {
           // Remove the event listener when the component is unmounted
           this.websocketService.off('message', this.receiveMessage);
       }
   });
   </script>