# Voice System Integration

This document describes the voice-to-voice interaction system integrated into LogseqSpringThing, enabling real-time speech-to-text (STT) and text-to-speech (TTS) capabilities.

## Architecture Overview

The voice system consists of three main components:

1. **Client-Side (TypeScript/React)**
   - `AudioInputService`: Manages microphone capture and audio streaming
   - `AudioOutputService`: Handles audio playback with queue management
   - `VoiceWebSocketService`: WebSocket client for bidirectional audio communication
   - Voice UI Components: `VoiceButton` and `VoiceIndicator`

2. **Backend (Rust/Actix)**
   - `/ws/speech` WebSocket endpoint for audio streaming
   - `SpeechService`: Orchestrates TTS and STT operations
   - Support for multiple providers (Kokoro TTS, Whisper STT)
   - Default providers: Kokoro for TTS, Whisper for STT

3. **External Services**
   - Kokoro TTS API (containerized)
   - Whisper STT API (fully integrated and operational at configurable endpoint)

## Audio Flow Diagram

```
┌─────────────┐     Audio Stream    ┌──────────────┐     HTTP/WS     ┌─────────────┐
│   Browser   │ ──────────────────> │   Backend    │ ──────────────> │   Whisper   │
│             │                      │   (/ws/      │                 │   Service   │
│ Microphone  │                      │   speech)    │                 └─────────────┘
└─────────────┘                      │              │                         │
       │                             │              │ <───────────────────────┘
       │                             │              │     Transcription
       │                             │              │
       v                             │              │                 ┌─────────────┐
┌─────────────┐     Audio Playback   │              │ ──────────────> │   Kokoro    │
│   Browser   │ <────────────────── │              │     TTS Request │   Service   │
│             │                      │              │                 └─────────────┘
│   Speaker   │                      │              │ <───────────────────────┘
└─────────────┘                      └──────────────┘     Audio Stream
```

## WebSocket Protocol

### Connection
- Endpoint: `ws://[host]/ws/speech`
- Heartbeat: 5-second ping/pong with 10-second timeout

### Message Types

#### Client → Server

1. **TTS Request**
```json
{
  "type": "tts",
  "text": "Hello, world!",
  "voice": "af_heart",     // optional
  "speed": 1.0,            // optional
  "stream": true           // optional
}
```

2. **STT Control**
```json
{
  "type": "stt",
  "action": "start",       // or "stop"
  "language": "en",        // optional
  "model": "whisper-1"     // optional
}
```

3. **Audio Data**
- Binary WebSocket frames containing audio chunks
- Format: `audio/webm;codecs=opus` (preferred)
- Sample rate: 48kHz, mono

#### Server → Client

1. **Connection Established**
```json
{
  "type": "connected",
  "message": "Connected to speech service"
}
```

2. **Transcription Result**
```json
{
  "type": "transcription",
  "data": {
    "text": "Hello, world!",
    "isFinal": true,
    "timestamp": 1234567890123
  }
}
```

3. **Audio Data**
- Binary WebSocket frames containing TTS audio
- Format: MP3 (default) or as configured

4. **Error**
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Client API Usage

### Basic Voice Interaction

```typescript
import { VoiceWebSocketService } from './services/VoiceWebSocketService';

// Get service instance
const voiceService = VoiceWebSocketService.getInstance();

// Connect to voice service
await voiceService.connectToSpeech('http://localhost:3000');

// Send text for TTS
await voiceService.sendTextForTTS({
  text: "Hello, I'm your AI assistant",
  voice: "af_heart",
  speed: 1.0,
  stream: true
});

// Start voice input (STT)
await voiceService.startAudioStreaming({
  language: "en"
});

// Listen for transcriptions
voiceService.on('transcription', (result) => {
  console.log('User said:', result.text);
});

// Stop voice input
voiceService.stopAudioStreaming();
```

### UI Components

```tsx
import { VoiceButton, VoiceIndicator } from './components';

// Voice control button
<VoiceButton 
  size="md" 
  variant="primary"
  className="my-voice-btn"
/>

// Voice status indicator with transcription display
<VoiceIndicator 
  showTranscription={true}
  showStatus={true}
/>
```

## Backend Configuration

### Settings Structure

```rust
// In settings.toml or environment variables

[kokoro]
api_url = "http://kokoro-service:8080"
default_voice = "af_heart"
default_speed = 1.0
default_format = "mp3"
stream = true

[whisper]
api_url = "http://whisper-service:8000"  // Configurable endpoint
model = "whisper-1"
default_language = "en"
```

### Docker Services

The voice services run within the Docker network:

```yaml
# docker-compose.yml excerpt
services:
  kokoro:
    image: kokoro-tts:latest
    ports:
      - "8080:8080"
    networks:
      - ragflow
      
  whisper:
    image: openai/whisper:latest
    ports:
      - "8000:8000"
    networks:
      - ragflow
```

## Implementation Status

### ✅ Completed
- TTS Backend with Kokoro integration
- STT Backend with Whisper integration
- WebSocket endpoint (`/ws/speech`)
- Audio streaming infrastructure
- Client-side audio services
- Voice UI components
- Full speech service architecture with provider switching

### 🚧 In Progress
- Full duplex audio communication optimization
- Voice activity detection (VAD)
- Enhanced error recovery for streaming

### 📋 Planned
- Multiple language support
- Voice command processing
- Audio visualizations
- Push-to-talk and hotkey support
- Noise gate and echo cancellation

## API Reference

### AudioInputService

```typescript
class AudioInputService {
  // Request microphone access
  requestMicrophoneAccess(constraints?: AudioConstraints): Promise<boolean>
  
  // Start/stop recording
  startRecording(mimeType?: string): Promise<void>
  stopRecording(): void
  
  // Audio level monitoring
  getAudioLevel(): number  // 0-1
  getFrequencyData(): Uint8Array
  
  // Events
  on('audioChunk', (chunk: AudioChunk) => void)
  on('audioLevel', (level: number) => void)
  on('stateChange', (state: AudioInputState) => void)
}
```

### AudioOutputService

```typescript
class AudioOutputService {
  // Queue audio for playback
  queueAudio(audioData: ArrayBuffer, id?: string): Promise<void>
  
  // Playback control
  stop(): void
  pause(): void
  resume(): void
  
  // Volume control
  setVolume(volume: number): void  // 0-1
  getVolume(): number
  
  // Events
  on('audioStarted', (item: AudioQueueItem) => void)
  on('audioEnded', (item: AudioQueueItem) => void)
  on('stateChange', (state: AudioOutputState) => void)
}
```

### VoiceWebSocketService

```typescript
class VoiceWebSocketService extends WebSocketService {
  // Connection
  connectToSpeech(baseUrl: string): Promise<void>
  
  // TTS
  sendTextForTTS(request: TTSRequest): Promise<void>
  
  // STT
  startAudioStreaming(options?: { language?: string }): Promise<void>
  stopAudioStreaming(): void
  
  // Events
  on('voiceConnected', (data: any) => void)
  on('transcription', (result: TranscriptionResult) => void)
  on('audioReceived', (buffer: ArrayBuffer) => void)
}
```

## Testing

### Manual Testing

1. **Test TTS**:
   ```bash
   # Send test message via WebSocket
   wscat -c ws://localhost:3000/ws/speech
   > {"type":"tts","text":"Hello world"}
   ```

2. **Test Audio Capture**:
   - Click the voice button in the UI
   - Check browser console for audio level logs
   - Verify microphone permission prompt

3. **Test End-to-End**:
   - Open the application
   - Click voice button to start recording
   - Speak a phrase
   - Verify transcription appears
   - System responds with TTS audio

### Integration Tests

See `tests/voice_integration_test.rs` for backend tests.

## Troubleshooting

### Common Issues

1. **No Audio Output**
   - Check Kokoro service is running: `docker ps | grep kokoro`
   - Verify audio format compatibility
   - Check browser audio permissions

2. **Microphone Not Working**
   - Ensure HTTPS or localhost (required for getUserMedia)
   - Check browser microphone permissions
   - Verify AudioContext is not suspended

3. **WebSocket Connection Failed**
   - Check `/ws/speech` endpoint is accessible
   - Verify CORS settings
   - Check for proxy/firewall issues

4. **Transcription Not Working**
   - Whisper service deployment pending
   - Check audio format compatibility
   - Verify audio data is being sent

### Debug Logging

Enable debug logs:
```bash
# Backend
RUST_LOG=debug cargo run

# Frontend
localStorage.setItem('debug', 'voice:*')
```