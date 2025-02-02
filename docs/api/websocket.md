# WebSocket API Documentation

[Previous content up to the Binary Messages section remains the same...]

## Voice Interaction Protocol

### Voice System Configuration
```json
{
  "type": "voice_config",
  "data": {
    "provider": "openai | local",
    "language": "en-US",
    "voice_id": "voice_identifier",
    "streaming": true
  }
}
```

### Speech-to-Text Messages

#### Start Speech Recognition
```json
{
  "type": "start_recognition",
  "data": {
    "session_id": "abc123",
    "language": "en-US",
    "provider": "openai | local"
  }
}
```

#### Speech Data (Binary)
Speech audio data is sent as binary WebSocket messages with the following format:
```
[2 bytes] message_type (0x02 for speech data)
[4 bytes] session_id
[N bytes] audio_data (16-bit PCM, 16kHz sample rate)
```

#### Recognition Result
```json
{
  "type": "recognition_result",
  "data": {
    "session_id": "abc123",
    "text": "recognized text",
    "confidence": 0.95,
    "is_final": true
  }
}
```

### Text-to-Speech Messages

#### Start Speech Synthesis
```json
{
  "type": "start_synthesis",
  "data": {
    "session_id": "abc123",
    "text": "Text to synthesize",
    "provider": "openai | local",
    "voice_id": "voice_identifier",
    "language": "en-US"
  }
}
```

#### Speech Audio (Binary)
Synthesized speech is streamed as binary WebSocket messages:
```
[2 bytes] message_type (0x03 for synthesized speech)
[4 bytes] session_id
[4 bytes] chunk_index
[4 bytes] total_chunks
[N bytes] audio_data (16-bit PCM, 24kHz sample rate)
```

#### Synthesis Status
```json
{
  "type": "synthesis_status",
  "data": {
    "session_id": "abc123",
    "status": "completed | error",
    "progress": 0.75,
    "error_message": "error details if status is error"
  }
}
```

### Provider-Specific Features

#### OpenAI Voice (Current Implementation)
- Real-time streaming synthesis
- High-quality voice models
- Natural prosody and intonation
- Cloud-based processing

Example configuration:
```json
{
  "type": "voice_config",
  "data": {
    "provider": "openai",
    "model": "tts-1",
    "voice_id": "alloy",
    "streaming": true,
    "api_version": "2024-02"
  }
}
```

#### Local GPU-Accelerated System (Planned)
- Kororo for text-to-speech
- Whisper for speech-to-text
- Full GPU acceleration
- Offline operation

Example configuration:
```json
{
  "type": "voice_config",
  "data": {
    "provider": "local",
    "tts_engine": "kororo",
    "stt_engine": "whisper",
    "model_size": "medium",
    "gpu_device": 0,
    "streaming": true
  }
}
```

### Error Handling

#### Recognition Error
```json
{
  "type": "recognition_error",
  "data": {
    "session_id": "abc123",
    "error_code": "audio_decode_failed",
    "message": "Failed to decode audio stream",
    "recoverable": true
  }
}
```

#### Synthesis Error
```json
{
  "type": "synthesis_error",
  "data": {
    "session_id": "abc123",
    "error_code": "synthesis_failed",
    "message": "Failed to synthesize speech",
    "recoverable": false
  }
}
```

### Performance Considerations

#### OpenAI Provider
- Requires stable internet connection
- Streaming reduces latency
- Higher quality at the cost of network dependency
- API rate limits apply

#### Local Provider
- GPU acceleration for real-time processing
- No network latency
- Consistent performance
- Resource usage scales with model size

Example resource configuration:
```json
{
  "type": "voice_resource_config",
  "data": {
    "provider": "local",
    "gpu_memory_limit": "2GB",
    "batch_size": 16,
    "stream_buffer_size": 4096,
    "max_concurrent_sessions": 4
  }
}
```

[Rest of the WebSocket documentation remains the same...]