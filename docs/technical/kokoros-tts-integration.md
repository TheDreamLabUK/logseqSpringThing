# Kokoros TTS Integration Plan

## Overview

This document outlines the detailed plan for integrating Kokoros TTS into our application, replacing the existing Sonata TTS implementation. The integration will be done in phases to ensure smooth transition and minimal disruption to the existing functionality.

## Phase 1: Server-Side Changes

### 1.1 Update Speech Types

First, we'll update the speech types in `src/types/speech.rs`:

```rust
#[derive(Debug, Clone)]
pub enum TTSProvider {
    OpenAI,
    Kokoros,  // Replacing Sonata
}

#[derive(Debug)]
pub enum SpeechCommand {
    Initialize,
    SendMessage(String),
    Close,
    SetTTSProvider(TTSProvider),
}

// Add Kokoros-specific error types
#[derive(Debug)]
pub enum SpeechError {
    WebSocketError(tungstenite::Error),
    ConnectionError(String),
    SendError(mpsc::error::SendError<SpeechCommand>),
    SerializationError(serde_json::Error),
    ProcessError(std::io::Error),
    Base64Error(base64::DecodeError),
    BroadcastError(String),
    TTSError(String),
    KokorosError(String),  // New error type for Kokoros
}
```

### 1.2 Implement Kokoros Client

Create a new module `src/services/kokoros_client.rs`:

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use crate::types::speech::SpeechError;

#[derive(Debug, Serialize)]
struct TTSRequest {
    text: String,
    voice: String,  // Optional: Configure based on settings
}

#[derive(Debug, Deserialize)]
struct TTSResponse {
    audio: Vec<u8>,
    success: bool,
}

pub struct KokorosClient {
    client: Client,
    endpoint: String,
}

impl KokorosClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            endpoint: "http://localhost:4001/v1/audio/tts".to_string(),
        }
    }

    pub async fn generate_speech(&self, text: String) -> Result<Vec<u8>, SpeechError> {
        let request = TTSRequest {
            text,
            voice: "default".to_string(),
        };

        let response = self.client
            .post(&self.endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|e| SpeechError::KokorosError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(SpeechError::KokorosError(
                format!("Kokoros API error: {}", response.status())
            ));
        }

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| SpeechError::KokorosError(e.to_string()))?;

        Ok(audio_data.to_vec())
    }
}
```

### 1.3 Update Speech Service

Modify `src/services/speech_service.rs` to integrate Kokoros:

```rust
use super::kokoros_client::KokorosClient;

pub struct SpeechService {
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    settings: Arc<RwLock<Settings>>,
    tts_provider: Arc<RwLock<TTSProvider>>,
    kokoros_client: KokorosClient,
}

impl SpeechService {
    pub fn new(settings: Arc<RwLock<Settings>>) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));

        let service = SpeechService {
            sender,
            settings,
            tts_provider: Arc::new(RwLock::new(TTSProvider::Kokoros)),
            kokoros_client: KokorosClient::new(),
        };

        service.start(rx);
        service
    }

    // Update message handling to use Kokoros
    async fn handle_tts(&self, text: String) -> Result<Vec<u8>, SpeechError> {
        match *self.tts_provider.read().await {
            TTSProvider::Kokoros => {
                self.kokoros_client.generate_speech(text).await
            },
            TTSProvider::OpenAI => {
                // Existing OpenAI implementation
            }
        }
    }
}
```

## Phase 2: Client-Side Implementation

### 2.1 Create Audio Player

Create new file `client/audio/AudioPlayer.ts`:

```typescript
export class AudioPlayer {
    private static instance: AudioPlayer | null = null;
    private audioContext: AudioContext;
    private audioQueue: AudioBuffer[] = [];
    private isPlaying: boolean = false;
    private currentSource: AudioBufferSourceNode | null = null;

    private constructor() {
        this.audioContext = new AudioContext();
    }

    public static getInstance(): AudioPlayer {
        if (!AudioPlayer.instance) {
            AudioPlayer.instance = new AudioPlayer();
        }
        return AudioPlayer.instance;
    }

    public async handleAudioChunk(data: ArrayBuffer, isLastChunk: boolean): Promise<void> {
        try {
            const audioBuffer = await this.audioContext.decodeAudioData(data);
            this.audioQueue.push(audioBuffer);

            if (!this.isPlaying) {
                this.playNextChunk();
            }
        } catch (error) {
            console.error('Error processing audio chunk:', error);
        }
    }

    private playNextChunk(): void {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        this.isPlaying = true;
        const buffer = this.audioQueue.shift()!;
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        source.onended = () => {
            this.playNextChunk();
        };

        this.currentSource = source;
        source.start(0);
    }

    public resume(): void {
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }

    public pause(): void {
        if (this.audioContext.state === 'running') {
            this.audioContext.suspend();
        }
    }

    public stop(): void {
        if (this.currentSource) {
            this.currentSource.stop();
            this.currentSource = null;
        }
        this.audioQueue = [];
        this.isPlaying = false;
    }
}
```

### 2.2 Update WebSocket Types

Create/update `client/types/websocket.ts`:

```typescript
export enum MessageType {
    POSITION_VELOCITY_UPDATE = 0x01,
    VOICE_DATA = 0x02,
}

export interface VoiceData {
    chunkId: number;
    isFinal: boolean;
    data: ArrayBuffer;
}

export interface VoiceDataMessage {
    type: MessageType.VOICE_DATA;
    data: VoiceData;
}
```

### 2.3 Update WebSocket Service

Modify `client/websocket/websocketService.ts`:

```typescript
import { AudioPlayer } from '../audio/AudioPlayer';
import { MessageType, VoiceData } from '../types/websocket';

export class WebSocketService {
    private readonly MessageType = {
        PositionVelocityUpdate: 0x01,
        VoiceData: 0x02
    } as const;

    private handleBinaryMessage(buffer: ArrayBuffer): void {
        try {
            const decompressedBuffer = this.tryDecompress(buffer);
            const dataView = new DataView(decompressedBuffer);
            let offset = 0;

            const messageType = dataView.getUint32(offset, true);
            offset += 4;

            switch (messageType) {
                case this.MessageType.PositionVelocityUpdate:
                    // Existing position update handling
                    break;

                case this.MessageType.VoiceData:
                    this.handleVoiceData(decompressedBuffer.slice(4));
                    break;

                default:
                    logger.warn('Unexpected binary message type:', messageType);
            }
        } catch (error) {
            logger.error('Failed to process binary message:', error);
        }
    }

    private handleVoiceData(buffer: ArrayBuffer): void {
        const dataView = new DataView(buffer);
        let offset = 0;

        const chunkId = dataView.getUint32(offset, true);
        offset += 4;

        const isFinal = dataView.getUint8(offset) === 1;
        offset += 1;

        const audioData = buffer.slice(offset);
        
        AudioPlayer.getInstance().handleAudioChunk(audioData, isFinal);
    }
}
```

## Phase 3: Build System Updates

### 3.1 Update Dockerfile

Remove Python-related sections from the Dockerfile:

```dockerfile
# Remove these sections
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
```

### 3.2 Update docker-compose.yml

Add Kokoros service dependency:

```yaml
services:
  webxr:
    depends_on:
      - kokoros
    # ... rest of webxr service config

  kokoros:
    image: kokoros
    ports:
      - "4001:4001"
    networks:
      - docker_ragflow
```

## Phase 4: Testing Strategy

### 4.1 Unit Tests

Create test files for each new component:

```typescript
// tests/audio/AudioPlayer.test.ts
describe('AudioPlayer', () => {
    let audioPlayer: AudioPlayer;

    beforeEach(() => {
        audioPlayer = AudioPlayer.getInstance();
    });

    it('should handle audio chunks correctly', async () => {
        // Test implementation
    });

    it('should manage audio queue properly', async () => {
        // Test implementation
    });
});
```

### 4.2 Integration Tests

Create end-to-end tests:

```typescript
// tests/integration/tts.test.ts
describe('TTS Integration', () => {
    it('should stream audio from Kokoros to client', async () => {
        // Test implementation
    });

    it('should handle TTS errors gracefully', async () => {
        // Test implementation
    });
});
```

## Phase 5: Deployment

### 5.1 Pre-deployment Checklist

1. Verify Kokoros container is running and accessible
2. Confirm all Python/Sonata components are removed
3. Test audio streaming in development environment
4. Verify WebSocket connections and error handling
5. Check resource usage and performance

### 5.2 Deployment Steps

1. Deploy updated Dockerfile and docker-compose.yml
2. Start Kokoros container
3. Deploy application updates
4. Monitor logs for any issues
5. Verify audio functionality in production

### 5.3 Rollback Plan

1. Keep backup of Sonata implementation
2. Maintain ability to switch TTS providers
3. Document rollback procedure

## Phase 6: Documentation Updates

1. Update API documentation
2. Update deployment guides
3. Add Kokoros configuration documentation
4. Update troubleshooting guides

## Timeline

1. Phase 1 (Server-Side Changes): 2-3 days
2. Phase 2 (Client-Side Implementation): 2-3 days
3. Phase 3 (Build System Updates): 1 day
4. Phase 4 (Testing): 2-3 days
5. Phase 5 (Deployment): 1-2 days
6. Phase 6 (Documentation): 1 day

Total estimated time: 9-13 days