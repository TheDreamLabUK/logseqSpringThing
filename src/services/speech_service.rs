use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream, tungstenite};
use tungstenite::http::Request;
use serde_json::json;
use std::sync::Arc;
use tokio::task;
use tokio::sync::broadcast;
use crate::config::AppFullSettings;
// use crate::config::Settings; // AppFullSettings is used from self.settings
use log::{info, error, debug};
use futures::{SinkExt, StreamExt};
use std::error::Error;
use tokio::net::TcpStream;
use url::Url;
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD as BASE64};
use crate::types::speech::{SpeechError, SpeechCommand, TTSProvider, STTProvider, SpeechOptions, TranscriptionOptions};
use reqwest::Client;


/// Centralized speech service managing both Text-to-Speech (TTS) and Speech-to-Text (STT) operations
///
/// This service orchestrates real-time voice interactions by:
/// - Managing TTS via Kokoro API for generating speech from text
/// - Managing STT via Whisper API for transcribing audio to text
/// - Broadcasting audio and transcription data to multiple WebSocket clients
/// - Handling provider switching and configuration management
///
/// The service uses async channels for command processing and broadcast channels
/// for distributing results to multiple subscribers simultaneously.
pub struct SpeechService {
    /// Command sender for internal message passing to the service task
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    /// Shared application settings containing API configurations
    settings: Arc<RwLock<AppFullSettings>>,
    /// Current Text-to-Speech provider (Kokoro, OpenAI, etc.)
    tts_provider: Arc<RwLock<TTSProvider>>,
    /// Current Speech-to-Text provider (Whisper, OpenAI, etc.)
    stt_provider: Arc<RwLock<STTProvider>>,
    /// Broadcast channel for distributing TTS audio data to all connected WebSocket clients
    /// Buffer size of 100 allows multiple clients without blocking
    audio_tx: broadcast::Sender<Vec<u8>>,
    /// Broadcast channel for distributing STT transcription results to all connected clients
    /// Each transcription result is sent as a String to all subscribers
    transcription_tx: broadcast::Sender<String>,
    /// Shared HTTP client for making API requests to external services (Kokoro, Whisper)
    /// Reused across all requests for connection pooling and efficiency
    http_client: Arc<Client>,
}

impl SpeechService {
    /// Creates a new SpeechService instance with default configurations
    ///
    /// # Arguments
    /// * `settings` - Shared application settings containing API configurations for TTS/STT providers
    ///
    /// # Returns
    /// * `SpeechService` - A new service instance ready for speech operations
    ///
    /// # Behavior
    /// - Initializes internal command channel with buffer size of 100 commands
    /// - Creates broadcast channels for audio (TTS output) and transcriptions (STT output)
    /// - Sets up shared HTTP client for efficient API communication
    /// - Defaults to Kokoro TTS and Whisper STT providers
    /// - Automatically starts the internal service task for command processing
    ///
    /// # Channel Buffers
    /// - Command channel: 100 commands (prevents blocking on rapid command submission)
    /// - Audio broadcast: 100 audio chunks (handles multiple clients with buffering)
    /// - Transcription broadcast: 100 transcriptions (handles multiple clients with buffering)
    pub fn new(settings: Arc<RwLock<AppFullSettings>>) -> Self {
        // Create internal command channel for async command processing
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));

        // Create broadcast channel for TTS audio data with buffer size of 100
        // This allows multiple WebSocket clients to receive the same audio simultaneously
        let (audio_tx, _) = broadcast::channel(100);

        // Create shared HTTP client for API requests to external services
        // Reuses connections for better performance across multiple requests
        let http_client = Arc::new(Client::new());

        // Create broadcast channel for STT transcription results
        // Multiple clients can subscribe to receive transcription text
        let (transcription_tx, _) = broadcast::channel(100);

        let service = SpeechService {
            sender,
            settings,
            tts_provider: Arc::new(RwLock::new(TTSProvider::Kokoro)), // Default to Kokoro for TTS
            stt_provider: Arc::new(RwLock::new(STTProvider::Whisper)), // Default to Whisper for STT
            audio_tx,
            transcription_tx,
            http_client,
        };

        // Start the internal service task for async command processing
        service.start(rx);
        service
    }

    fn start(&self, mut receiver: mpsc::Receiver<SpeechCommand>) {
        let settings: Arc<RwLock<AppFullSettings>> = Arc::clone(&self.settings);
        let http_client = Arc::clone(&self.http_client);
        let tts_provider = Arc::clone(&self.tts_provider);
        let stt_provider = Arc::clone(&self.stt_provider);
        let audio_tx = self.audio_tx.clone();
        let transcription_tx = self.transcription_tx.clone();

        task::spawn(async move {
            let mut ws_stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>> = None;

            while let Some(command) = receiver.recv().await {
                match command {
                    SpeechCommand::Initialize => {
                        let settings_read = settings.read().await;

                        // Safely get OpenAI API key
                        let openai_api_key = match settings_read.openai.as_ref().and_then(|o| o.api_key.as_ref()) {
                            Some(key) if !key.is_empty() => key.clone(),
                            _ => {
                                error!("OpenAI API key not configured or empty. Cannot initialize OpenAI Realtime API.");
                                continue; // Skip initialization if key is missing
                            }
                        };

                        let url_str = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01";
                        let url = match Url::parse(url_str) {
                            Ok(url) => url,
                            Err(e) => {
                                error!("Failed to parse OpenAI URL '{}': {}", url_str, e);
                                continue;
                            }
                        };

                        let request = match Request::builder()
                            .uri(url.as_str())
                            .header("Authorization", format!("Bearer {}", openai_api_key))
                            .header("OpenAI-Beta", "realtime=v1")
                            .header("Content-Type", "application/json")
                            .header("User-Agent", "WebXR Graph")
                            .header("Sec-WebSocket-Version", "13")
                            .header("Sec-WebSocket-Key", tungstenite::handshake::client::generate_key())
                            .header("Connection", "Upgrade")
                            .header("Upgrade", "websocket")
                            .body(()) {
                                Ok(req) => req,
                                Err(e) => {
                                    error!("Failed to build request: {}", e);
                                    continue;
                                }
                            };

                        match connect_async(request).await {
                            Ok((mut stream, _)) => {
                                info!("Connected to OpenAI Realtime API");

                                let init_event = json!({
                                    "type": "response.create",
                                    "response": {
                                        "modalities": ["text", "audio"],
                                        "instructions": "You are a helpful AI assistant. Respond naturally and conversationally."
                                    }
                                });

                                if let Err(e) = stream.send(tungstenite::Message::Text(init_event.to_string())).await {
                                    error!("Failed to send initial response.create event: {}", e);
                                    continue;
                                }

                                ws_stream = Some(stream);
                            },
                            Err(e) => error!("Failed to connect to OpenAI Realtime API: {}", e),
                        }
                    },
                    SpeechCommand::SendMessage(msg) => {
                        if let Some(stream) = &mut ws_stream {
                            let msg_event = json!({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": msg
                                    }]
                                }
                            });

                            if let Err(e) = stream.send(tungstenite::Message::Text(msg_event.to_string())).await {
                                error!("Failed to send message to OpenAI: {}", e);
                                continue;
                            }

                            let response_event = json!({
                                "type": "response.create"
                            });

                            if let Err(e) = stream.send(tungstenite::Message::Text(response_event.to_string())).await {
                                error!("Failed to request response from OpenAI: {}", e);
                                continue;
                            }

                            while let Some(message) = stream.next().await {
                                match message {
                                    Ok(tungstenite::Message::Text(text)) => {
                                        let event = match serde_json::from_str::<serde_json::Value>(&text) {
                                            Ok(event) => event,
                                            Err(e) => {
                                                error!("Failed to parse server event: {}", e);
                                                continue;
                                            }
                                        };

                                        match event["type"].as_str() {
                                            Some("conversation.item.created") => {
                                                if let Some(content) = event["item"]["content"].as_array() {
                                                    for item in content {
                                                        if item["type"] == "audio" {
                                                            if let Some(audio_data) = item["audio"].as_str() {
                                                                match BASE64.decode(audio_data) {
                                                                    Ok(audio_bytes) => {
                                                                        debug!("Received audio data of size: {}", audio_bytes.len());
                                                                    },
                                                                    Err(e) => error!("Failed to decode audio data: {}", e),
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            Some("error") => {
                                                error!("OpenAI Realtime API error: {:?}", event);
                                                break;
                                            },
                                            Some("response.completed") => break,
                                            _ => {}
                                        }
                                    },
                                    Ok(tungstenite::Message::Close(_)) => break,
                                    Err(e) => {
                                        error!("Error receiving from OpenAI: {}", e);
                                        break;
                                    },
                                    _ => {}
                                }
                            }
                        } else {
                            error!("OpenAI WebSocket not initialized");
                        }
                    },
                    SpeechCommand::Close => {
                        if let Some(mut stream) = ws_stream.take() {
                            if let Err(e) = stream.send(tungstenite::Message::Close(None)).await {
                                error!("Failed to send close frame: {}", e);
                            }
                        }
                        break;
                    },
                    SpeechCommand::SetTTSProvider(provider) => {
                        let mut current_provider = tts_provider.write().await;
                        *current_provider = provider.clone();
                        info!("TTS provider updated to: {:?}", provider);
                    },
                    SpeechCommand::TextToSpeech(text, options) => {
                        let provider = tts_provider.read().await.clone();

                        match provider {
                            TTSProvider::OpenAI => {
                                info!("TextToSpeech command with OpenAI provider not implemented");
                            },
                            TTSProvider::Kokoro => {
                                info!("Processing TextToSpeech command with Kokoro provider");
                                let kokoro_config = {
                                    let s = settings.read().await;
                                    s.kokoro.clone()
                                };

                                if let Some(config) = kokoro_config {
                                    let api_url_base = match config.api_url.as_deref() {
                                        Some(url) if !url.is_empty() => url,
                                        _ => {
                                            error!("Kokoro API URL not configured or empty.");
                                            continue;
                                        }
                                    };
                                    let api_url = format!("{}/v1/audio/speech", api_url_base.trim_end_matches('/'));
                                    info!("Sending TTS request to Kokoro API: {}", api_url);

                                    let response_format = config.default_format.as_deref().unwrap_or("mp3");

                                    let request_body = json!({
                                        "model": "kokoro",
                                        "input": text,
                                        "voice": options.voice.clone(),
                                        "response_format": response_format,
                                        "speed": options.speed,
                                        "stream": options.stream
                                    });

                                    let response = match http_client
                                        .post(&api_url)
                                        .header("Content-Type", "application/json")
                                        .body(request_body.to_string())
                                        .send()
                                        .await
                                    {
                                        Ok(response) => {
                                            if !response.status().is_success() {
                                                let status = response.status();
                                                let error_text = response.text().await.unwrap_or_default();
                                                error!("Kokoro API error {}: {}", status, error_text);
                                                continue;
                                            }
                                            response
                                        }
                                        Err(e) => {
                                            error!("Failed to connect to Kokoro API: {}", e);
                                            continue;
                                        }
                                    };

                                    if options.stream {
                                        let stream = response.bytes_stream();
                                        let audio_broadcaster = audio_tx.clone();

                                        tokio::spawn(async move {
                                            let mut stream = Box::pin(stream);

                                            while let Some(item) = stream.next().await {
                                                match item {
                                                    Ok(bytes) => {
                                                        if let Err(e) = audio_broadcaster.send(bytes.to_vec()) {
                                                            error!("Failed to broadcast audio chunk: {}", e);
                                                        }
                                                    }
                                                    Err(e) => {
                                                        error!("Error receiving audio stream: {}", e);
                                                        break;
                                                    }
                                                }
                                            }
                                            debug!("Finished streaming audio from Kokoro");
                                        });
                                    } else {
                                        match response.bytes().await {
                                            Ok(bytes) => {
                                                if let Err(e) = audio_tx.send(bytes.to_vec()) {
                                                    error!("Failed to send audio data: {}", e);
                                                } else {
                                                    debug!("Sent {} bytes of audio data", bytes.len());
                                                }
                                            }
                                            Err(e) => {
                                                error!("Failed to get audio bytes: {}", e);
                                            }
                                        }
                                    }
                                } else {
                                    error!("Kokoro configuration not found");
                                }
                            }
                        }
                    },
                    SpeechCommand::SetSTTProvider(provider) => {
                        let mut current_provider = stt_provider.write().await;
                        *current_provider = provider.clone();
                        info!("STT provider updated to: {:?}", provider);
                    },
                    SpeechCommand::StartTranscription(options) => {
                        let provider = stt_provider.read().await.clone();

                        match provider {
                            STTProvider::Whisper => {
                                info!("Starting Whisper transcription with options: {:?}", options);

                                let whisper_config = {
                                    let s = settings.read().await;
                                    s.whisper.clone()
                                };

                                if let Some(config) = whisper_config {
                                    let api_url = config.api_url.as_deref().unwrap_or("http://172.18.0.4:8000");
                                    info!("Whisper STT initialized with API URL: {}", api_url);

                                    let _ = transcription_tx.send("Whisper STT ready".to_string());
                                } else {
                                    error!("Whisper configuration not found");
                                    let _ = transcription_tx.send("Whisper STT configuration missing".to_string());
                                }
                            },
                            STTProvider::OpenAI => {
                                info!("Starting OpenAI transcription with options: {:?}", options);
                                // TODO: Implement OpenAI STT
                            }
                        }
                    },
                    SpeechCommand::StopTranscription => {
                        info!("Stopping transcription");
                        // TODO: Implement stop logic
                    },
                    SpeechCommand::ProcessAudioChunk(audio_data) => {
                        debug!("Processing audio chunk of size: {} bytes", audio_data.len());

                        let provider = stt_provider.read().await.clone();

                        match provider {
                            STTProvider::Whisper => {
                                let whisper_config = {
                                    let s = settings.read().await;
                                    s.whisper.clone()
                                };

                                if let Some(config) = whisper_config {
                                    let api_url_base = config.api_url.as_deref().unwrap_or("http://172.18.0.4:8000");
                                    let api_url = format!("{}/transcription/", api_url_base.trim_end_matches('/'));

                                    let form = reqwest::multipart::Form::new()
                                        .part("file", reqwest::multipart::Part::bytes(audio_data)
                                            .file_name("audio.wav")
                                            .mime_str("audio/wav").unwrap_or_else(|_| reqwest::multipart::Part::bytes(vec![]).mime_str("audio/wav").unwrap()));

                                    let mut form = form;
                                    if let Some(model) = &config.default_model {
                                        form = form.text("model", model.clone());
                                    }
                                    if let Some(language) = &config.default_language {
                                        form = form.text("language", language.clone());
                                    }
                                    if let Some(temperature) = config.temperature {
                                        form = form.text("temperature", temperature.to_string());
                                    }
                                    if let Some(vad_filter) = config.vad_filter {
                                        form = form.text("vad_filter", vad_filter.to_string());
                                    }
                                    if let Some(word_timestamps) = config.word_timestamps {
                                        form = form.text("word_timestamps", word_timestamps.to_string());
                                    }
                                    if let Some(initial_prompt) = &config.initial_prompt {
                                        form = form.text("initial_prompt", initial_prompt.clone());
                                    }

                                    let http_client_clone = Arc::clone(&http_client);
                                    let transcription_broadcaster = transcription_tx.clone();

                                    tokio::spawn(async move {
                                        match http_client_clone
                                            .post(&api_url)
                                            .multipart(form)
                                            .send()
                                            .await
                                        {
                                            Ok(response) => {
                                                if response.status().is_success() {
                                                    match response.json::<serde_json::Value>().await {
                                                        Ok(json) => {
                                                            if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
                                                                if !text.trim().is_empty() {
                                                                    debug!("Whisper transcription: {}", text);
                                                                    let _ = transcription_broadcaster.send(text.to_string());
                                                                }
                                                            } else {
                                                                error!("No text field in Whisper response: {:?}", json);
                                                            }
                                                        }
                                                        Err(e) => {
                                                            error!("Failed to parse Whisper response JSON: {}", e);
                                                        }
                                                    }
                                                } else {
                                                    let status = response.status();
                                                    let error_text = response.text().await.unwrap_or_default();
                                                    error!("Whisper API error {}: {}", status, error_text);
                                                }
                                            }
                                            Err(e) => {
                                                error!("Failed to connect to Whisper API: {}", e);
                                            }
                                        }
                                    });
                                } else {
                                    error!("Whisper configuration not found for audio processing");
                                }
                            },
                            STTProvider::OpenAI => {
                                debug!("OpenAI STT audio processing not implemented");
                                // TODO: Implement OpenAI STT processing
                            }
                        }
                    }
                }
            }
        });
    }

    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::Initialize;
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    pub async fn send_message(&self, message: String) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::SendMessage(message);
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    /// Converts text to speech using the configured TTS provider
    ///
    /// # Arguments
    /// * `text` - The text to be converted to speech
    /// * `options` - Speech generation options including voice, speed, and streaming preferences
    ///
    /// # Returns
    /// * `Ok(())` if the command was successfully queued for processing
    /// * `Err` if the command channel is closed or other error occurs
    ///
    /// # Behavior
    /// - Queues the TTS request for async processing by the service task
    /// - Audio output is broadcast to all subscribers via the audio channel
    /// - Supports both streaming and non-streaming audio generation
    /// - Uses Kokoro API by default with fallback error handling
    pub async fn text_to_speech(&self, text: String, options: SpeechOptions) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::TextToSpeech(text, options);
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    pub async fn close(&self) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::Close;
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    pub async fn set_tts_provider(&self, provider: TTSProvider) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::SetTTSProvider(provider);
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    /// Creates a new subscriber to the audio broadcast channel for receiving TTS audio data
    ///
    /// # Returns
    /// * `broadcast::Receiver<Vec<u8>>` - A receiver that will get all audio chunks from TTS operations
    ///
    /// # Usage
    /// Multiple WebSocket connections can subscribe to receive the same audio data simultaneously.
    /// Each subscriber gets its own independent receiver with a buffer to handle temporary disconnections.
    /// Audio data is broadcast as raw bytes (typically MP3 or WAV format from Kokoro TTS).
    pub fn subscribe_to_audio(&self) -> broadcast::Receiver<Vec<u8>> {
        self.audio_tx.subscribe()
    }

    // Current provider
    pub async fn get_tts_provider(&self) -> TTSProvider {
        self.tts_provider.read().await.clone()
    }

    pub async fn set_stt_provider(&self, provider: STTProvider) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::SetSTTProvider(provider);
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    pub async fn start_transcription(&self, options: TranscriptionOptions) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::StartTranscription(options);
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    pub async fn stop_transcription(&self) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::StopTranscription;
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    /// Processes audio data for speech-to-text transcription using the configured STT provider
    ///
    /// # Arguments
    /// * `audio_data` - Raw audio bytes in WAV format from client microphone input
    ///
    /// # Returns
    /// * `Ok(())` if the audio chunk was successfully queued for processing
    /// * `Err` if the command channel is closed or other error occurs
    ///
    /// # Behavior
    /// - Queues audio data for async STT processing by the service task
    /// - Sends audio to Whisper API at configured endpoint (default: http://172.18.0.4:8000)
    /// - Transcription results are broadcast to all subscribers via transcription channel
    /// - Supports configurable Whisper parameters (model, language, temperature, etc.)
    /// - Handles multipart form upload format required by Whisper-WebUI-Backend
    pub async fn process_audio_chunk(&self, audio_data: Vec<u8>) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::ProcessAudioChunk(audio_data);
        self.sender.lock().await.send(command).await.map_err(|e| Box::new(SpeechError::from(e)))?;
        Ok(())
    }

    /// Creates a new subscriber to the transcription broadcast channel for receiving STT results
    ///
    /// # Returns
    /// * `broadcast::Receiver<String>` - A receiver that will get all transcription text from STT operations
    ///
    /// # Usage
    /// Multiple WebSocket connections can subscribe to receive the same transcription results simultaneously.
    /// Each subscriber gets its own independent receiver with a buffer to handle temporary disconnections.
    /// Transcription results are broadcast as plain text strings from Whisper STT processing.
    pub fn subscribe_to_transcriptions(&self) -> broadcast::Receiver<String> {
        self.transcription_tx.subscribe()
    }
}
