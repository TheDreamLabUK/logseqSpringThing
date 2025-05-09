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
use crate::types::speech::{SpeechError, SpeechCommand, TTSProvider, SpeechOptions};
use reqwest::Client;


pub struct SpeechService {
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    settings: Arc<RwLock<AppFullSettings>>,
    tts_provider: Arc<RwLock<TTSProvider>>,
    // Audio broadcast channel for distributing TTS audio to all connected clients
    audio_tx: broadcast::Sender<Vec<u8>>,
    http_client: Arc<Client>,
}

impl SpeechService {
    pub fn new(settings: Arc<RwLock<AppFullSettings>>) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));

        // Create a broadcast channel for audio data with buffer size of 100
        let (audio_tx, _) = broadcast::channel(100);
        
        // Create HTTP client for Kokoro TTS API
        let http_client = Arc::new(Client::new());

        let service = SpeechService {
            sender,
            settings,
            tts_provider: Arc::new(RwLock::new(TTSProvider::Kokoro)), // Updated default to Kokoro
            audio_tx,
            http_client,
        };

        service.start(rx);
        service
    }

    fn start(&self, mut receiver: mpsc::Receiver<SpeechCommand>) {
        let settings: Arc<RwLock<AppFullSettings>> = Arc::clone(&self.settings);
        let _http_client = Arc::clone(&self.http_client);
        let tts_provider = Arc::clone(&self.tts_provider);
        let _audio_tx = self.audio_tx.clone();

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
                                                                        // Note: Audio data will be handled by socket-flow server
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
                        // Update the provider
                        let mut current_provider = tts_provider.write().await;
                        *current_provider = provider.clone();
                        info!("TTS provider updated to: {:?}", provider);
                    },
                    SpeechCommand::TextToSpeech(_text, _options) => {
                        // // Check which provider to use
                        // let provider = {
                        //     let p = tts_provider.read().await;
                        //     p.clone()
                        // };
                        //
                        // match provider {
                        //     TTSProvider::OpenAI => {
                        //         // Ignore OpenAI for now and just log
                        //         info!("TextToSpeech command with OpenAI provider not implemented");
                        //     },
                        //     TTSProvider::Kokoro => {
                        //         info!("Processing TextToSpeech command with Kokoro provider");
                        //         let kokoro_config = { // Read settings within scope
                        //             let s = settings.read().await;
                        //             s.kokoro.clone() // Clone the Option<KokoroSettings>
                        //         };
                        //
                        //         // Check if Kokoro is configured
                        //         if let Some(config) = kokoro_config {
                        //             // Safely get API URL or skip if missing
                        //             let api_url_base = match config.api_url.as_deref() {
                        //                 Some(url) if !url.is_empty() => url,
                        //                 _ => {
                        //                     error!("Kokoro API URL not configured or empty.");
                        //                     continue; // Skip this TTS request
                        //                 }
                        //             };
                        //             let api_url = format!("{}/v1/audio/speech", api_url_base.trim_end_matches('/'));
                        //             info!("Sending TTS request to Kokoro API: {}", api_url);
                        //
                        //             // Use defaults from config if available, otherwise hardcoded defaults
                        //             let response_format = config.default_format.as_deref().unwrap_or("mp3");
                        //
                        //             let request_body = json!({
                        //                 "model": "kokoro", // Assuming model is fixed
                        //                 "input": text,
                        //                 "voice": options.voice.clone(), // Voice comes from request options
                        //                 "response_format": response_format,
                        //                 "speed": options.speed, // Speed comes from request options
                        //                 "stream": options.stream // Stream comes from request options
                        //             });
                        //
                        //         let response = match http_client
                        //             .post(&api_url)
                        //             .header("Content-Type", "application/json")
                        //             .body(request_body.to_string())
                        //             .send()
                        //             .await
                        //         {
                        //             Ok(response) => {
                        //                 if !response.status().is_success() {
                        //                     let status = response.status();
                        //                     let error_text = response.text().await.unwrap_or_default();
                        //                     error!("Kokoro API error {}: {}", status, error_text);
                        //                     continue;
                        //                 }
                        //                 response
                        //             }
                        //             Err(e) => {
                        //                 error!("Failed to connect to Kokoro API: {}", e);
                        //                 continue;
                        //             }
                        //         };
                        //
                        //         // Handle the response (streaming or not)
                        //         if options.stream {
                        //             let stream = response.bytes_stream();
                        //             let audio_broadcaster = audio_tx.clone();
                        //
                        //             // Process the streaming response
                        //             tokio::spawn(async move {
                        //                 let mut stream = Box::pin(stream);
                        //
                        //                 while let Some(item) = stream.next().await {
                        //                     match item {
                        //                         Ok(bytes) => {
                        //                             // Send audio chunk to all connected clients
                        //                             if let Err(e) = audio_broadcaster.send(bytes.to_vec()) {
                        //                                 error!("Failed to broadcast audio chunk: {}", e);
                        //                             }
                        //                         }
                        //                         Err(e) => {
                        //                             error!("Error receiving audio stream: {}", e);
                        //                             break;
                        //                         }
                        //                     }
                        //                 }
                        //                 debug!("Finished streaming audio from Kokoro");
                        //             });
                        //         } else {
                        //             // Handle non-streaming response
                        //             match response.bytes().await {
                        //                 Ok(bytes) => {
                        //                     // Send the complete audio file in one chunk
                        //                     if let Err(e) = audio_tx.send(bytes.to_vec()) {
                        //                         error!("Failed to send audio data: {}", e);
                        //                     } else {
                        //                         debug!("Sent {} bytes of audio data", bytes.len());
                        //                     }
                        //                 }
                        //                 Err(e) => {
                        //                     error!("Failed to get audio bytes: {}", e);
                        //                 }
                        //             }
                        //         }
                        //     }
                        // }
                        info!("TextToSpeech arm commented out for debugging delimiter issue.");
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

    // Get a subscriber to the audio broadcast channel
    pub fn subscribe_to_audio(&self) -> broadcast::Receiver<Vec<u8>> {
        self.audio_tx.subscribe()
    }
    
    // Current provider
    pub async fn get_tts_provider(&self) -> TTSProvider {
        self.tts_provider.read().await.clone()
    }
}
