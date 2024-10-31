use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream};
use tungstenite::protocol::Message;
use tungstenite::http::Request;
use serde_json::json;
use std::sync::Arc;
use tokio::task;
use crate::config::Settings;
use log::{info, error, debug};
use futures::{SinkExt, StreamExt};
use std::error::Error;
use crate::utils::websocket_manager::WebSocketManager;
use tokio::net::TcpStream;
use url::Url;
use actix_web::{web, Error as ActixError, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use std::time::{Duration, Instant};
use actix::{StreamHandler, AsyncContext, Actor};
use std::process::Command;
use tokio::io::AsyncWriteExt;
use base64::{Engine as _, engine::general_purpose};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub enum TTSProvider {
    OpenAI,
    Sonata,
}

#[derive(Debug)]
enum SpeechCommand {
    Initialize,
    SendMessage(String),
    Close,
    SetTTSProvider(TTSProvider),
}

pub struct SpeechService {
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    websocket_manager: Arc<WebSocketManager>,
    settings: Arc<RwLock<Settings>>,
    tts_provider: Arc<RwLock<TTSProvider>>,
}

impl SpeechService {
    pub fn new(websocket_manager: Arc<WebSocketManager>, settings: Arc<RwLock<Settings>>) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));

        let service = SpeechService {
            sender,
            websocket_manager,
            settings,
            tts_provider: Arc::new(RwLock::new(TTSProvider::Sonata)),
        };

        service.start(rx);
        service
    }

    fn start(&self, mut receiver: mpsc::Receiver<SpeechCommand>) {
        let websocket_manager = Arc::clone(&self.websocket_manager);
        let settings = Arc::clone(&self.settings);
        let tts_provider = Arc::clone(&self.tts_provider);

        task::spawn(async move {
            let mut ws_stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>> = None;

            while let Some(command) = receiver.recv().await {
                match command {
                    SpeechCommand::Initialize => {
                        if let TTSProvider::OpenAI = *tts_provider.read().await {
                            let settings_guard = settings.read().await;
                            let url = Url::parse("wss://api.openai.com/v1/audio/speech")
                                .expect("Failed to parse URL");
                            
                            let request = Request::builder()
                                .uri(url.as_str())
                                .header("Authorization", format!("Bearer {}", settings_guard.openai.openai_api_key))
                                .header("Content-Type", "application/json")
                                .header("User-Agent", "WebXR Graph")
                                .header("Origin", "https://api.openai.com")
                                .header("Sec-WebSocket-Version", "13")
                                .header("Sec-WebSocket-Key", tungstenite::handshake::client::generate_key())
                                .header("Connection", "Upgrade")
                                .header("Upgrade", "websocket")
                                .body(())
                                .expect("Failed to build request");

                            drop(settings_guard);

                            match connect_async(request).await {
                                Ok((stream, _)) => {
                                    info!("Connected to OpenAI Audio API");
                                    ws_stream = Some(stream);
                                },
                                Err(e) => error!("Failed to connect to OpenAI Audio API: {}", e),
                            }
                        }
                    },
                    SpeechCommand::SendMessage(msg) => {
                        match *tts_provider.read().await {
                            TTSProvider::OpenAI => {
                                if let Some(stream) = &mut ws_stream {
                                    let settings_guard = settings.read().await;
                                    let request = json!({
                                        "model": "tts-1",
                                        "input": msg,
                                        "voice": "alloy",
                                        "response_format": "mp3"
                                    });

                                    if let Err(e) = stream.send(Message::Text(request.to_string())).await {
                                        error!("Failed to send message to OpenAI: {}", e);
                                    } else {
                                        while let Some(message) = stream.next().await {
                                            match message {
                                                Ok(Message::Binary(audio_data)) => {
                                                    if let Err(e) = websocket_manager.broadcast_audio(audio_data).await {
                                                        error!("Failed to broadcast audio: {}", e);
                                                    }
                                                    break;
                                                },
                                                Ok(Message::Close(_)) => break,
                                                Err(e) => {
                                                    error!("Error receiving audio from OpenAI: {}", e);
                                                    break;
                                                },
                                                _ => {}
                                            }
                                        }
                                    }
                                } else {
                                    error!("OpenAI WebSocket not initialized");
                                }
                            },
                            TTSProvider::Sonata => {
                                let mut child = Command::new("python3")
                                    .arg("src/generate_audio.py")
                                    .stdin(std::process::Stdio::piped())
                                    .stdout(std::process::Stdio::piped())
                                    .spawn()
                                    .expect("Failed to spawn Python process");

                                if let Some(mut stdin) = child.stdin.take() {
                                    tokio::io::AsyncWriteExt::write_all(&mut stdin, msg.as_bytes()).await
                                        .expect("Failed to write to stdin");
                                }

                                let output = child.wait_with_output()
                                    .expect("Failed to wait on child process");

                                if output.status.success() {
                                    if let Err(e) = websocket_manager.broadcast_audio(output.stdout).await {
                                        error!("Failed to broadcast Sonata audio: {}", e);
                                    }
                                } else {
                                    error!("Sonata TTS failed: {}", String::from_utf8_lossy(&output.stderr));
                                }
                            }
                        }
                    },
                    SpeechCommand::Close => {
                        if let Some(stream) = &mut ws_stream {
                            if let Err(e) = stream.close(None).await {
                                error!("Failed to close WebSocket connection: {}", e);
                            }
                        }
                        break;
                    },
                    SpeechCommand::SetTTSProvider(provider) => {
                        *tts_provider.write().await = provider;
                        info!("TTS provider set to: {:?}", provider);
                    }
                }
            }
        });
    }

    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::Initialize;
        self.sender.lock().await.send(command).await?;
        Ok(())
    }

    pub async fn send_message(&self, message: String) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::SendMessage(message);
        self.sender.lock().await.send(command).await?;
        Ok(())
    }

    pub async fn close(&self) -> Result<(), Box<dyn Error>> {
        let command = SpeechCommand::Close;
        self.sender.lock().await.send(command).await?;
        Ok(())
    }

    pub async fn set_tts_provider(&self, use_openai: bool) -> Result<(), Box<dyn Error>> {
        let provider = if use_openai {
            TTSProvider::OpenAI
        } else {
            TTSProvider::Sonata
        };
        let command = SpeechCommand::SetTTSProvider(provider);
        self.sender.lock().await.send(command).await?;
        Ok(())
    }
}

pub struct SpeechWs {
    hb: Instant,
    websocket_manager: Arc<WebSocketManager>,
    settings: Arc<RwLock<Settings>>,
}

impl SpeechWs {
    pub fn new(websocket_manager: Arc<WebSocketManager>, settings: Arc<RwLock<Settings>>) -> Self {
        Self {
            hb: Instant::now(),
            websocket_manager,
            settings,
        }
    }

    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_later(Duration::from_secs(0), |act, ctx| {
            act.check_heartbeat(ctx);
            ctx.run_interval(HEARTBEAT_INTERVAL, |act, ctx| {
                act.check_heartbeat(ctx);
            });
        });
    }

    fn check_heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        if Instant::now().duration_since(self.hb) > CLIENT_TIMEOUT {
            info!("Websocket Client heartbeat failed, disconnecting!");
            ctx.close(None);
            return;
        }
        ctx.ping(b"");
    }
}

impl Actor for SpeechWs {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.hb(ctx);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SpeechWs {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.hb = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                debug!("Received text message: {}", text);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let (Some(message), Some(use_openai)) = (json["message"].as_str(), json["useOpenAI"].as_bool()) {
                        let speech_service = SpeechService::new(
                            Arc::clone(&self.websocket_manager),
                            Arc::clone(&self.settings)
                        );
                        let message = message.to_string();
                        actix::spawn(async move {
                            if let Err(e) = speech_service.set_tts_provider(use_openai).await {
                                error!("Failed to set TTS provider: {}", e);
                            }
                            if let Err(e) = speech_service.send_message(message).await {
                                error!("Failed to send message: {}", e);
                            }
                        });
                    }
                }
            }
            Ok(ws::Message::Binary(bin)) => {
                debug!("Received binary message of {} bytes", bin.len());
                ctx.binary(bin);
            }
            Ok(ws::Message::Close(reason)) => {
                info!("Closing websocket connection: {:?}", reason);
                ctx.close(reason);
                return;
            }
            _ => (),
        }
    }
}

pub async fn start_websocket(
    req: HttpRequest,
    stream: web::Payload,
    websocket_manager: web::Data<Arc<WebSocketManager>>,
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> Result<HttpResponse, ActixError> {
    let ws = SpeechWs::new(Arc::clone(&websocket_manager), Arc::clone(&settings));
    ws::start(ws, &req, stream)
}
