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
use actix::{StreamHandler, AsyncContext};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

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

impl actix::Actor for SpeechWs {
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
                // Handle text messages
                ctx.text(text);
            }
            Ok(ws::Message::Binary(bin)) => {
                debug!("Received binary message of {} bytes", bin.len());
                // Handle binary messages
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

pub struct SpeechService {
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    websocket_manager: Arc<WebSocketManager>,
    settings: Arc<RwLock<Settings>>,
}

#[derive(Debug)]
enum SpeechCommand {
    Initialize,
    SendMessage(String),
    Close,
}

impl SpeechService {
    pub fn new(websocket_manager: Arc<WebSocketManager>, settings: Arc<RwLock<Settings>>) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));

        let service = SpeechService {
            sender,
            websocket_manager,
            settings,
        };

        service.start(rx);
        service
    }

    fn start(&self, mut receiver: mpsc::Receiver<SpeechCommand>) {
        let websocket_manager = Arc::clone(&self.websocket_manager);
        let settings = Arc::clone(&self.settings);

        task::spawn(async move {
            let mut ws_stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>> = None;

            while let Some(command) = receiver.recv().await {
                match command {
                    SpeechCommand::Initialize => {
                        let settings_guard = settings.read().await;
                        let url = Url::parse("wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01")
                            .expect("Failed to parse URL");
                        
                        let request = Request::builder()
                            .uri(url.as_str())
                            .header("Authorization", format!("Bearer {}", settings_guard.openai.openai_api_key))
                            .header("OpenAI-Beta", "realtime=v1")
                            .header("User-Agent", "WebXR Graph")
                            .header("Origin", "https://api.openai.com")
                            .header("Sec-WebSocket-Version", "13")
                            .header("Sec-WebSocket-Key", tungstenite::handshake::client::generate_key())
                            .header("Connection", "Upgrade")
                            .header("Upgrade", "websocket")
                            .body(())
                            .expect("Failed to build request");

                        drop(settings_guard); // Release the lock before the async operation

                        match connect_async(request).await {
                            Ok((stream, _)) => {
                                info!("Connected to OpenAI Realtime API");
                                ws_stream = Some(stream);
                                
                                if let Some(stream) = &mut ws_stream {
                                    let init_event = json!({
                                        "type": "response.create",
                                        "response": {
                                            "modalities": ["text"],
                                            "instructions": "Please assist the user.",
                                        }
                                    });
                                    if let Err(e) = stream.send(Message::Text(init_event.to_string())).await {
                                        error!("Failed to send initial configuration: {}", e);
                                    }
                                }
                            },
                            Err(e) => error!("Failed to connect to OpenAI Realtime API: {}", e),
                        }
                    },
                    SpeechCommand::SendMessage(msg) => {
                        if let Some(stream) = &mut ws_stream {
                            let (mut write, mut read) = stream.split();

                            let event = json!({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": msg
                                        }
                                    ]
                                }
                            });

                            if let Err(e) = write.send(Message::Text(event.to_string())).await {
                                error!("Failed to send message to OpenAI: {}", e);
                            } else {
                                info!("Sent message to OpenAI: {}", msg);
                            }

                            let response_event = json!({
                                "type": "response.create"
                            });
                            if let Err(e) = write.send(Message::Text(response_event.to_string())).await {
                                error!("Failed to trigger response: {}", e);
                            }

                            while let Some(message) = read.next().await {
                                match message {
                                    Ok(Message::Text(text)) => {
                                        debug!("Received message from OpenAI: {}", text);
                                        if let Ok(json_msg) = serde_json::from_str::<serde_json::Value>(&text) {
                                            match json_msg["type"].as_str() {
                                                Some("response.text.delta") => {
                                                    if let Some(content) = json_msg["delta"]["text"].as_str() {
                                                        debug!("Received text delta: {}", content);
                                                    }
                                                },
                                                Some("response.text.done") => {
                                                    debug!("Text response complete");
                                                },
                                                Some("response.done") => {
                                                    debug!("Full response complete");
                                                    break;
                                                },
                                                _ => {}
                                            }
                                        }
                                    },
                                    Ok(Message::Close(_)) => {
                                        info!("OpenAI WebSocket connection closed by server");
                                        break;
                                    },
                                    Err(e) => {
                                        error!("OpenAI WebSocket error: {}", e);
                                        break;
                                    },
                                    _ => {},
                                }
                            }
                        } else {
                            error!("WebSocket connection not initialized");
                        }
                    },
                    SpeechCommand::Close => {
                        if let Some(stream) = &mut ws_stream {
                            if let Err(e) = stream.send(Message::Close(None)).await {
                                error!("Failed to close WebSocket connection: {}", e);
                            }
                        }
                        break;
                    },
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
}
