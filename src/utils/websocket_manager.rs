use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use crate::AppState;
use log::{info, error, debug};
use std::sync::{Mutex, Arc};
use serde_json::{json, Value};
use futures::future::join_all;
use futures::StreamExt;
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct WebSocketManager {
    pub sessions: Mutex<Vec<Addr<WebSocketSession>>>,
    pub conversation_id: Arc<Mutex<Option<String>>>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        WebSocketManager {
            sessions: Mutex::new(Vec::new()),
            conversation_id: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn initialize(&self, ragflow_service: &crate::services::ragflow_service::RAGFlowService) -> Result<(), Box<dyn std::error::Error>> {
        let conversation_id = ragflow_service.create_conversation("default_user".to_string()).await?;
        *self.conversation_id.lock().unwrap() = Some(conversation_id.clone());
        info!("Initialized conversation with ID: {}", conversation_id);
        Ok(())
    }

    pub async fn handle_websocket(req: HttpRequest, stream: web::Payload, state: web::Data<AppState>) -> Result<HttpResponse, Error> {
        info!("New WebSocket connection request");
        let session = WebSocketSession::new(state.clone());
        let resp = ws::start(session, &req, stream)?;
        info!("WebSocket connection established");
        Ok(resp)
    }

    pub async fn broadcast_message(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        let sessions = self.sessions.lock().unwrap().clone();
        let futures = sessions.iter().map(|session| {
            session.send(BroadcastMessage(message.to_string()))
        });
        
        join_all(futures).await;
        debug!("Broadcasted message to {} sessions", sessions.len());
        Ok(())
    }

    pub async fn broadcast_audio(&self, audio: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let sessions = self.sessions.lock().unwrap().clone();
        let audio_base64 = general_purpose::STANDARD.encode(&audio);
        let message = json!({
            "type": "audio",
            "data": audio_base64
        });
        
        self.broadcast_message(&message.to_string()).await?;
        debug!("Broadcasted audio to {} sessions", sessions.len());
        Ok(())
    }

    pub async fn broadcast_ragflow_response(&self, response: &str) -> Result<(), Box<dyn std::error::Error>> {
        let message = json!({
            "type": "ragflowResponse",
            "response": response
        });
        self.broadcast_message(&message.to_string()).await?;
        Ok(())
    }

    pub async fn broadcast_perplexity_response(&self, response: &str) -> Result<(), Box<dyn std::error::Error>> {
        let message = json!({
            "type": "perplexityResponse",
            "response": response
        });
        self.broadcast_message(&message.to_string()).await?;
        Ok(())
    }

    pub async fn broadcast_tts_method(&self, use_openai: bool) -> Result<(), Box<dyn std::error::Error>> {
        let message = json!({
            "type": "ttsMethodSet",
            "useOpenAI": use_openai
        });
        self.broadcast_message(&message.to_string()).await?;
        Ok(())
    }
}

pub struct WebSocketSession {
    state: web::Data<AppState>,
    hb: Instant,
}

impl WebSocketSession {
    fn new(state: web::Data<AppState>) -> Self {
        WebSocketSession {
            state,
            hb: Instant::now(),
        }
    }

    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(HEARTBEAT_INTERVAL, |act, ctx| {
            if Instant::now().duration_since(act.hb) > CLIENT_TIMEOUT {
                error!("Websocket Client heartbeat failed, disconnecting!");
                ctx.stop();
                return;
            }
            ctx.ping(b"");
        });
    }

    fn handle_message(&mut self, text: &str, ctx: &mut ws::WebsocketContext<Self>) {
        if let Ok(msg) = serde_json::from_str::<Value>(text) {
            match msg["type"].as_str() {
                Some("ttsRequest") => {
                    if let (Some(text), Some(use_openai)) = (msg["text"].as_str(), msg["useOpenAI"].as_bool()) {
                        let speech_service = self.state.speech_service.clone();
                        let message = text.to_string();
                        
                        actix::spawn(async move {
                            if let Err(e) = speech_service.set_tts_provider(use_openai).await {
                                error!("Failed to set TTS provider: {}", e);
                            }
                            if let Err(e) = speech_service.send_message(message).await {
                                error!("Failed to send message: {}", e);
                            }
                        });
                    }
                },
                Some("ragflowQuery") => {
                    if let Some(message) = msg["message"].as_str() {
                        let ragflow_service = self.state.ragflow_service.clone();
                        let conversation_id = self.state.websocket_manager.conversation_id.lock().unwrap().clone();
                        let websocket_manager = self.state.websocket_manager.clone();
                        
                        actix::spawn(async move {
                            if let Some(conv_id) = conversation_id {
                                match ragflow_service.send_message(conv_id, message.to_string(), false, None, false).await {
                                    Ok(mut stream) => {
                                        while let Some(result) = stream.next().await {
                                            match result {
                                                Ok(response) => {
                                                    if let Err(e) = websocket_manager.broadcast_ragflow_response(&response).await {
                                                        error!("Failed to broadcast RAGFlow response: {}", e);
                                                    }
                                                },
                                                Err(e) => error!("Error in RAGFlow stream: {}", e),
                                            }
                                        }
                                    },
                                    Err(e) => error!("Failed to send message to RAGFlow: {}", e),
                                }
                            }
                        });
                    }
                },
                Some("perplexityQuery") => {
                    if let Some(message) = msg["message"].as_str() {
                        let perplexity_service = self.state.perplexity_service.clone();
                        let websocket_manager = self.state.websocket_manager.clone();
                        
                        actix::spawn(async move {
                            match perplexity_service.process_markdown(message).await {
                                Ok(response) => {
                                    if let Err(e) = websocket_manager.broadcast_perplexity_response(&response).await {
                                        error!("Failed to broadcast Perplexity response: {}", e);
                                    }
                                },
                                Err(e) => error!("Failed to process Perplexity query: {}", e),
                            }
                        });
                    }
                },
                _ => debug!("Unhandled message type: {:?}", msg["type"]),
            }
        }
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.hb(ctx);
        
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().push(addr);
        info!("WebSocket session started");
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("WebSocket session stopped");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
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
                self.handle_message(&text, ctx);
            }
            Ok(ws::Message::Binary(bin)) => {
                debug!("Received binary message of {} bytes", bin.len());
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastMessage(pub String);

impl Handler<BroadcastMessage> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum ClientMessage {
    #[serde(rename = "ttsRequest")]
    TTSRequest {
        text: String,
        use_openai: bool,
    },
    #[serde(rename = "ragflowQuery")]
    RagFlowQuery {
        message: String,
    },
    #[serde(rename = "perplexityQuery")]
    PerplexityQuery {
        message: String,
    },
}
