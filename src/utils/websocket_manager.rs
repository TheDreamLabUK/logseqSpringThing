use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use crate::AppState;
use log::{info, error, debug};
use std::sync::{Mutex, Arc};
use serde_json::{json, Value};
use futures::future::join_all;
use futures::StreamExt;
use tokio::task;
use std::collections::HashSet;

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
        let futures = sessions.iter().map(|session| {
            session.send(BroadcastAudio(audio.clone()))
        });
        
        join_all(futures).await;
        debug!("Broadcasted audio to {} sessions", sessions.len());
        Ok(())
    }
}

pub struct WebSocketSession {
    state: web::Data<AppState>,
}

impl WebSocketSession {
    fn new(state: web::Data<AppState>) -> Self {
        WebSocketSession { state }
    }

    fn send_json_response(&self, ctx: &mut ws::WebsocketContext<Self>, data: Value) {
        if let Ok(json_string) = serde_json::to_string(&data) {
            ctx.text(json_string.clone());
            debug!("Sent JSON response: {}", json_string);
        } else {
            error!("Failed to serialize JSON response");
        }
    }

    fn handle_chat_message(&self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling chat message: {:?}", msg);
        match msg["type"].as_str() {
            Some("ragflowQuery") => self.handle_ragflow_query(ctx, msg),
            Some("openaiQuery") => self.handle_openai_query(ctx, msg),
            _ => {
                error!("Unknown chat message type");
                self.send_json_response(ctx, json!({
                    "type": "error",
                    "message": "Unknown chat message type"
                }));
            }
        }
    }

    fn handle_ragflow_query(&self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling RAGflow query: {:?}", msg);
        let state = self.state.clone();
        let conversation_id = state.websocket_manager.conversation_id.lock().unwrap().clone();
        let addr = ctx.address();
        
        let fut = async move {
            let result = Self::process_ragflow_query(state, conversation_id, msg).await;
            addr.do_send(RAGFlowQueryResult(result));
        };

        ctx.spawn(actix::fut::wrap_future(fut));
    }

    fn handle_openai_query(&self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling OpenAI query: {:?}", msg);
        let state = self.state.clone();
        let addr = ctx.address();
        
        let fut = async move {
            if let Some(message) = msg["message"].as_str() {
                if let Err(e) = state.speech_service.send_message(message.to_string()).await {
                    error!("Failed to send message to SpeechService: {}", e);
                    addr.do_send(OpenAIQueryResult(Err(e.to_string())));
                } else {
                    addr.do_send(OpenAIQueryResult(Ok(())));
                }
            } else {
                addr.do_send(OpenAIQueryResult(Err("Invalid message format".to_string())));
            }
        };

        ctx.spawn(actix::fut::wrap_future(fut));
    }

    async fn process_ragflow_query(state: web::Data<AppState>, conversation_id: Option<String>, msg: Value) -> Result<Vec<u8>, String> {
        match conversation_id {
            Some(conv_id) => {
                let message = msg["message"].as_str().unwrap_or("").to_string();
                let quote = msg["quote"].as_bool().unwrap_or(false);
                let doc_ids = msg["docIds"].as_array().map(|arr| {
                    arr.iter().filter_map(|v| v.as_str()).map(String::from).collect::<Vec<String>>()
                });
                let stream = msg["stream"].as_bool().unwrap_or(false);

                match state.ragflow_service.send_message(conv_id, message, quote, doc_ids, stream).await {
                    Ok(mut audio_stream) => {
                        let mut audio_data = Vec::new();
                        while let Some(chunk_result) = audio_stream.next().await {
                            match chunk_result {
                                Ok(chunk) => audio_data.extend_from_slice(&chunk),
                                Err(e) => return Err(format!("Error in audio stream: {}", e)),
                            }
                        }
                        Ok(audio_data)
                    },
                    Err(e) => Err(format!("Failed to send message: {}", e)),
                }
            },
            None => Err("Chat not initialized. Please try again later.".to_string()),
        }
    }

    fn handle_graph_update(&self, ctx: &mut ws::WebsocketContext<Self>) {
        let state = self.state.clone();
        ctx.spawn(async move {
            let graph_data = state.graph_data.read().await;
            let nodes_with_file_size: Vec<_> = graph_data.nodes.iter().map(|node| {
                let mut node_with_metadata = node.clone();
                if let Some(metadata) = graph_data.metadata.get(&node.id) {
                    node_with_metadata.metadata.insert("file_size".to_string(), metadata.file_size.to_string());
                }
                node_with_metadata
            }).collect();
            json!({
                "type": "graphUpdate",
                "graphData": {
                    "nodes": nodes_with_file_size,
                    "edges": graph_data.edges,
                }
            })
        }.into_actor(self).map(|response, act, ctx| {
            act.send_json_response(ctx, response);
        }));
    }
}

impl Actor for WebSocketSession {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().push(addr.clone());
        info!("WebSocket session started. Total sessions: {}", self.state.websocket_manager.sessions.lock().unwrap().len());
        
        // Register with SpeechService
        let speech_service = self.state.speech_service.clone();
        task::spawn(async move {
            speech_service.register_session(addr).await;
        });
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().retain(|session| session != &addr);
        info!("WebSocket session stopped. Total sessions: {}", self.state.websocket_manager.sessions.lock().unwrap().len());
        
        // Unregister from SpeechService
        let speech_service = self.state.speech_service.clone();
        task::spawn(async move {
            speech_service.unregister_session(&addr).await;
        });
    }
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastMessage(pub String);

#[derive(Message)]
#[rtype(result = "()")]
struct RAGFlowQueryResult(Result<Vec<u8>, String>);

#[derive(Message)]
#[rtype(result = "()")]
struct OpenAIQueryResult(Result<(), String>);

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastAudio(pub Vec<u8>);

impl Handler<BroadcastMessage> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        ctx.text(msg.0);
        debug!("Broadcasted message to client");
    }
}

impl Handler<RAGFlowQueryResult> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: RAGFlowQueryResult, ctx: &mut Self::Context) {
        match msg.0 {
            Ok(audio_data) => {
                ctx.binary(audio_data);
            },
            Err(e) => {
                error!("Error in RAGFlow query: {}", e);
                self.send_json_response(ctx, json!({
                    "type": "error",
                    "message": e
                }));
            }
        }
    }
}

impl Handler<OpenAIQueryResult> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: OpenAIQueryResult, ctx: &mut Self::Context) {
        match msg.0 {
            Ok(()) => {
                debug!("OpenAI query processed successfully");
            },
            Err(e) => {
                error!("Error in OpenAI query: {}", e);
                self.send_json_response(ctx, json!({
                    "type": "error",
                    "message": e
                }));
            }
        }
    }
}

impl Handler<BroadcastAudio> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: BroadcastAudio, ctx: &mut Self::Context) {
        ctx.binary(msg.0);
        debug!("Broadcasted audio to client");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketSession {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                ctx.pong(&msg);
            },
            Ok(ws::Message::Pong(_)) => {
                // Optionally handle pong responses.
            },
            Ok(ws::Message::Text(text)) => {
                info!("Received message from client: {}", text);
                match serde_json::from_str::<Value>(&text) {
                    Ok(json_data) => {
                        if let Some(msg_type) = json_data["type"].as_str() {
                            match msg_type {
                                "getInitialData" => {
                                    debug!("Handling getInitialData request");
                                    self.handle_graph_update(ctx);
                                },
                                "ragflowQuery" | "openaiQuery" => {
                                    debug!("Handling chat message: {}", msg_type);
                                    self.handle_chat_message(ctx, json_data);
                                },
                                _ => {
                                    error!("Received unknown message type: {}", msg_type);
                                    let error_response = json!({
                                        "type": "error",
                                        "message": format!("Unknown message type: {}", msg_type),
                                    });
                                    self.send_json_response(ctx, error_response);
                                }
                            }
                        } else {
                            error!("Received message without a type field");
                            let error_response = json!({
                                "type": "error",
                                "message": "Message type not specified",
                            });
                            self.send_json_response(ctx, error_response);
                        }
                    },
                    Err(e) => {
                        error!("Failed to parse incoming message as JSON: {}", e);
                        let error_response = json!({
                            "type": "error",
                            "message": "Invalid JSON format",
                        });
                        self.send_json_response(ctx, error_response);
                    }
                }
            },
            Ok(ws::Message::Binary(bin)) => {
                let bin_clone = bin.clone();
                ctx.binary(bin);
                debug!("Received binary message of {} bytes", bin_clone.len());
            },
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket closed: {:?}", reason);
                ctx.stop();
            },
            _ => (),
        }
    }
}