use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::prelude::*;
use crate::AppState;
use log::{info, error, debug};
use std::sync::Mutex;
use serde_json::{json, Value};
use futures::future::join_all;
use crate::handlers::ragflow_handler::{MessageRequest, InitChatRequest};
use actix_web::body::MessageBody;

pub struct WebSocketManager {
    pub sessions: Mutex<Vec<Addr<WebSocketSession>>>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        WebSocketManager {
            sessions: Mutex::new(Vec::new()),
        }
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
}

pub struct WebSocketSession {
    state: web::Data<AppState>,
    active_conversation_id: Option<String>,
}

impl WebSocketSession {
    fn new(state: web::Data<AppState>) -> Self {
        WebSocketSession { 
            state,
            active_conversation_id: None,
        }
    }

    fn send_json_response(&self, ctx: &mut ws::WebsocketContext<Self>, data: Value) {
        if let Ok(json_string) = serde_json::to_string(&data) {
            ctx.text(json_string.clone());
            debug!("Sent JSON response: {}", json_string);
        } else {
            error!("Failed to serialize JSON response");
        }
    }

    fn handle_chat_message(&mut self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling chat message: {:?}", msg);
        match msg["type"].as_str() {
            Some("initChat") => self.handle_init_chat(ctx, msg),
            Some("ragflowQuery") => self.handle_ragflow_query(ctx, msg),
            Some("chatHistory") => self.handle_chat_history(ctx, msg),
            _ => {
                error!("Unknown chat message type");
                self.send_json_response(ctx, json!({
                    "type": "error",
                    "message": "Unknown chat message type"
                }));
            }
        }
    }

    fn handle_init_chat(&mut self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling init chat request: {:?}", msg);
        let user_id = msg["userId"].as_str().unwrap_or("default_user").to_string();
        let state = self.state.clone();
        ctx.spawn(async move {
            let init_request = InitChatRequest { user_id };
            let result = crate::handlers::ragflow_handler::init_chat(state, web::Json(init_request)).await;
            match result.into_body().try_into_bytes() {
                Ok(bytes) => {
                    if let Ok(response) = serde_json::from_slice::<Value>(&bytes) {
                        let conversation_id = response["data"]["id"].as_str().unwrap_or("").to_string();
                        info!("Chat initialized with conversation ID: {}", conversation_id);
                        json!({
                            "type": "chatInitResponse",
                            "success": true,
                            "conversationId": conversation_id
                        })
                    } else {
                        error!("Failed to parse init chat response");
                        json!({
                            "type": "chatInitResponse",
                            "success": false,
                            "conversationId": null,
                            "message": "Failed to parse response"
                        })
                    }
                },
                Err(e) => {
                    error!("Failed to initialize chat: {:?}", e);
                    json!({
                        "type": "chatInitResponse",
                        "success": false,
                        "conversationId": null,
                        "message": "Failed to initialize chat"
                    })
                },
            }
        }.into_actor(self).map(|response, act, ctx| {
            if let Some(conversation_id) = response["conversationId"].as_str() {
                act.active_conversation_id = Some(conversation_id.to_string());
            }
            act.send_json_response(ctx, response);
        }));
    }

    fn handle_ragflow_query(&self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling RAGflow query: {:?}", msg);
        let state = self.state.clone();
        let conversation_id = self.active_conversation_id.clone();
        
        ctx.spawn(async move {
            match conversation_id {
                Some(conv_id) => {
                    let message = msg["message"].as_str().unwrap_or("").to_string();
                    let quote = msg["quote"].as_bool().unwrap_or(false);
                    let doc_ids = msg["docIds"].as_array().map(|arr| {
                        arr.iter().filter_map(|v| v.as_str()).map(String::from).collect::<Vec<String>>()
                    });
                    let stream = msg["stream"].as_bool().unwrap_or(false);

                    let msg_request = MessageRequest { 
                        conversation_id: conv_id,
                        messages: vec![crate::services::ragflow_service::Message {
                            role: "user".to_string(),
                            content: message,
                        }],
                        quote: Some(quote),
                        doc_ids,
                        stream: Some(stream),
                    };
                    let result = crate::handlers::ragflow_handler::send_message(state, web::Json(msg_request)).await;
                    match result.into_body().try_into_bytes() {
                        Ok(bytes) => {
                            if let Ok(response) = serde_json::from_slice::<Value>(&bytes) {
                                info!("RAGflow query successful: {:?}", response);
                                json!({
                                    "type": "ragflowResponse",
                                    "data": response
                                })
                            } else {
                                error!("Failed to parse RAGflow response");
                                json!({
                                    "type": "error",
                                    "message": "Failed to parse response"
                                })
                            }
                        },
                        Err(e) => {
                            error!("Failed to send message: {:?}", e);
                            json!({
                                "type": "error",
                                "message": "Failed to send message"
                            })
                        },
                    }
                },
                None => {
                    error!("Chat not initialized");
                    json!({
                        "type": "error",
                        "message": "Chat not initialized. Please initialize chat first."
                    })
                },
            }
        }.into_actor(self).map(|response, act, ctx| {
            act.send_json_response(ctx, response);
        }));
    }

    fn handle_chat_history(&self, ctx: &mut ws::WebsocketContext<Self>, msg: Value) {
        info!("Handling chat history request: {:?}", msg);
        let state = self.state.clone();
        let conversation_id = msg["conversationId"].as_str().unwrap_or("").to_string();
        ctx.spawn(async move {
            let result = crate::handlers::ragflow_handler::get_chat_history(state, web::Path::from(conversation_id)).await;
            match result.into_body().try_into_bytes() {
                Ok(bytes) => {
                    if let Ok(history) = serde_json::from_slice::<Value>(&bytes) {
                        info!("Chat history retrieved successfully");
                        json!({
                            "type": "chatHistoryResponse",
                            "data": history
                        })
                    } else {
                        error!("Failed to parse chat history");
                        json!({
                            "type": "error",
                            "message": "Failed to parse chat history"
                        })
                    }
                },
                Err(e) => {
                    error!("Failed to fetch chat history: {:?}", e);
                    json!({
                        "type": "error",
                        "message": "Failed to fetch chat history"
                    })
                },
            }
        }.into_actor(self).map(|response, act, ctx| {
            act.send_json_response(ctx, response);
        }));
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
        self.state.websocket_manager.sessions.lock().unwrap().push(addr);
        info!("WebSocket session started. Total sessions: {}", self.state.websocket_manager.sessions.lock().unwrap().len());
    }

    fn stopped(&mut self, ctx: &mut Self::Context) {
        let addr = ctx.address();
        self.state.websocket_manager.sessions.lock().unwrap().retain(|session| session != &addr);
        info!("WebSocket session stopped. Total sessions: {}", self.state.websocket_manager.sessions.lock().unwrap().len());
    }
}

#[derive(Message)]
#[rtype(result = "()")]
struct BroadcastMessage(String);

impl Handler<BroadcastMessage> for WebSocketSession {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        ctx.text(msg.0);
        debug!("Broadcasted message to client");
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
                                "initChat" | "ragflowQuery" | "chatHistory" => {
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
