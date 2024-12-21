use actix::{Actor, StreamHandler, ActorContext, AsyncContext, Handler, Message as ActixMessage, ActorFutureExt};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use log::{error, warn, info};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::app_state::AppState;
use crate::utils::socket_flow_messages::{BinaryNodeData, Message};
use crate::utils::socket_flow_constants::{
    HEARTBEAT_INTERVAL as HEARTBEAT_INTERVAL_SECS,
    MAX_CLIENT_TIMEOUT as MAX_CLIENT_TIMEOUT_SECS,
};

// Convert seconds to Duration
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(HEARTBEAT_INTERVAL_SECS);
const MAX_CLIENT_TIMEOUT: Duration = Duration::from_secs(MAX_CLIENT_TIMEOUT_SECS);

#[derive(Debug)]
enum WebSocketResponse {
    Text(String),
    Binary(Vec<u8>),
    None,
}

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct SendMessage(String);

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct UpdateNodeOrder(Vec<String>);

impl Handler<UpdateNodeOrder> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: UpdateNodeOrder, _: &mut Self::Context) {
        self.node_order = msg.0;
    }
}

#[derive(Clone)]
pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    last_heartbeat: Instant,
    node_order: Vec<String>, // Store node IDs in order for binary updates
    settings: Arc<RwLock<crate::config::Settings>>,
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[WS] WebSocket connection established");
        if let Ok(settings) = self.settings.try_read() {
            if settings.server_debug.enabled {
                info!("[WS] Debug mode enabled");
                info!("[WS] WebSocket settings: {:?}", settings.websocket);
            }
        }
        self.app_state.increment_connections();
        let current = self.app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed);
        info!("[WS] Active connections: {}", current);
        self.heartbeat(ctx);
        self.start_position_updates(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("[WS] WebSocket connection closed");
        self.app_state.decrement_connections();
        let current = self.app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed);
        info!("[WS] Remaining active connections: {}", current);
    }
}

impl Handler<SendMessage> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SendMessage, ctx: &mut Self::Context) {
        if let Ok(settings) = self.settings.try_read() {
            if settings.server_debug.enabled {
                info!("[WS] Sending message: {}", msg.0);
            }
        }
        ctx.text(msg.0);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                if let Ok(settings) = self.settings.try_read() {
                    if settings.server_debug.enable_websocket_debug {
                        info!("[WS] Received ping message");
                    }
                }
                self.last_heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                if let Ok(settings) = self.settings.try_read() {
                    if settings.server_debug.enable_websocket_debug {
                        info!("[WS] Received pong message");
                    }
                }
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                if let Ok(settings) = self.settings.try_read() {
                    if settings.server_debug.enabled {
                        info!("[WS] Received text message: {}", text);
                    }
                }
                match serde_json::from_str::<Message>(&text) {
                    Ok(message) => {
                        let this = self.clone();
                        let settings = self.settings.clone();
                        let fut = async move {
                            this.handle_message(message).await
                        };
                        ctx.spawn(actix::fut::wrap_future(fut).map(move |response, _, ctx: &mut ws::WebsocketContext<SocketFlowServer>| {
                            match response {
                                WebSocketResponse::Text(text) => {
                                    if let Ok(settings) = settings.try_read() {
                                        if settings.server_debug.enabled {
                                            info!("[WS] Sending text response: {}", text);
                                        }
                                    }
                                    ctx.text(text);
                                }
                                WebSocketResponse::Binary(data) => {
                                    if let Ok(settings) = settings.try_read() {
                                        if settings.server_debug.enabled {
                                            info!("[WS] Sending binary response: {} bytes", data.len());
                                        }
                                    }
                                    ctx.binary(data);
                                }
                                WebSocketResponse::None => {
                                    if let Ok(settings) = settings.try_read() {
                                        if settings.server_debug.enabled {
                                            info!("[WS] No response to send");
                                        }
                                    }
                                }
                            }
                        }));
                    }
                    Err(e) => error!("[WS] Failed to parse message: {}", e),
                }
            }
            Ok(ws::Message::Binary(data)) => {
                let settings = self.settings.clone();
                let fut = async move {
                    let settings = settings.read().await;
                    let debug_enabled = settings.server_debug.enabled;
                    if debug_enabled {
                        info!("[WS] Received binary message of size: {}", data.len());
                        if settings.server_debug.log_binary_headers {
                            info!("[WS] Binary message first 32 bytes: {:?}", &data.get(..32.min(data.len())));
                        }
                    }
                    // Process binary message
                };
                ctx.spawn(actix::fut::wrap_future(fut));
            }
            Ok(ws::Message::Close(reason)) => {
                info!("[WS] WebSocket closing with reason: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            Err(e) => {
                error!("[WS] WebSocket protocol error: {}", e);
                ctx.stop();
            }
            _ => (),
        }
    }
}

impl SocketFlowServer {
    const POSITION_UPDATE_INTERVAL: Duration = Duration::from_millis(16); // ~60fps

    pub fn new(app_state: Arc<AppState>, settings: Arc<RwLock<crate::config::Settings>>) -> Self {
        Self {
            app_state,
            last_heartbeat: Instant::now(),
            node_order: Vec::new(),
            settings,
        }
    }

    fn start_position_updates(&self, ctx: &mut ws::WebsocketContext<Self>) {
        let app_state = self.app_state.clone();
        let settings = self.settings.clone();

        ctx.run_interval(Self::POSITION_UPDATE_INTERVAL, {
            let app_state = app_state.clone();
            let settings = settings.clone();
            move |_, ctx| {
                let app_state_inner = app_state.clone();
                let settings_inner = settings.clone();
                let fut = async move {
                    let graph = app_state_inner.graph_service.graph_data.read().await;
                    let binary_nodes: Vec<BinaryNodeData> = graph.nodes.iter()
                        .map(|node| BinaryNodeData::from_node_data(&node.data))
                        .collect();

                    // Create binary message
                    let version: i32 = 1;
                    let mut buffer = Vec::with_capacity(4 + binary_nodes.len() * std::mem::size_of::<BinaryNodeData>());
                    
                    // Write version header (4 bytes)
                    buffer.extend_from_slice(&version.to_le_bytes());
                    
                    // Write node data
                    let node_count = binary_nodes.len();
                    for node in &binary_nodes {
                        buffer.extend_from_slice(&node.position[0].to_le_bytes());
                        buffer.extend_from_slice(&node.position[1].to_le_bytes());
                        buffer.extend_from_slice(&node.position[2].to_le_bytes());
                        buffer.extend_from_slice(&node.velocity[0].to_le_bytes());
                        buffer.extend_from_slice(&node.velocity[1].to_le_bytes());
                        buffer.extend_from_slice(&node.velocity[2].to_le_bytes());
                    }

                    if let Ok(settings) = settings_inner.try_read() {
                        if settings.server_debug.enabled && settings.server_debug.log_binary_headers {
                            info!("[WS] Sending position update: {} nodes, {} bytes", 
                                node_count, buffer.len());
                        }
                    }

                    buffer
                };

                ctx.spawn(actix::fut::wrap_future(fut).map(|buffer, _, ctx: &mut ws::WebsocketContext<SocketFlowServer>| {
                    ctx.binary(buffer);
                }));
            }
        });
    }

    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        let settings = self.settings.clone();
        ctx.run_interval(HEARTBEAT_INTERVAL, move |act, ctx| {
            if let Ok(settings_guard) = settings.try_read() {
                let time_since_last = Instant::now().duration_since(act.last_heartbeat);
                let heartbeat_timeout = Duration::from_millis(settings_guard.websocket.heartbeat_timeout);
                
                if settings_guard.server_debug.enable_websocket_debug {
                    info!("[WS] Heartbeat check - Time since last: {:?}", time_since_last);
                }
                
                if time_since_last > heartbeat_timeout {
                    warn!("[WS] Client heartbeat timeout - last heartbeat: {:?} ago", time_since_last);
                    if time_since_last > MAX_CLIENT_TIMEOUT {
                        warn!("[WS] Client exceeded maximum timeout, closing connection");
                        ctx.stop();
                        return;
                    }
                }
                
                // Send ping with timestamp for latency tracking
                let timestamp = Instant::now().elapsed().as_millis().to_string();
                if settings_guard.server_debug.enable_websocket_debug {
                    info!("[WS] Sending ping with timestamp: {}", timestamp);
                }
                ctx.ping(timestamp.as_bytes());
            } else {
                error!("[WS] Failed to acquire settings lock in heartbeat");
            }
        });
    }

    async fn handle_message(&self, message: Message) -> WebSocketResponse {
        let settings = self.settings.read().await;
        let debug_enabled = settings.server_debug.enabled;
        let log_binary = debug_enabled && settings.server_debug.log_binary_headers;
        let log_json = debug_enabled && settings.server_debug.log_full_json;
        
        match message {
            Message::BinaryPositionUpdate { nodes } => {
                if log_binary {
                    info!("[WS] Binary position update with {} nodes", nodes.len());
                }
                WebSocketResponse::None
            },
            Message::UpdatePositions(update_msg) => {
                if log_json {
                    info!("[WS] Update positions message: {:?}", update_msg);
                }
                WebSocketResponse::None
            },
            Message::InitialData { graph } => {
                if log_json {
                    info!("[WS] Initial data message with graph: {:?}", graph);
                }
                WebSocketResponse::None
            },
            Message::SimulationModeSet { mode } => {
                info!("[WS] Simulation mode set to: {}", mode);
                WebSocketResponse::None
            },
            Message::RequestInitialData => {
                info!("[WS] Received request for initial data");
                let graph = self.app_state.graph_service.graph_data.read().await;
                let initial_data = Message::InitialData { 
                    graph: (*graph).clone() 
                };
                
                match serde_json::to_string(&initial_data) {
                    Ok(message) => {
                        if log_json {
                            info!("[WS] Full JSON message: {}", message);
                        }
                        WebSocketResponse::Text(message)
                    }
                    Err(e) => {
                        error!("[WS] Failed to serialize initial data: {}", e);
                        WebSocketResponse::None
                    }
                }
            },
            Message::EnableBinaryUpdates => {
                info!("[WS] Binary updates enabled");
                let graph = self.app_state.graph_service.graph_data.read().await;
                let binary_nodes: Vec<BinaryNodeData> = graph.nodes.iter()
                    .map(|node| BinaryNodeData::from_node_data(&node.data))
                    .collect();

                // Create binary message
                let version: i32 = 1; // Protocol version
                let mut buffer = Vec::with_capacity(4 + binary_nodes.len() * std::mem::size_of::<BinaryNodeData>());
                
                // Write version header (4 bytes)
                buffer.extend_from_slice(&version.to_le_bytes());
                
                // Write node data
                let node_count = binary_nodes.len();
                for node in &binary_nodes {
                    // Write position (12 bytes)
                    buffer.extend_from_slice(&node.position[0].to_le_bytes());
                    buffer.extend_from_slice(&node.position[1].to_le_bytes());
                    buffer.extend_from_slice(&node.position[2].to_le_bytes());
                    
                    // Write velocity (12 bytes)
                    buffer.extend_from_slice(&node.velocity[0].to_le_bytes());
                    buffer.extend_from_slice(&node.velocity[1].to_le_bytes());
                    buffer.extend_from_slice(&node.velocity[2].to_le_bytes());
                }

                if log_binary {
                    info!("[WS] Sending binary update: {} nodes, {} bytes", 
                        node_count, buffer.len());
                    info!("[WS] Binary header: version={}", version);
                }

                WebSocketResponse::Binary(buffer)
            },
            Message::SetSimulationMode { mode } => {
                info!("[WS] Setting simulation mode to: {}", mode);
                match serde_json::to_string(&Message::SimulationModeSet { mode }) {
                    Ok(message) => {
                        if log_json {
                            info!("[WS] Full JSON message: {}", message);
                        }
                        WebSocketResponse::Text(message)
                    }
                    Err(e) => {
                        error!("[WS] Failed to serialize simulation mode: {}", e);
                        WebSocketResponse::None
                    }
                }
            },
            Message::Ping => {
                info!("[WS] Received ping");
                WebSocketResponse::Text("pong".to_string())
            },
            Message::Pong => {
                info!("[WS] Received pong");
                WebSocketResponse::None
            }
        }
    }
}

pub async fn ws_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
    settings: web::Data<Arc<RwLock<crate::config::Settings>>>,
) -> Result<HttpResponse, Error> {
    info!("[WS] New WebSocket connection request from {:?}", req.peer_addr());
    
    // Enhanced connection debugging
    if let Ok(settings_guard) = settings.get_ref().try_read() {
        if settings_guard.server_debug.enabled {
            info!("[WS] WebSocket connection details:");
            info!("[WS] Headers: {:?}", req.headers());
            info!("[WS] URI: {:?}", req.uri());
            info!("[WS] Method: {:?}", req.method());
            info!("[WS] Version: {:?}", req.version());
            if let Some(origin) = req.headers().get("origin") {
                info!("[WS] Origin: {:?}", origin);
            }
            if let Some(protocols) = req.headers().get("sec-websocket-protocol") {
                info!("[WS] Protocols: {:?}", protocols);
            }
            if let Some(upgrade) = req.headers().get("upgrade") {
                info!("[WS] Upgrade header: {:?}", upgrade);
            }
            if let Some(connection) = req.headers().get("connection") {
                info!("[WS] Connection header: {:?}", connection);
            }
            if let Some(key) = req.headers().get("sec-websocket-key") {
                info!("[WS] WebSocket key: {:?}", key);
            }
            if let Some(version) = req.headers().get("sec-websocket-version") {
                info!("[WS] WebSocket version: {:?}", version);
            }
            info!("[WS] Current active connections: {}", app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed));
        }
    }

    // Check connection limits
    let settings_guard = settings.get_ref().try_read().map_err(|_| {
        error!("[WS] Failed to acquire settings lock");
        actix_web::error::ErrorServiceUnavailable("Internal server error")
    })?;
    let max_connections = settings_guard.websocket.max_connections;
    let current_connections = app_state.active_connections.load(std::sync::atomic::Ordering::Relaxed);
    
    if current_connections >= max_connections {
        error!("[WS] Connection limit reached: {}/{}", current_connections, max_connections);
        return Ok(HttpResponse::ServiceUnavailable().finish());
    }

    let socket_server = SocketFlowServer::new(
        app_state.into_inner(),
        settings.get_ref().clone()
    );
    
    info!("[WS] Starting WebSocket connection");
    match ws::start(socket_server, &req, stream) {
        Ok(response) => {
            info!("[WS] WebSocket connection successfully established");
            Ok(response)
        }
        Err(e) => {
            error!("[WS] Failed to establish WebSocket connection: {}", e);
            Err(e)
        }
    }
}
