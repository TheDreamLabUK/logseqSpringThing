use actix::{Actor, StreamHandler, ActorContext, AsyncContext, WrapFuture, Handler, Message as ActixMessage};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use log::{error, warn, debug, info};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::app_state::AppState;
use crate::utils::socket_flow_messages::{
    BinaryNodeData,
    Message,
};
use crate::utils::socket_flow_constants::{
    HEARTBEAT_INTERVAL as HEARTBEAT_INTERVAL_SECS,
    CLIENT_TIMEOUT as CLIENT_TIMEOUT_SECS,
    MAX_CLIENT_TIMEOUT as MAX_CLIENT_TIMEOUT_SECS,
};

// Convert seconds to Duration
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(HEARTBEAT_INTERVAL_SECS);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(CLIENT_TIMEOUT_SECS);
const MAX_CLIENT_TIMEOUT: Duration = Duration::from_secs(MAX_CLIENT_TIMEOUT_SECS);

#[derive(ActixMessage)]
#[rtype(result = "()")]
struct SendMessage(String);

pub struct SocketFlowServer {
    app_state: Arc<AppState>,
    last_heartbeat: Instant,
}

impl Actor for SocketFlowServer {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        debug!("WebSocket actor started");
        self.heartbeat(ctx);
    }
}

impl Handler<SendMessage> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SendMessage, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SocketFlowServer {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.last_heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                debug!("Received text message: {}", text);
                match serde_json::from_str::<Message>(&text) {
                    Ok(message) => self.handle_message(message, ctx),
                    Err(e) => error!("Failed to parse message: {}", e),
                }
            }
            Ok(ws::Message::Binary(_)) => {
                warn!("Unexpected binary message");
            }
            Ok(ws::Message::Close(reason)) => {
                debug!("WebSocket closing with reason: {:?}", reason);
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

impl SocketFlowServer {
    pub fn new(app_state: Arc<AppState>) -> Self {
        Self {
            app_state,
            last_heartbeat: Instant::now(),
        }
    }

    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(HEARTBEAT_INTERVAL, |act, ctx| {
            let time_since_last = Instant::now().duration_since(act.last_heartbeat);
            if time_since_last > CLIENT_TIMEOUT {
                warn!("Client heartbeat timeout - last heartbeat: {:?} ago", time_since_last);
                if time_since_last > MAX_CLIENT_TIMEOUT {
                    warn!("Client exceeded maximum timeout, closing connection");
                    ctx.stop();
                    return;
                }
            }
            ctx.ping(b"");
        });
    }

    fn handle_message(&mut self, message: Message, ctx: &mut ws::WebsocketContext<Self>) {
        match message {
            Message::RequestInitialData => {
                debug!("Handling RequestInitialData message");
                let app_state = self.app_state.clone();
                let addr = ctx.address();
                ctx.spawn(
                    async move {
                        let graph = app_state.graph_service.graph_data.read().await;
                        info!("Sending initial graph data: {} nodes, {} edges", 
                            graph.nodes.len(), 
                            graph.edges.len()
                        );
                        
                        let initial_data = Message::InitialData { 
                            graph: (*graph).clone() 
                        };
                        if let Ok(message) = serde_json::to_string(&initial_data) {
                            addr.do_send(SendMessage(message));
                        }
                    }
                    .into_actor(self)
                );
            }
            Message::UpdatePositions(update_msg) => {
                debug!("Handling UpdatePositions message with {} nodes", update_msg.nodes.len());
                let app_state = self.app_state.clone();
                ctx.spawn(
                    async move {
                        let mut graph = app_state.graph_service.graph_data.write().await;
                        for node_update in update_msg.nodes {
                            if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_update.node_id) {
                                node.data = node_update.data;
                            }
                        }
                    }
                    .into_actor(self)
                );
            }
            Message::EnableBinaryUpdates => {
                debug!("Handling EnableBinaryUpdates message");
                let app_state = self.app_state.clone();
                let addr = ctx.address();
                ctx.spawn(
                    async move {
                        let graph = app_state.graph_service.graph_data.read().await;
                        let binary_nodes: Vec<BinaryNodeData> = graph.nodes.iter()
                            .map(|node| BinaryNodeData::from_node_data(&node.id, &node.data))
                            .collect();
                        
                        let binary_update = Message::BinaryPositionUpdate {
                            nodes: binary_nodes
                        };
                        if let Ok(message) = serde_json::to_string(&binary_update) {
                            addr.do_send(SendMessage(message));
                        }
                    }
                    .into_actor(self)
                );
            }
            Message::SetSimulationMode { mode } => {
                debug!("Setting simulation mode to: {}", mode);
                if let Ok(message) = serde_json::to_string(&Message::SimulationModeSet { mode }) {
                    ctx.text(message);
                }
            }
            Message::Ping => {
                debug!("Handling Ping message");
                if let Ok(message) = serde_json::to_string(&Message::Pong) {
                    ctx.text(message);
                }
            }
            _ => {
                warn!("Unhandled message type");
            }
        }
    }
}

pub async fn ws_handler(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("New WebSocket connection request from {:?}", req.peer_addr());
    debug!("WebSocket request headers: {:?}", req.headers());
    let socket_server = SocketFlowServer::new(Arc::new(app_state.as_ref().clone()));
    ws::start(socket_server, &req, stream)
}
