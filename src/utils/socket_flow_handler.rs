use actix::{Actor, StreamHandler, ActorContext, AsyncContext, WrapFuture, Handler, Message as ActixMessage};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use log::{error, warn, debug, info};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::fs;
use toml;

use crate::app_state::AppState;
use crate::utils::socket_flow_messages::{
    BinaryNodeData,
    Message,
    SettingsUpdate,
    UpdateSettings,
};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

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
            if Instant::now().duration_since(act.last_heartbeat) > CLIENT_TIMEOUT {
                warn!("Client heartbeat timeout");
                ctx.stop();
                return;
            }
            ctx.ping(b"");
        });
    }

    fn write_settings_to_file(settings: &serde_json::Value) -> Result<(), String> {
        // Try to read current settings file
        let settings_content = match fs::read_to_string("settings.toml") {
            Ok(content) => content,
            Err(e) => {
                warn!("Could not read settings.toml: {}", e);
                return Ok(());  // Continue with in-memory settings
            }
        };

        // Try to parse current settings
        let mut current_settings: toml::Value = match toml::from_str(&settings_content) {
            Ok(settings) => settings,
            Err(e) => {
                warn!("Could not parse settings.toml: {}", e);
                return Ok(());  // Continue with in-memory settings
            }
        };

        // Update visualization-related sections
        if let Some(current_table) = current_settings.as_table_mut() {
            if let Some(new_settings) = settings.as_object() {
                let sections = [
                    "rendering", "nodes", "edges", "labels", 
                    "bloom", "ar", "physics", "animations", "audio"
                ];

                for section in sections.iter() {
                    if let Some(new_section) = new_settings.get(*section) {
                        if let Ok(converted) = toml::Value::try_from(new_section.clone()) {
                            current_table.insert(section.to_string(), converted);
                        }
                    }
                }
            }
        }

        // Try to write back to file, but don't fail if we can't
        match toml::to_string_pretty(&current_settings)
            .map_err(|e| format!("Failed to serialize settings: {}", e))
            .and_then(|content| fs::write("settings.toml", content)
                .map_err(|e| format!("Failed to write settings.toml: {}", e))) 
        {
            Ok(_) => {
                debug!("Successfully wrote settings to file");
                Ok(())
            },
            Err(e) => {
                warn!("Could not write to settings.toml ({}), continuing with in-memory settings", e);
                Ok(())  // Continue with in-memory settings
            }
        }
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
            Message::UpdateSettings(UpdateSettings { settings }) => {
                debug!("Updating settings");
                // Try to write to file but continue even if it fails
                let _ = Self::write_settings_to_file(&settings);
                
                // Always broadcast settings update to all clients
                let settings_update = Message::SettingsUpdated(SettingsUpdate { settings });
                if let Ok(message) = serde_json::to_string(&settings_update) {
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
    info!("New WebSocket connection request");
    let socket_server = SocketFlowServer::new(Arc::new(app_state.as_ref().clone()));
    ws::start(socket_server, &req, stream)
}
