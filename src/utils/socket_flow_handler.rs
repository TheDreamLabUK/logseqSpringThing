use actix::{Actor, StreamHandler, ActorContext, AsyncContext, WrapFuture, Handler, Message};
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
    ClientMessage,
    ServerMessage,
};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Message)]
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
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => self.handle_client_message(client_msg, ctx),
                    Err(e) => error!("Failed to parse client message: {}", e),
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

    fn write_settings_to_file(settings: &crate::config::Settings) -> Result<(), String> {
        // Read current settings file
        let settings_content = fs::read_to_string("settings.toml")
            .map_err(|e| format!("Failed to read settings.toml: {}", e))?;

        // Parse current settings to maintain non-visualization sections
        let mut current_settings: toml::Value = toml::from_str(&settings_content)
            .map_err(|e| format!("Failed to parse current settings: {}", e))?;

        // Convert new settings to Value
        let new_settings = toml::Value::try_from(settings)
            .map_err(|e| format!("Failed to convert settings: {}", e))?;

        // Update only visualization-related sections
        if let (Some(current_table), Some(new_table)) = (current_settings.as_table_mut(), new_settings.as_table()) {
            let sections = [
                "rendering", "nodes", "edges", "labels", 
                "bloom", "ar", "physics", "animations", "audio"
            ];

            for section in sections.iter() {
                if let Some(new_section) = new_table.get(*section) {
                    current_table.insert(section.to_string(), new_section.clone());
                }
            }
        }

        // Convert back to TOML string
        let updated_content = toml::to_string_pretty(&current_settings)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;

        // Write back to file
        fs::write("settings.toml", updated_content)
            .map_err(|e| format!("Failed to write settings.toml: {}", e))?;

        Ok(())
    }

    fn handle_client_message(&mut self, message: ClientMessage, ctx: &mut ws::WebsocketContext<Self>) {
        match message {
            ClientMessage::RequestInitialData => {
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
                        debug!("First node position: {:?}", 
                            graph.nodes.first().map(|n| n.data.position)
                        );
                        
                        let initial_data = ServerMessage::InitialData { 
                            graph: (*graph).clone() 
                        };
                        if let Ok(message) = serde_json::to_string(&initial_data) {
                            debug!("Serialized initial data message length: {}", message.len());
                            addr.do_send(SendMessage(message));
                        }
                    }
                    .into_actor(self)
                );
            }
            ClientMessage::UpdatePositions(update_msg) => {
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
            ClientMessage::EnableBinaryUpdates => {
                debug!("Handling EnableBinaryUpdates message");
                let app_state = self.app_state.clone();
                let addr = ctx.address();
                ctx.spawn(
                    async move {
                        let graph = app_state.graph_service.graph_data.read().await;
                        let binary_nodes: Vec<BinaryNodeData> = graph.nodes.iter()
                            .map(|node| BinaryNodeData::from_node_data(&node.id, &node.data))
                            .collect();
                        
                        debug!("Sending binary update with {} nodes", binary_nodes.len());
                        if let Some(first) = binary_nodes.first() {
                            debug!("First node position in binary update: {:?}", first.data.position);
                        }
                        
                        let binary_update = ServerMessage::BinaryPositionUpdate {
                            nodes: binary_nodes
                        };
                        if let Ok(message) = serde_json::to_string(&binary_update) {
                            addr.do_send(SendMessage(message));
                        }
                    }
                    .into_actor(self)
                );
            }
            ClientMessage::SetSimulationMode { mode } => {
                debug!("Setting simulation mode to: {}", mode);
                if let Ok(message) = serde_json::to_string(&ServerMessage::SimulationModeSet { mode }) {
                    ctx.text(message);
                }
            }
            ClientMessage::UpdateSettings { settings } => {
                debug!("Updating settings");
                match Self::write_settings_to_file(&settings) {
                    Ok(_) => {
                        info!("Successfully wrote settings to settings.toml");
                        if let Ok(message) = serde_json::to_string(&ServerMessage::SettingsUpdated { settings }) {
                            ctx.text(message);
                        }
                    },
                    Err(e) => {
                        error!("Failed to write settings to file: {}", e);
                        // Still update in-memory settings even if file write fails
                        if let Ok(message) = serde_json::to_string(&ServerMessage::SettingsUpdated { settings }) {
                            ctx.text(message);
                        }
                    }
                }
            }
            ClientMessage::Ping => {
                debug!("Handling Ping message");
                if let Ok(message) = serde_json::to_string(&ServerMessage::Pong) {
                    ctx.text(message);
                }
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
