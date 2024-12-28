use actix::{Actor, StreamHandler};
use actix_web::{web, Error};
use actix_web_actors::ws;
use super::protocol::BinaryProtocol;
use super::constants::{MAX_MESSAGE_SIZE, HEARTBEAT_INTERVAL, MAX_CLIENT_TIMEOUT, POSITION_UPDATE_INTERVAL, MessageType};
use super::routes::decrement_connections;
use std::time::{Instant, Duration};
use std::sync::atomic::Ordering;
use crate::state::AppState;

pub struct WebSocketConnection {
    last_heartbeat: Instant,
    graph_data: web::Data<AppState>,
}

impl WebSocketConnection {
    pub fn new(graph_data: web::Data<AppState>) -> Self {
        Self {
            last_heartbeat: Instant::now(),
            graph_data,
        }
    }

    fn heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_millis(HEARTBEAT_INTERVAL), |act, ctx| {
            if Instant::now().duration_since(act.last_heartbeat) > Duration::from_millis(MAX_CLIENT_TIMEOUT) {
                log::warn!("Client heartbeat timeout, disconnecting!");
                ctx.stop();
                return;
            }

            ctx.ping(b"");
        });
    }

    fn schedule_position_updates(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_millis(POSITION_UPDATE_INTERVAL), |act, ctx| {
            let graph = act.graph_data.graph.read().unwrap();
            let positions: Vec<[f32; 3]> = graph.get_nodes()
                .iter()
                .map(|node| node.position)
                .collect();
            
            if !positions.is_empty() {
                let buffer = BinaryProtocol::create_position_update(&positions);
                ctx.binary(buffer);
            }
        });
    }

    fn handle_message_type(&mut self, msg_type: MessageType, data: &[u8], ctx: &mut ws::WebsocketContext<Self>) {
        match msg_type {
            MessageType::PositionUpdate => {
                if let Ok(positions) = BinaryProtocol::parse_position_update(data) {
                    // Update positions in graph data
                }
            },
            MessageType::GpuComputeStatus => {
                if let Ok(enabled) = BinaryProtocol::parse_gpu_compute_status(data) {
                    self.graph_data.gpu_compute_enabled.store(enabled, Ordering::Relaxed);
                }
            },
            _ => log::warn!("Unhandled message type: {:?}", msg_type),
        }
    }
}

impl Actor for WebSocketConnection {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.heartbeat(ctx);
        self.schedule_position_updates(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        decrement_connections();
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WebSocketConnection {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.last_heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Binary(bin)) => {
                if bin.len() > MAX_MESSAGE_SIZE {
                    ctx.close(Some(ws::CloseReason {
                        code: ws::CloseCode::Size,
                        description: Some("Message too large".to_string()),
                    }));
                    return;
                }

                match BinaryProtocol::validate_header(&bin) {
                    Ok((version, msg_type)) => {
                        log::debug!("Received binary message: version={}, type={:?}", version, msg_type);
                        match msg_type {
                            MessageType::CompressedData => {
                                if let Ok(decompressed) = BinaryProtocol::parse_compressed_update(&bin) {
                                    // Process the decompressed data
                                    if let Ok((_, inner_type)) = BinaryProtocol::validate_header(&decompressed) {
                                        self.handle_message_type(inner_type, &decompressed[8..], ctx);
                                    }
                                }
                            },
                            _ => self.handle_message_type(msg_type, &bin[8..], ctx),
                        }
                    }
                    Err(e) => {
                        log::error!("Invalid binary message: {}", e);
                        ctx.close(Some(ws::CloseReason {
                            code: ws::CloseCode::Protocol,
                            description: Some("Invalid protocol".to_string()),
                        }));
                    }
                }
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => {}
        }
    }
} 