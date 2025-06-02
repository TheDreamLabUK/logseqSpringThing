//! Client Manager Actor to replace static APP_CLIENT_MANAGER singleton

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::actors::messages::*;
use crate::handlers::socket_flow_handler::SocketFlowServer;
use actix_web_actors::ws::Message as WsMessage;
use log::{debug, warn};

pub struct ClientManagerActor {
    clients: HashMap<usize, Addr<SocketFlowServer>>,
    next_id: AtomicUsize,
}

impl ClientManagerActor {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            next_id: AtomicUsize::new(1),
        }
    }

    pub fn register_client(&mut self, addr: Addr<SocketFlowServer>) -> usize {
        let client_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.clients.insert(client_id, addr);
        debug!("Client {} registered. Total clients: {}", client_id, self.clients.len());
        client_id
    }

    pub fn unregister_client(&mut self, client_id: usize) {
        if self.clients.remove(&client_id).is_some() {
            debug!("Client {} unregistered. Total clients: {}", client_id, self.clients.len());
        } else {
            warn!("Attempted to unregister non-existent client {}", client_id);
        }
    }

    pub fn broadcast_to_all(&self, data: Vec<u8>) {
        if self.clients.is_empty() {
            return;
        }

        debug!("Broadcasting {} bytes to {} clients", data.len(), self.clients.len());
        
        for (client_id, addr) in &self.clients {
            addr.do_send(WsMessage::Binary(data.clone()));
        }
    }

    pub fn broadcast_message(&self, message: String) {
        if self.clients.is_empty() {
            return;
        }

        debug!("Broadcasting message to {} clients", self.clients.len());
        
        for (client_id, addr) in &self.clients {
            addr.do_send(WsMessage::Text(message.clone()));
        }
    }

    pub fn get_client_count(&self) -> usize {
        self.clients.len()
    }
}

impl Actor for ClientManagerActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        debug!("ClientManagerActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        debug!("ClientManagerActor stopped with {} clients", self.clients.len());
    }
}

impl Handler<RegisterClient> for ClientManagerActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, msg: RegisterClient, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.register_client(msg.addr))
    }
}

impl Handler<UnregisterClient> for ClientManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UnregisterClient, _ctx: &mut Self::Context) -> Self::Result {
        self.unregister_client(msg.client_id);
        Ok(())
    }
}

impl Handler<BroadcastNodePositions> for ClientManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        self.broadcast_to_all(msg.positions);
        Ok(())
    }
}

impl Handler<BroadcastMessage> for ClientManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.broadcast_message(msg.message);
        Ok(())
    }
}

impl Handler<GetClientCount> for ClientManagerActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, _msg: GetClientCount, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_client_count())
    }
}