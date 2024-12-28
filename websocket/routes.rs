use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use super::handler::WebSocketConnection;
use super::constants::MAX_CONNECTIONS;
use std::sync::atomic::{AtomicUsize, Ordering};

static CONNECTIONS: AtomicUsize = AtomicUsize::new(0);

pub async fn ws_route(
    req: HttpRequest,
    stream: web::Payload,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    // Check connection limit
    let connections = CONNECTIONS.load(Ordering::SeqCst);
    if connections >= MAX_CONNECTIONS {
        log::warn!("Connection limit reached: {}", connections);
        return Ok(HttpResponse::ServiceUnavailable().finish());
    }

    // Increment connection counter
    CONNECTIONS.fetch_add(1, Ordering::SeqCst);

    // Create WebSocket connection
    let ws = WebSocketConnection::new(state);
    let resp = ws::start(ws, &req, stream)?;

    log::info!("New WebSocket connection established. Total connections: {}", 
        CONNECTIONS.load(Ordering::SeqCst));

    Ok(resp)
}

pub fn decrement_connections() {
    CONNECTIONS.fetch_sub(1, Ordering::SeqCst);
    log::debug!("Connection closed. Total connections: {}", 
        CONNECTIONS.load(Ordering::SeqCst));
} 