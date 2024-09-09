use crate::AppState;
use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct Message {
    pub content: String,
}

pub async fn send_message(state: web::Data<AppState>, message: web::Json<Message>) -> HttpResponse {
    // Implementation goes here
    HttpResponse::Ok().json(message.0)
}
