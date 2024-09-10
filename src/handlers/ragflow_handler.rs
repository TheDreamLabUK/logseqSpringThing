use actix_web::{web, HttpResponse};
use crate::AppState;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct Message {
    pub content: String,
}

pub async fn send_message(_state: web::Data<AppState>, message: web::Json<Message>) -> HttpResponse {
    // Placeholder implementation
    println!("Received message: {}", message.content);
    HttpResponse::Ok().finish()
}
