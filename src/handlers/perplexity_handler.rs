use crate::AppState;
use actix_web::{post, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use log::{error, info};

#[derive(Debug, Deserialize)]
pub struct PerplexityRequest {
    pub query: String,
    pub conversation_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PerplexityResponse {
    pub answer: String,
    pub conversation_id: String,
}

#[post("")]
pub async fn handle_perplexity(
    request: web::Json<PerplexityRequest>,
    state: web::Data<AppState>,
) -> impl Responder {
    info!("Received perplexity request: {:?}", request);

    let perplexity_service = &state.perplexity_service;
    let conversation_id = request.conversation_id.clone().unwrap_or_else(|| "default".to_string());

    match perplexity_service.query(&request.query, &conversation_id).await {
        Ok(answer) => {
            let response = PerplexityResponse {
                answer,
                conversation_id,
            };
            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            error!("Error processing perplexity request: {}", e);
            HttpResponse::InternalServerError().json(format!("Error: {}", e))
        }
    }
}
