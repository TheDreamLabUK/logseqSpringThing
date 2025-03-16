use crate::AppState;
use actix_web::{post, web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use log::{error, info};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerplexityRequest {
    pub query: String,
    pub conversation_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PerplexityResponse {
    pub answer: String,
    pub conversation_id: String,
}

#[post("")]
pub async fn handle_perplexity(
    state: web::Data<AppState>,
    request: web::Json<PerplexityRequest>,
) -> impl Responder {
    info!("Received perplexity request: {:?}", request);

    let perplexity_service = match &state.perplexity_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "Perplexity service is not available"
        }))
    };

    let conversation_id = state.ragflow_session_id.clone();
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
