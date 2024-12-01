use crate::app_state::AppState;
use crate::models::metadata::MetadataStore;
use crate::services::file_service::ProcessedFile;
use crate::services::perplexity_service::PerplexityService;
use actix_web::{post, web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;

#[derive(Debug, Serialize, Deserialize)]
pub struct PerplexityRequest {
    pub file_name: String,
}

pub struct PerplexityHandler;

impl PerplexityHandler {
    pub async fn process_file(
        file_name: String,
        perplexity_service: &PerplexityService,
        metadata_store: &mut MetadataStore,
    ) -> Result<ProcessedFile, Box<dyn StdError + Send + Sync>> {
        info!("Processing file with Perplexity: {}", file_name);
        
        let processed_file = perplexity_service.process_file(&file_name).await?;
        
        // Update metadata store with processed file's metadata
        metadata_store.insert(file_name.clone(), processed_file.metadata.clone());
        
        Ok(processed_file)
    }
}

#[post("")]
pub async fn handle_perplexity(
    data: web::Json<PerplexityRequest>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let file_name = data.file_name.clone();
    
    let mut metadata_store = app_state.metadata.write().await;
    
    match PerplexityHandler::process_file(
        file_name,
        &app_state.perplexity_service,
        &mut metadata_store,
    ).await {
        Ok(processed_file) => HttpResponse::Ok().json(processed_file),
        Err(e) => {
            error!("Failed to process file with Perplexity: {}", e);
            HttpResponse::InternalServerError().json("Failed to process file")
        }
    }
}
