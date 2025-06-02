use actix_web::{web, Error as ActixError, HttpResponse};
use std::sync::Arc;
use crate::actors::messages::{GetSettings, UpdateMetadata, BuildGraphFromMetadata, GetNodeData as GetGpuNodeData};
use crate::models::graph::GraphData as ModelsGraphData;
use serde_json::json;
use log::{info, debug, error};

use crate::AppState;
use crate::services::file_service::{FileService, MARKDOWN_DIR};
use crate::services::graph_service::GraphService;

pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    info!("Initiating optimized file fetch and processing");

    let mut metadata_store = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load or create metadata: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to initialize metadata: {}", e)
            }));
        }
    };
    
    let settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => Arc::new(s),
        _ => {
            error!("Failed to retrieve settings from SettingsActor");
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": "Failed to retrieve application settings"
            }));
        }
    };
    
    let file_service = FileService::new(settings.clone());
    
    match file_service.fetch_and_process_files(state.content_api.clone(), settings.clone(), &mut metadata_store).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} public markdown files", processed_files.len());

            {
                // Send UpdateMetadata message to MetadataActor
                if let Err(e) = state.metadata_addr.send(UpdateMetadata { metadata: metadata_store.clone() }).await {
                    error!("Failed to send UpdateMetadata message to MetadataActor: {}", e);
                    // Decide if this is a critical error to return
                }
            }

            // FileService::save_metadata might also need to be an actor message if it implies shared state,
            // but for now, assuming it's a static utility or local file operation.
            // If it interacts with shared state, it should be refactored.
            if let Err(e) = FileService::save_metadata(&metadata_store) {
                error!("Failed to save metadata: {}", e);
                return HttpResponse::InternalServerError().json(json!({
                    "status": "error",
                    "message": format!("Failed to save metadata: {}", e)
                }));
            }

            // Send BuildGraphFromMetadata message to GraphServiceActor
            match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
                Ok(Ok(())) => {
                    info!("Graph data structure updated successfully via GraphServiceActor");

                    // If GPU is present, potentially trigger an update or fetch data
                    if let Some(gpu_addr) = &state.gpu_compute_addr {
                        // Example: Trigger GPU re-initialization or update if necessary
                        // This depends on how GPUComputeActor handles graph updates.
                        // For now, let's assume GraphServiceActor coordinates with GPUComputeActor if needed.
                        // Or, if we just need to get data for a response:
                        match gpu_addr.send(GetGpuNodeData).await {
                            Ok(Ok(_nodes)) => {
                                debug!("GPU node data fetched successfully after graph update");
                            }
                            Ok(Err(e)) => {
                                error!("Failed to get node data from GPU actor: {}", e);
                            }
                            Err(e) => {
                                error!("Mailbox error getting node data from GPU actor: {}", e);
                            }
                        }
                    }

                    HttpResponse::Ok().json(json!({
                        "status": "success",
                        "processed_files": file_names
                    }))
                },
                Err(e) => {
                    error!("Failed to build graph data: {}", e);
                    HttpResponse::InternalServerError().json(json!({
                        "status": "error",
                        "message": format!("Failed to build graph data: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            error!("Error processing files: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Error processing files: {}", e)
            }))
        }
    }
}

pub async fn get_file_content(_state: web::Data<AppState>, file_name: web::Path<String>) -> HttpResponse {
    let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
    match std::fs::read_to_string(&file_path) {
        Ok(content) => HttpResponse::Ok().body(content),
        Err(e) => {
            error!("Failed to read file {}: {}", file_name, e);
            HttpResponse::NotFound().json(json!({
                "status": "error",
                "message": format!("File not found or unreadable: {}", file_name)
            }))
        }
    }
}

pub async fn refresh_graph(state: web::Data<AppState>) -> HttpResponse {
    info!("Manually triggering graph refresh");

    let metadata_store = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to load metadata: {}", e)
            }));
        }
    };

    match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
        Ok(Ok(())) => {
            info!("Graph data structure refreshed successfully via GraphServiceActor");

            if let Some(gpu_addr) = &state.gpu_compute_addr {
                 match gpu_addr.send(GetGpuNodeData).await {
                    Ok(Ok(_nodes)) => {
                        debug!("GPU node data fetched successfully after graph refresh");
                    }
                    Ok(Err(e)) => {
                        error!("Failed to get node data from GPU actor after refresh: {}", e);
                    }
                    Err(e) => {
                        error!("Mailbox error getting node data from GPU actor after refresh: {}", e);
                    }
                }
            }

            HttpResponse::Ok().json(json!({
                "status": "success",
                "message": "Graph refreshed successfully"
            }))
        },
        Err(e) => {
            error!("Failed to refresh graph data: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to refresh graph data: {}", e)
            }))
        }
    }
}

pub async fn update_graph(state: web::Data<AppState>) -> Result<HttpResponse, ActixError> {
    let metadata_store = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to load metadata: {}", e)
            })));
        }
    };

    match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
        Ok(Ok(())) => {
            info!("Graph data structure updated successfully via GraphServiceActor in update_graph");

            if let Some(gpu_addr) = &state.gpu_compute_addr {
                match gpu_addr.send(GetGpuNodeData).await {
                    Ok(Ok(_nodes)) => {
                        debug!("GPU node data fetched successfully after graph update in update_graph");
                    }
                    Ok(Err(e)) => {
                        error!("Failed to get node data from GPU actor after update in update_graph: {}", e);
                    }
                    Err(e) => {
                        error!("Mailbox error getting node data from GPU actor after update in update_graph: {}", e);
                    }
                }
            }
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "success",
                "message": "Graph updated successfully"
            })))
        },
        Err(e) => {
            error!("Failed to build graph: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to build graph: {}", e)
            })))
        }
    }
}

// Configure routes using snake_case
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/files")
            .route("/process", web::post().to(fetch_and_process_files))
            .route("/get_content/{filename}", web::get().to(get_file_content))
            .route("/refresh_graph", web::post().to(refresh_graph))
            .route("/update_graph", web::post().to(update_graph))
    );
}
