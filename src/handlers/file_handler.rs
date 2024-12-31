use actix_web::{web::{self, ServiceConfig}, Error as ActixError, HttpResponse};
use serde_json::json;
use log::{info, debug, error};

use crate::AppState;
use crate::services::file_service::FileService;
use crate::services::graph_service::GraphService;

pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    info!("Initiating optimized file fetch and processing");

    // Load or create metadata
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
    
    // Process files with optimized approach
    let file_service = FileService::new(state.settings.clone());
    match file_service.fetch_and_process_files(&*state.github_service, state.settings.clone(), &mut metadata_store).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} public markdown files", processed_files.len());

            // Update metadata store
            {
                let mut metadata = state.metadata.write().await;
                for processed_file in &processed_files {
                    metadata_store.insert(processed_file.file_name.clone(), processed_file.metadata.clone());
                    debug!("Updated metadata for: {}", processed_file.file_name);
                }
                *metadata = metadata_store.clone();
            }

            // Save the updated metadata
            if let Err(e) = FileService::save_metadata(&metadata_store) {
                error!("Failed to save metadata: {}", e);
                return HttpResponse::InternalServerError().json(json!({
                    "status": "error",
                    "message": format!("Failed to save metadata: {}", e)
                }));
            }

            HttpResponse::Ok().json(json!({
                "status": "success",
                "message": format!("Successfully processed {} files", processed_files.len()),
                "files": file_names
            }))
        }
        Err(e) => {
            error!("Failed to process files: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to process files: {}", e)
            }))
        }
    }
}

pub async fn get_file_content(_state: web::Data<AppState>, file_name: web::Path<String>) -> HttpResponse {
    let file_path = format!("{}/{}", crate::services::file_service::MARKDOWN_DIR, file_name);
    
    match tokio::fs::read_to_string(&file_path).await {
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

    // Load metadata from file
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

    // Build graph directly from metadata
    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(graph_data) => {
            let mut graph = state.graph_service.graph_data.write().await;
            *graph = graph_data.clone();
            info!("Graph data structure refreshed successfully");

            // Send binary position update to clients
            if let Some(gpu) = &state.gpu_compute {
                if let Ok(_nodes) = gpu.read().await.get_node_data() {
                    // Note: Socket-flow server will handle broadcasting
                    debug!("GPU node positions updated successfully");
                } else {
                    error!("Failed to get node positions from GPU");
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

pub fn config(cfg: &mut ServiceConfig) {
    cfg.service(web::resource("/fetch").to(fetch_and_process_files))
       .service(web::resource("/content/{file_name}").to(get_file_content))
       .service(web::resource("/refresh").to(refresh_graph))
       .service(web::resource("/update").to(update_graph));
}

pub async fn update_graph(state: web::Data<AppState>) -> Result<HttpResponse, ActixError> {
    // Load metadata from file
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

    // Build graph directly from metadata
    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(graph) => {
            // Update graph data
            *state.graph_service.graph_data.write().await = graph.clone();
            
            // Send binary position update to clients
            if let Some(gpu) = &state.gpu_compute {
                if let Ok(_nodes) = gpu.read().await.get_node_data() {
                    // Note: Socket-flow server will handle broadcasting
                    debug!("GPU node positions updated successfully");
                } else {
                    error!("Failed to get node positions from GPU");
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
