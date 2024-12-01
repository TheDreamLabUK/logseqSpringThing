use actix_web::{web, Error as ActixError, HttpResponse};
use serde_json::json;
use log::{info, debug, error};

use crate::AppState;
use crate::services::file_service::{FileService, MARKDOWN_DIR};
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
    match FileService::fetch_and_process_files(&*state.github_service, state.settings.clone(), &mut metadata_store).await {
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

            // Update graph with processed files
            match GraphService::build_graph(&state).await {
                Ok(graph_data) => {
                    let mut graph = state.graph_service.graph_data.write().await;
                    *graph = graph_data.clone();
                    info!("Graph data structure updated successfully");

                    // Send binary position update to clients
                    if let Some(gpu) = &state.gpu_compute {
                        let gpu = gpu.clone();
                        let gpu_write = gpu.write().await;
                        if let Ok(nodes) = gpu_write.get_node_positions().await {
                            if let Err(e) = state.websocket_manager.broadcast_binary(&nodes, true).await {
                                error!("Failed to broadcast binary update: {}", e);
                            }
                        }
                    }

                    // Send metadata update separately as JSON
                    let metadata_msg = json!({
                        "type": "metadata_update",
                        "metadata": graph_data.metadata
                    });

                    if let Err(e) = state.websocket_manager.broadcast_message(metadata_msg.to_string()).await {
                        error!("Failed to broadcast metadata update: {}", e);
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
    // Read file directly from disk
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
                let gpu = gpu.clone();
                let gpu_write = gpu.write().await;
                if let Ok(nodes) = gpu_write.get_node_positions().await {
                    if let Err(e) = state.websocket_manager.broadcast_binary(&nodes, true).await {
                        error!("Failed to broadcast binary update: {}", e);
                    }
                }
            }

            // Send metadata update separately as JSON
            let metadata_msg = json!({
                "type": "metadata_update",
                "metadata": graph_data.metadata
            });

            if let Err(e) = state.websocket_manager.broadcast_message(metadata_msg.to_string()).await {
                error!("Failed to broadcast metadata update: {}", e);
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
                let gpu = gpu.clone();
                let gpu_write = gpu.write().await;
                if let Ok(nodes) = gpu_write.get_node_positions().await {
                    if let Err(e) = state.websocket_manager.broadcast_binary(&nodes, true).await {
                        error!("Failed to broadcast binary update: {}", e);
                    }
                }
            }

            // Send metadata update separately as JSON
            let metadata_msg = json!({
                "type": "metadata_update",
                "metadata": graph.metadata
            });

            if let Err(e) = state.websocket_manager.broadcast_message(metadata_msg.to_string()).await {
                error!("Failed to broadcast metadata update: {}", e);
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
