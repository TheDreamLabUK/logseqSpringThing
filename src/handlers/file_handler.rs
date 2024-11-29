use actix_web::{web, Error as ActixError, HttpResponse};
use serde_json::json;
use log::{info, error, debug};
use std::sync::Arc;

use crate::AppState;
use crate::services::file_service::FileService;
use crate::services::graph_service::GraphService;
use crate::utils::websocket_manager::{BroadcastGraph, BroadcastError};

pub async fn handle_file_upload(
    state: web::Data<AppState>,
    file_service: web::Data<FileService>,
    payload: web::Bytes,
) -> Result<HttpResponse, ActixError> {
    debug!("Handling file upload request");

    match file_service.get_ref().process_file_upload(payload).await {
        Ok(graph_data) => {
            debug!("File processed successfully, updating graph data");
            
            // Update shared graph data
            {
                let mut graph = state.graph_data.write().await;
                *graph = graph_data.clone();
            }

            // Broadcast update to all connected clients
            let graph_msg = BroadcastGraph {
                graph: Arc::new(graph_data)
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(graph_msg);
            }

            Ok(HttpResponse::Ok().json(json!({
                "message": "File processed successfully"
            })))
        },
        Err(e) => {
            error!("Error processing file: {}", e);
            
            // Broadcast error to clients
            let error_msg = BroadcastError {
                message: format!("Error processing file: {}", e)
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(error_msg);
            }

            Ok(HttpResponse::BadRequest().json(json!({
                "error": format!("Error processing file: {}", e)
            })))
        }
    }
}

pub async fn handle_file_list(
    state: web::Data<AppState>,
    file_service: web::Data<FileService>,
) -> Result<HttpResponse, ActixError> {
    debug!("Handling file list request");

    match file_service.get_ref().list_files().await {
        Ok(files) => {
            Ok(HttpResponse::Ok().json(json!({
                "files": files
            })))
        },
        Err(e) => {
            error!("Error listing files: {}", e);
            
            // Broadcast error to clients
            let error_msg = BroadcastError {
                message: format!("Error listing files: {}", e)
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(error_msg);
            }

            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Error listing files: {}", e)
            })))
        }
    }
}

pub async fn handle_file_load(
    state: web::Data<AppState>,
    file_service: web::Data<FileService>,
    filename: web::Path<String>,
) -> Result<HttpResponse, ActixError> {
    debug!("Handling file load request for: {}", filename);

    match file_service.get_ref().load_file(&filename).await {
        Ok(graph_data) => {
            debug!("File loaded successfully, updating graph data");
            
            // Update shared graph data
            {
                let mut graph = state.graph_data.write().await;
                *graph = graph_data.clone();
            }

            // Broadcast update to all connected clients
            let graph_msg = BroadcastGraph {
                graph: Arc::new(graph_data)
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(graph_msg);
            }

            Ok(HttpResponse::Ok().json(json!({
                "message": "File loaded successfully"
            })))
        },
        Err(e) => {
            error!("Error loading file: {}", e);
            
            // Broadcast error to clients
            let error_msg = BroadcastError {
                message: format!("Error loading file: {}", e)
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(error_msg);
            }

            Ok(HttpResponse::BadRequest().json(json!({
                "error": format!("Error loading file: {}", e)
            })))
        }
    }
}

pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    info!("Initiating optimized file fetch and processing");

    // Load or create metadata, which now ensures directories exist
    let mut metadata_map = match FileService::load_or_create_metadata() {
        Ok(map) => map,
        Err(e) => {
            error!("Failed to load or create metadata: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to initialize metadata: {}", e)
            }));
        }
    };
    
    // Process files with optimized approach
    match FileService::fetch_and_process_files(&*state.github_service, state.settings.clone(), &mut metadata_map).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} public markdown files", processed_files.len());

            // Update file cache and graph metadata with processed files
            {
                let mut file_cache = state.file_cache.write().await;
                let mut graph = state.graph_data.write().await;
                for processed_file in &processed_files {
                    // Only public files reach this point due to optimization
                    metadata_map.insert(processed_file.file_name.clone(), processed_file.metadata.clone());
                    file_cache.insert(processed_file.file_name.clone(), processed_file.content.clone());
                    debug!("Updated file cache with: {}", processed_file.file_name);
                }
                // Update graph metadata
                graph.metadata = metadata_map.clone();
            }

            // Save the updated metadata
            if let Err(e) = FileService::save_metadata(&metadata_map) {
                error!("Failed to save metadata: {}", e);
                return HttpResponse::InternalServerError().json(json!({
                    "status": "error",
                    "message": format!("Failed to save metadata: {}", e)
                }));
            }

            // Update graph with processed files
            match GraphService::build_graph(&state).await {
                Ok(graph_data) => {
                    let mut graph = state.graph_data.write().await;
                    *graph = graph_data.clone();
                    info!("Graph data structure updated successfully");

                    // Broadcast update to all connected clients
                    let graph_msg = BroadcastGraph {
                        graph: Arc::new(graph_data)
                    };

                    if let Some(addr) = state.websocket_manager.get_addr() {
                        addr.do_send(graph_msg);
                        debug!("Graph update broadcasted successfully");
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
            error!("Error processing files: {:?}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Error processing files: {:?}", e)
            }))
        }
    }
}

pub async fn get_file_content(state: web::Data<AppState>, file_name: web::Path<String>) -> HttpResponse {
    let file_cache = state.file_cache.read().await;
    
    match file_cache.get(file_name.as_str()) {
        Some(content) => HttpResponse::Ok().body(content.clone()),
        None => {
            error!("File not found in cache: {}", file_name);
            HttpResponse::NotFound().json(json!({
                "status": "error",
                "message": format!("File not found: {}", file_name)
            }))
        }
    }
}

pub async fn refresh_graph(state: web::Data<AppState>) -> HttpResponse {
    info!("Manually triggering graph refresh");

    // Load metadata from file
    let metadata_map = match FileService::load_or_create_metadata() {
        Ok(map) => map,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to load metadata: {}", e)
            }));
        }
    };

    // Build graph directly from metadata
    match GraphService::build_graph_from_metadata(&metadata_map).await {
        Ok(graph_data) => {
            let mut graph = state.graph_data.write().await;
            *graph = graph_data.clone();
            info!("Graph data structure refreshed successfully");

            // Broadcast update to all connected clients
            let graph_msg = BroadcastGraph {
                graph: Arc::new(graph_data)
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(graph_msg);
                debug!("Graph update broadcasted successfully");
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
    let metadata_map = match FileService::load_or_create_metadata() {
        Ok(map) => map,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to load metadata: {}", e)
            })));
        }
    };

    // Build graph directly from metadata
    match GraphService::build_graph_from_metadata(&metadata_map).await {
        Ok(graph) => {
            // Update graph data
            *state.graph_data.write().await = graph.clone();
            
            // Broadcast update to all connected clients
            let graph_msg = BroadcastGraph {
                graph: Arc::new(graph.clone())
            };

            if let Some(addr) = state.websocket_manager.get_addr() {
                addr.do_send(graph_msg);
            }
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "success",
                "message": "Graph updated successfully",
                "data": graph
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
