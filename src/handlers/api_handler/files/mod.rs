use actix_web::{web, Error as ActixError, HttpResponse};
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
    
    let file_service = FileService::new(state.settings.clone());
    
    match file_service.fetch_and_process_files(&state.content_api, state.settings.clone(), &mut metadata_store).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} public markdown files", processed_files.len());

            {
                let mut metadata = state.metadata.write().await;
                for processed_file in &processed_files {
                    metadata_store.insert(processed_file.file_name.clone(), processed_file.metadata.clone());
                    debug!("Updated metadata for: {}", processed_file.file_name);
                }
                *metadata = metadata_store.clone();
            }

            if let Err(e) = FileService::save_metadata(&metadata_store) {
                error!("Failed to save metadata: {}", e);
                return HttpResponse::InternalServerError().json(json!({
                    "status": "error",
                    "message": format!("Failed to save metadata: {}", e)
                }));
            }

            match GraphService::build_graph(&state).await {
                Ok(graph_data) => {
                    let mut graph = state.graph_service.get_graph_data_mut().await;
                    let mut node_map = state.graph_service.get_node_map_mut().await;
                    *graph = graph_data.clone();
                    
                    // Update node_map with new graph nodes
                    node_map.clear();
                    for node in &graph.nodes {
                        node_map.insert(node.id.clone(), node.clone());
                    }
                    
                    info!("Graph data structure updated successfully");

                    if let Some(gpu) = &state.gpu_compute {
                        if let Ok(_nodes) = gpu.read().await.get_node_data() {
                            debug!("GPU node positions updated successfully");
                        } else {
                            error!("Failed to get node positions from GPU");
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

    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(graph_data) => {
            let mut graph = state.graph_service.get_graph_data_mut().await;
            let mut node_map = state.graph_service.get_node_map_mut().await;
            *graph = graph_data.clone();
            
            // Update node_map with new graph nodes
            node_map.clear();
            for node in &graph.nodes {
                node_map.insert(node.id.clone(), node.clone());
            }
            
            info!("Graph data structure refreshed successfully");

            if let Some(gpu) = &state.gpu_compute {
                if let Ok(_nodes) = gpu.read().await.get_node_data() {
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

    match GraphService::build_graph_from_metadata(&metadata_store).await {
        Ok(graph) => {
            let mut graph_data = state.graph_service.get_graph_data_mut().await;
            let mut node_map = state.graph_service.get_node_map_mut().await;
            *graph_data = graph.clone();
            
            // Update node_map with new graph nodes
            node_map.clear();
            for node in &graph_data.nodes {
                node_map.insert(node.id.clone(), node.clone());
            }
            
            if let Some(gpu) = &state.gpu_compute {
                if let Ok(_nodes) = gpu.read().await.get_node_data() {
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
