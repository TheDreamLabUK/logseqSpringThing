use actix_web::{web::{self, ServiceConfig}, HttpResponse, Responder, Result, Error as ActixError};
use serde_json::json;
use crate::AppState;
use serde::{Serialize, Deserialize};
use log::{info, debug, error, warn};
use std::collections::HashMap;
use std::sync::Arc;
use crate::models::metadata::Metadata;
use crate::utils::socket_flow_messages::Node;
use crate::services::file_service::FileService;
use crate::services::graph_service::GraphService;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<crate::models::edge::Edge>,
    pub metadata: HashMap<String, Metadata>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PaginatedGraphResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<crate::models::edge::Edge>,
    pub metadata: HashMap<String, Metadata>,
    pub total_pages: usize,
    pub current_page: usize,
    pub total_items: usize,
    pub page_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphQuery {
    pub query: Option<String>,
    pub page: Option<usize>,
    #[serde(rename = "pageSize")]
    pub page_size: Option<usize>,
    pub sort: Option<String>,
    pub filter: Option<String>,
}

pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    info!("Received request for graph data");
    let graph = state.graph_service.graph_data.read().await;
    
    debug!("Preparing graph response with {} nodes and {} edges",
        graph.nodes.len(),
        graph.edges.len()
    );

    let response = GraphResponse {
        nodes: graph.nodes.clone(),
        edges: graph.edges.clone(),
        metadata: graph.metadata.clone(),
    };

    HttpResponse::Ok().json(response)
}

pub async fn get_paginated_graph_data(
    state: web::Data<AppState>,
    query: web::Query<GraphQuery>,
) -> impl Responder {
    info!("Received request for paginated graph data with params: {:?}", query);

    // Convert to 0-based indexing internally
    let page = query.page.map(|p| p.saturating_sub(1)).unwrap_or(0);
    let page_size = query.page_size.unwrap_or(100);

    if page_size == 0 {
        error!("Invalid page size: {}", page_size);
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Page size must be greater than 0"
        }));
    }

    let graph = state.graph_service.graph_data.read().await;
    let total_items = graph.nodes.len();
    
    if total_items == 0 {
        debug!("Graph is empty");
        return HttpResponse::Ok().json(PaginatedGraphResponse {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
            total_pages: 0,
            current_page: 1, // Return 1-based page number
            total_items: 0,
            page_size,
        });
    }

    let total_pages = (total_items + page_size - 1) / page_size;

    if page >= total_pages {
        warn!("Requested page {} exceeds total pages {}", page + 1, total_pages);
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": format!("Page {} exceeds total available pages {}", page + 1, total_pages)
        }));
    }

    let start = page * page_size;
    let end = std::cmp::min(start + page_size, total_items);

    debug!("Calculating slice from {} to {} out of {} total items", start, end, total_items);

    let page_nodes = graph.nodes[start..end].to_vec();

    // Get edges where either source or target is in the current page
    let node_ids: std::collections::HashSet<_> = page_nodes.iter()
        .map(|node| node.id.clone())
        .collect();

    let relevant_edges: Vec<_> = graph.edges.iter()
        .filter(|edge| {
            // Include edges where either the source or target is in our page
            node_ids.contains(&edge.source) || node_ids.contains(&edge.target)
        })
        .cloned()
        .collect();

    debug!("Found {} relevant edges for {} nodes", relevant_edges.len(), page_nodes.len());

    let response = PaginatedGraphResponse {
        nodes: page_nodes,
        edges: relevant_edges,
        metadata: graph.metadata.clone(),
        total_pages,
        current_page: page + 1, // Convert back to 1-based indexing for response
        total_items,
        page_size,
    };

    HttpResponse::Ok().json(response)
}

// Rebuild graph from existing metadata
pub async fn refresh_graph(state: web::Data<AppState>) -> Result<HttpResponse, ActixError> {
    info!("Refreshing graph data");

    // Load or create metadata
    let mut metadata = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load or create metadata: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to initialize metadata: {}", e)
            })));
        }
    };

    // Process files with optimized approach
    let file_service = FileService::new(state.settings.clone());
    match file_service.fetch_and_process_files(&*state.github_service, state.settings.clone(), &mut metadata).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} public markdown files", processed_files.len());

            // Update metadata store
            {
                let mut metadata_store = state.metadata.write().await;
                for processed_file in &processed_files {
                    metadata_store.insert(processed_file.file_name.clone(), processed_file.metadata.clone());
                }
            }

            // Save the updated metadata
            if let Err(e) = FileService::save_metadata(&metadata) {
                error!("Failed to save metadata: {}", e);
                return Ok(HttpResponse::InternalServerError().json(json!({
                    "status": "error",
                    "message": format!("Failed to save metadata: {}", e)
                })));
            }

            Ok(HttpResponse::Ok().json(json!({
                "status": "success",
                "message": format!("Successfully processed {} files", processed_files.len()),
                "files": file_names
            })))
        }
        Err(e) => {
            error!("Failed to process files: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to process files: {}", e)
            })))
        }
    }
}

// Fetch new metadata and rebuild graph
pub fn config(cfg: &mut ServiceConfig) {
    cfg.service(web::resource("/data").to(get_graph_data))
       .service(web::resource("/data/paginated").to(get_paginated_graph_data))
       .service(
           web::resource("/update")
               .route(web::post().to(update_graph))
       );
}

pub async fn update_graph(state: web::Data<AppState>) -> impl Responder {
    info!("Received request to update graph");
    
    // Load current metadata
    let mut metadata = match FileService::load_or_create_metadata() {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("Failed to load metadata: {}", e)
            }));
        }
    };
    
    // Fetch and process new files
    match FileService::fetch_and_process_files(&*state.github_service, Arc::clone(&state.settings), &mut metadata).await {
        Ok(processed_files) => {
            if processed_files.is_empty() {
                debug!("No new files to process");
                return HttpResponse::Ok().json(serde_json::json!({
                    "success": true,
                    "message": "No updates needed"
                }));
            }
            
            debug!("Processing {} new files", processed_files.len());
            
            // Update metadata in app state
            {
                let mut app_metadata = state.metadata.write().await;
                *app_metadata = metadata.clone();
            }
            
            // Build new graph
            match GraphService::build_graph_from_metadata(&metadata).await {
                Ok(mut new_graph) => {
                    let mut graph = state.graph_service.graph_data.write().await;
                    
                    // Preserve existing node positions
                    let old_positions: HashMap<String, (f32, f32, f32)> = graph.nodes.iter()
                        .map(|node| (node.id.clone(), (node.x(), node.y(), node.z())))
                        .collect();
                    
                    // Update positions in new graph
                    for node in &mut new_graph.nodes {
                        if let Some(&(x, y, z)) = old_positions.get(&node.id) {
                            node.set_x(x);
                            node.set_y(y);
                            node.set_z(z);
                        }
                    }

                    // Calculate layout using GPU if available
                    let settings = state.settings.read().await;
                    let params = settings.graph.simulation_params.clone();
                    drop(settings);

                    if let Err(e) = GraphService::calculate_layout(
                        &state.gpu_compute,
                        &mut new_graph,
                        &params
                    ).await {
                        error!("Failed to calculate layout: {}", e);
                    }
                    
                    *graph = new_graph;
                    debug!("Graph updated successfully");
                    
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": true,
                        "message": format!("Graph updated with {} new files", processed_files.len())
                    }))
                },
                Err(e) => {
                    error!("Failed to build graph: {}", e);
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to build graph: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            error!("Failed to fetch and process files: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("Failed to fetch and process files: {}", e)
            }))
        }
    }
}
