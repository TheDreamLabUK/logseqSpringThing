use actix_web::{web, HttpResponse, Responder};
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
#[serde(rename_all = "snake_case")]
pub struct GraphResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<crate::models::edge::Edge>,
    pub metadata: HashMap<String, Metadata>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
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
            current_page: 1,
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

    let node_ids: std::collections::HashSet<_> = page_nodes.iter()
        .map(|node| node.id.clone())
        .collect();

    let relevant_edges: Vec<_> = graph.edges.iter()
        .filter(|edge| {
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
        current_page: page + 1,
        total_items,
        page_size,
    };

    HttpResponse::Ok().json(response)
}

pub async fn refresh_graph(state: web::Data<AppState>) -> impl Responder {
    info!("Received request to refresh graph");
    
    let metadata = state.metadata.read().await.clone();
    debug!("Building graph from {} metadata entries", metadata.len());
    
    match GraphService::build_graph_from_metadata(&metadata).await {
        Ok(mut new_graph) => {
            let mut graph = state.graph_service.graph_data.write().await;
            
            let old_positions: HashMap<String, (f32, f32, f32)> = graph.nodes.iter()
                .map(|node| (node.id.clone(), (node.x(), node.y(), node.z())))
                .collect();
            
            debug!("Preserved positions for {} existing nodes", old_positions.len());
            
            for node in &mut new_graph.nodes {
                if let Some(&(x, y, z)) = old_positions.get(&node.id) {
                    node.set_x(x);
                    node.set_y(y);
                    node.set_z(z);
                }
            }
            
            *graph = new_graph;
            info!("Graph refreshed successfully with {} nodes and {} edges", 
                graph.nodes.len(), 
                graph.edges.len()
            );
            
            HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "message": "Graph refreshed successfully"
            }))
        },
        Err(e) => {
            error!("Failed to refresh graph: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("Failed to refresh graph: {}", e)
            }))
        }
    }
}

pub async fn update_graph(state: web::Data<AppState>) -> impl Responder {
    info!("Received request to update graph");
    
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
    
    let file_service = FileService::new(Arc::clone(&state.settings));
    match file_service.fetch_and_process_files(state.content_api.clone(), Arc::clone(&state.settings), &mut metadata).await {
        Ok(processed_files) => {
            if processed_files.is_empty() {
                debug!("No new files to process");
                return HttpResponse::Ok().json(serde_json::json!({
                    "success": true,
                    "message": "No updates needed"
                }));
            }
            
            debug!("Processing {} new files", processed_files.len());
            
            {
                let mut app_metadata = state.metadata.write().await;
                *app_metadata = metadata.clone();
            }
            
            match GraphService::build_graph_from_metadata(&metadata).await {
                Ok(mut new_graph) => {
                    let mut graph = state.graph_service.graph_data.write().await;
                    
                    let old_positions: HashMap<String, (f32, f32, f32)> = graph.nodes.iter()
                        .map(|node| (node.id.clone(), (node.x(), node.y(), node.z())))
                        .collect();
                    
                    for node in &mut new_graph.nodes {
                        if let Some(&(x, y, z)) = old_positions.get(&node.id) {
                            node.set_x(x);
                            node.set_y(y);
                            node.set_z(z);
                        }
                    }
                    
                    *graph = new_graph;
                    debug!("Graph updated successfully");
                    
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": true,
                        "message": format!("Graph updated with {} new files", processed_files.len())
                    }))
                },
                Err(e) => {
                    error!("Failed to build new graph: {}", e);
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to build new graph: {}", e)
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

// Configure routes using snake_case
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/graph")
            // Match client's endpoint pattern exactly
            .route("/data", web::get().to(get_graph_data))
            .route("/data/paginated", web::get().to(get_paginated_graph_data))
            .route("/update", web::post().to(update_graph))
            // Keep refresh endpoint for admin/maintenance
            .route("/refresh", web::post().to(refresh_graph))
    );
}
