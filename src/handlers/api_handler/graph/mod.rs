use actix_web::{web, HttpResponse, Responder};
use crate::AppState;
use serde::{Serialize, Deserialize};
use log::{info, debug, error, warn};
use std::collections::HashMap;
use std::sync::Arc;
use crate::models::metadata::{Metadata, MetadataStore};
use crate::models::node::Node; // Changed from socket_flow_messages::Node
use crate::services::file_service::FileService;
// GraphService direct import is no longer needed as we use actors
// use crate::services::graph_service::GraphService;
use crate::actors::messages::{GetGraphData, GetMetadata, GetSettings, BuildGraphFromMetadata};
use crate::models::graph::GraphData as ModelsGraphData; // For message payload

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
    pub page_size: Option<usize>,
    pub sort: Option<String>,
    pub filter: Option<String>,
}

pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    info!("Received request for graph data");
    let graph_data_result = state.graph_service_addr.send(GetGraphData).await;

    match graph_data_result {
        Ok(Ok(graph)) => {
            debug!("Preparing graph response with {} nodes and {} edges",
                graph.nodes.len(),
                graph.edges.len()
            );

            // Assuming GraphResponse expects crate::models::node::Node
            // and crate::models::metadata::MetadataStore (which is HashMap<String, Metadata>)
            let response = GraphResponse {
                nodes: graph.nodes.clone(),
                edges: graph.edges.clone(),
                metadata: graph.metadata.clone(), // This is already HashMap<String, Metadata>
            };
            HttpResponse::Ok().json(response)
        }
        Ok(Err(e)) => {
            error!("Failed to get graph data from actor: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({"error": "Failed to retrieve graph data"}))
        }
        Err(e) => {
            error!("Mailbox error getting graph data: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({"error": "Graph service unavailable"}))
        }
    }
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

    let graph_data_result = state.graph_service_addr.send(GetGraphData).await;

    match graph_data_result {
        Ok(Ok(graph)) => {
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
                .map(|node| node.id) // Assuming node.id is u32
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
        Ok(Err(e)) => {
            error!("Failed to get graph data from actor for pagination: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({"error": "Failed to retrieve graph data"}))
        }
        Err(e) => {
            error!("Mailbox error getting graph data for pagination: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({"error": "Graph service unavailable"}))
        }
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

    // This part is complex due to mutable access.
    // For now, let's assume get_graph_data_mut was for reading and we use GetGraphData.
    // If mutable access is truly needed, specific messages for modifications are required.
    let graph_result = state.graph_service_addr.send(GetGraphData).await;
    let graph = match graph_result {
        Ok(Ok(g)) => g,
        _ => {
            error!("Failed to get graph data for pagination");
            return HttpResponse::InternalServerError().json(serde_json::json!({"error": "Failed to retrieve graph data"}));
        }
    };
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
        .map(|node| node.id) // Assuming node.id is u32
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
    
    let metadata_result = state.metadata_addr.send(GetMetadata).await;
    
    match metadata_result {
        Ok(Ok(metadata_store)) => {
            debug!("Building graph from {} metadata entries", metadata_store.len());
            
            // Send BuildGraphFromMetadata message to GraphServiceActor
            match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store }).await {
                Ok(Ok(())) => {
                    // Optionally, if we need to preserve old positions, that logic would need to be
                    // part of the GraphServiceActor's BuildGraphFromMetadata handler or a subsequent message.
                    // For simplicity here, we assume the actor handles the build correctly.
                    info!("Graph refreshed successfully via GraphServiceActor");
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": true,
                        "message": "Graph refreshed successfully"
                    }))
                }
                Ok(Err(e)) => {
                    error!("GraphServiceActor failed to build graph from metadata: {}", e);
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to build graph: {}", e)
                    }))
                }
                Err(e) => {
                    error!("Mailbox error sending BuildGraphFromMetadata: {}", e);
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": "Graph service unavailable during refresh"
                    }))
                }
            }
        }
        Ok(Err(e)) => {
            error!("Failed to get metadata from MetadataActor: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Failed to retrieve metadata for graph refresh"
            }))
        }
        Err(e) => {
            error!("Mailbox error getting metadata: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Metadata service unavailable"
            }))
        }
// Removed duplicate success block from here
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
    
    let settings_result = state.settings_addr.send(GetSettings).await;
    let settings = match settings_result {
        Ok(Ok(s)) => Arc::new(s),
        _ => {
            error!("Failed to retrieve settings for FileService in update_graph");
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Failed to retrieve application settings"
            }));
        }
    };
    
    let file_service = FileService::new(settings.clone());
    match file_service.fetch_and_process_files(state.content_api.clone(), settings.clone(), &mut metadata).await {
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
                // Send UpdateMetadata message to MetadataActor
                if let Err(e) = state.metadata_addr.send(crate::actors::messages::UpdateMetadata { metadata: metadata.clone() }).await {
                     error!("Failed to send UpdateMetadata to MetadataActor: {}", e);
                     // Potentially return error if this is critical
                }
            }
            
            // Send BuildGraphFromMetadata message to GraphServiceActor
            match state.graph_service_addr.send(BuildGraphFromMetadata { metadata }).await {
                Ok(Ok(())) => {
                    // Position preservation logic would need to be handled by the actor or subsequent messages.
                    debug!("Graph updated successfully via GraphServiceActor after file processing");
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
