use actix_web::{web, HttpResponse, Responder};
use crate::AppState;
use serde::Serialize;
use log::{info, debug, error};
use std::collections::HashMap;
use crate::models::metadata::Metadata;
use crate::utils::socket_flow_messages::Node;
use crate::models::pagination::PaginationParams;

/// Struct to serialize GraphData for HTTP responses.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphResponse {
    /// List of nodes in the graph.
    pub nodes: Vec<Node>,
    /// List of edges connecting the nodes.
    pub edges: Vec<crate::models::edge::Edge>,
    /// Additional metadata about the graph.
    pub metadata: HashMap<String, Metadata>,
}

/// Handler to retrieve the current graph data.
///
/// This function performs the following steps:
/// 1. Reads the shared graph data from the application state.
/// 2. Serializes the graph data into JSON format.
/// 3. Returns the serialized graph data as an HTTP response.
///
/// # Arguments
///
/// * `state` - Shared application state.
///
/// # Returns
///
/// An HTTP response containing the graph data or an error.
pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    info!("Received request for graph data");

    // Step 1: Acquire read access to the shared graph data.
    let graph = state.graph_service.graph_data.read().await;

    debug!("Preparing graph response with {} nodes and {} edges",
        graph.nodes.len(),
        graph.edges.len()
    );

    // Step 2: Prepare the response struct.
    let response = GraphResponse {
        nodes: graph.nodes.clone(),
        edges: graph.edges.clone(),
        metadata: graph.metadata.clone(),
    };

    // Step 3: Respond with the serialized graph data.
    HttpResponse::Ok().json(response)
}

/// Handler to retrieve paginated graph data.
///
/// This function performs the following steps:
/// 1. Extracts pagination parameters from the query
/// 2. Retrieves a page of graph data from the service
/// 3. Returns the paginated data as an HTTP response
///
/// # Arguments
///
/// * `state` - Shared application state
/// * `query` - Query parameters containing page and page_size
///
/// # Returns
///
/// An HTTP response containing the paginated graph data or an error.
pub async fn get_paginated_graph_data(
    state: web::Data<AppState>,
    query: web::Query<PaginationParams>,
) -> impl Responder {
    debug!("Received request for paginated graph data with params: {:?}", query);

    let page = query.page.unwrap_or(0);
    let page_size = query.page_size.unwrap_or(100);

    match state.graph_service.get_paginated_graph_data(page, page_size).await {
        Ok(data) => {
            debug!("Returning page {} with {} nodes and {} edges",
                data.current_page,
                data.nodes.len(),
                data.edges.len()
            );
            HttpResponse::Ok().json(data)
        },
        Err(e) => {
            error!("Failed to get paginated graph data: {}", e);
            HttpResponse::InternalServerError().finish()
        }
    }
}
