use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use crate::state::AppState;
use crate::models::graph::{Node, Edge};

#[derive(Debug, Serialize)]
pub struct GraphData {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Debug, Deserialize)]
pub struct GraphUpdateRequest {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Debug, Serialize)]
pub struct PaginatedGraphData {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    total_nodes: usize,
    total_edges: usize,
    page: usize,
    page_size: usize,
}

pub async fn get_graph_data(state: web::Data<AppState>) -> HttpResponse {
    let graph = state.graph.read().await;
    
    let data = GraphData {
        nodes: graph.get_nodes(),
        edges: graph.get_edges(),
    };

    HttpResponse::Ok().json(data)
}

pub async fn get_paginated_data(
    query: web::Query<PaginationParams>,
    state: web::Data<AppState>,
) -> HttpResponse {
    let graph = state.graph.read().await;
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(100);

    let start = (page - 1) * page_size;
    let nodes = graph.get_nodes_paginated(start, page_size);
    let edges = graph.get_edges_paginated(start, page_size);

    let data = PaginatedGraphData {
        nodes,
        edges,
        total_nodes: graph.node_count(),
        total_edges: graph.edge_count(),
        page,
        page_size,
    };

    HttpResponse::Ok().json(data)
}

pub async fn update_graph(
    update: web::Json<GraphUpdateRequest>,
    state: web::Data<AppState>,
) -> HttpResponse {
    let mut graph = state.graph.write().await;
    
    match graph.update_graph(update.nodes.clone(), update.edges.clone()).await {
        Ok(_) => HttpResponse::Ok().finish(),
        Err(e) => {
            log::error!("Failed to update graph: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to update graph: {}", e)
            }))
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    page: Option<usize>,
    page_size: Option<usize>,
} 