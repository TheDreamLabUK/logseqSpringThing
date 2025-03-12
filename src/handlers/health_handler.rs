use actix_web::{web, HttpResponse, Result, get};
use serde::{Deserialize, Serialize};
use crate::AppState;
use log::info;
use chrono::Utc;

#[derive(Serialize, Deserialize)]
pub struct PhysicsSimulationStatus {
    status: String,
    details: String,
    timestamp: String,
}

pub async fn health_check(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let metadata = app_state.metadata.read().await;
    let graph = app_state.graph_service.get_graph_data_mut().await;
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "metadata_count": metadata.len(),
        "nodes_count": graph.nodes.len(),
        "edges_count": graph.edges.len()
    })))
}

#[get("/physics")]
pub async fn check_physics_simulation(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let current_time = Utc::now();
    
    // Get diagnostic information from the graph service
    let diagnostics = app_state.graph_service.get_simulation_diagnostics().await;
    
    info!("Physics simulation diagnostic check at {}: {}", current_time, diagnostics);
    
    // Determine overall status
    let status = if diagnostics.contains("Is this instance active: true") && 
                  diagnostics.contains("Global running flag: true") {
        "healthy".to_string()
    } else {
        "warning".to_string()  // Not an error, but indicates potential issues
    };
    
    Ok(HttpResponse::Ok().json(PhysicsSimulationStatus {
        status,
        details: diagnostics,
        timestamp: current_time.to_rfc3339(),
    }))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(health_check))
    );
    cfg.service(check_physics_simulation);
}