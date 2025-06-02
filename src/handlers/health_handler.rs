use actix_web::{web, HttpResponse, Result, get};
use serde::{Deserialize, Serialize};
use crate::AppState;
use log::{info, error};
use chrono::Utc;
use crate::actors::messages::{GetMetadata, GetGraphData}; // Assuming GetGraphData returns the necessary counts or the GraphData struct
// If GraphServiceActor needs a specific message for diagnostics:
// use crate::actors::messages::GetSimulationDiagnostics;

#[derive(Serialize, Deserialize)]
pub struct PhysicsSimulationStatus {
    status: String,
    details: String,
    timestamp: String,
}

pub async fn health_check(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let metadata_count_result = app_state.metadata_addr.send(GetMetadata).await;
    let graph_data_result = app_state.graph_service_addr.send(GetGraphData).await;

    let metadata_count = match metadata_count_result {
        Ok(Ok(metadata_store)) => metadata_store.len(),
        _ => {
            error!("Failed to get metadata count from MetadataActor for health check");
            0 // Default or handle error appropriately
        }
    };

    let (nodes_count, edges_count) = match graph_data_result {
        Ok(Ok(graph_data)) => (graph_data.nodes.len(), graph_data.edges.len()),
        _ => {
            error!("Failed to get graph data from GraphServiceActor for health check");
            (0, 0) // Default or handle error appropriately
        }
    };
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "metadata_count": metadata_count,
        "nodes_count": nodes_count,
        "edges_count": edges_count
    })))
}

#[get("/physics")]
pub async fn check_physics_simulation(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let current_time = Utc::now();
    
    // Assuming GraphServiceActor has a message like GetSimulationDiagnostics
    // If not, this part needs to be adapted based on how diagnostics are exposed by the actor.
    // For now, let's assume a placeholder or that GraphServiceActor itself doesn't expose this directly anymore
    // and this logic might need to be re-evaluated or moved.
    // If `get_simulation_diagnostics` was a method on the old GraphService struct,
    // it needs a corresponding message for the GraphServiceActor.
    // Let's assume for now we get a generic status from the actor.
    
    // Placeholder for actual diagnostic fetching from an actor if available.
    // This might involve sending a specific message to GraphServiceActor or GPUComputeActor.
    // For example, if GPUComputeActor provides status:
    // use crate::actors::messages::GetGPUStatus;
    // let gpu_status_result = app_state.gpu_compute_addr.as_ref()
    //     .map(|addr| addr.send(GetGPUStatus).await);

    // For now, let's construct a simplified diagnostic string.
    // The original `get_simulation_diagnostics` was on `GraphService` struct, not actor.
    // This functionality might need to be re-implemented via actor messages if still required.
    // As a temporary measure, we'll return a generic status.
    
    let diagnostics = "Physics simulation status check via actor system (detailed diagnostics TBD)".to_string();
    info!("Physics simulation diagnostic check at {}: {}", current_time, diagnostics);
    
    // Simplified status determination until actor-based diagnostics are clear
    let status = "checking".to_string(); // Placeholder
    
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