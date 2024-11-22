use crate::config::Settings;
use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use log::error;

pub async fn get_visualization_settings(
    settings: web::Data<Arc<Settings>>,
) -> HttpResponse {
    let settings_json = serde_json::json!({
        "visualization": {
            "nodeColor": settings.visualization.node_color,
            "edgeColor": settings.visualization.edge_color,
            "hologramColor": settings.visualization.hologram_color,
            "hologramScale": settings.visualization.hologram_scale,
            "hologramOpacity": settings.visualization.hologram_opacity,
            "edgeOpacity": settings.visualization.edge_opacity,
            "labelFontSize": settings.visualization.label_font_size,
            "fogDensity": settings.visualization.fog_density,
            "forceDirectedIterations": settings.visualization.force_directed_iterations,
            "forceDirectedRepulsion": settings.visualization.force_directed_repulsion,
            "forceDirectedAttraction": settings.visualization.force_directed_attraction,
        },
        "bloom": {
            "nodeBloomStrength": settings.bloom.node_bloom_strength,
            "nodeBloomRadius": settings.bloom.node_bloom_radius,
            "nodeBloomThreshold": settings.bloom.node_bloom_threshold,
            "edgeBloomStrength": settings.bloom.edge_bloom_strength,
            "edgeBloomRadius": settings.bloom.edge_bloom_radius,
            "edgeBloomThreshold": settings.bloom.edge_bloom_threshold,
            "environmentBloomStrength": settings.bloom.environment_bloom_strength,
            "environmentBloomRadius": settings.bloom.environment_bloom_radius,
            "environmentBloomThreshold": settings.bloom.environment_bloom_threshold,
        },
        "fisheye": {
            "enabled": settings.fisheye.enabled,
            "strength": settings.fisheye.strength,
            "focusPoint": [
                settings.fisheye.focus_x,
                settings.fisheye.focus_y,
                settings.fisheye.focus_z
            ],
            "radius": settings.fisheye.radius,
        }
    });

    HttpResponse::Ok().json(settings_json)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FisheyeUpdateRequest {
    pub enabled: bool,
    pub strength: f32,
    pub focus_point: [f32; 3],
    pub radius: f32,
}

pub async fn update_fisheye_settings(
    gpu_compute: web::Data<Arc<Mutex<crate::utils::gpu_compute::GPUCompute>>>,
    request: web::Json<FisheyeUpdateRequest>,
) -> HttpResponse {
    let mut gpu = match gpu_compute.lock() {
        Ok(gpu) => gpu,
        Err(e) => {
            error!("Failed to acquire GPU compute lock: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to acquire GPU compute lock"
            }));
        }
    };

    // Update GPU compute service with new fisheye settings
    if let Err(e) = gpu.update_fisheye_params(
        request.enabled,
        request.strength,
        request.focus_point,
        request.radius,
    ) {
        error!("Failed to update fisheye settings: {}", e);
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Failed to update fisheye settings: {}", e)
        }));
    }

    // Return success response with updated settings
    HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "settings": {
            "enabled": request.enabled,
            "strength": request.strength,
            "focusPoint": request.focus_point,
            "radius": request.radius
        }
    }))
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/visualization")
            .route("/settings", web::get().to(get_visualization_settings))
            .route("/fisheye", web::post().to(update_fisheye_settings))
    );
}
