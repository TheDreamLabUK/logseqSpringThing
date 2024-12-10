use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;

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
        }
    });

    HttpResponse::Ok().json(settings_json)
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/visualization")
            .route("/settings", web::get().to(get_visualization_settings))
    );
}
