use actix_web::{web, HttpResponse, Result};
use crate::AppState;
use serde_json::json;
use crate::utils::case_conversion::to_camel_case;

pub async fn get_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    
    // Transform settings into client-expected structure
    let client_settings = json!({
        "visualization": {
            "animations": settings.animations,
            "bloom": settings.bloom,
            "edges": settings.edges,
            "hologram": json!({
                "enabled": false  // Default value since server doesn't have this yet
            }),
            "labels": settings.labels,
            "nodes": settings.nodes,
            "physics": settings.physics,
            "rendering": settings.rendering
        },
        "system": {
            "network": settings.network,
            "websocket": settings.websocket,
            "security": settings.security,
            "debug": settings.client_debug
        },
        "xr": {
            "mode": "ar",
            "roomScale": settings.ar.room_scale,
            "spaceType": "bounded",
            "quality": "high",
            "input": "hands",
            "visuals": {
                "handMeshEnabled": settings.ar.hand_mesh_enabled,
                "handMeshColor": settings.ar.hand_mesh_color,
                "handMeshOpacity": settings.ar.hand_mesh_opacity,
                "handPointSize": settings.ar.hand_point_size,
                "handRayEnabled": settings.ar.hand_ray_enabled,
                "handRayColor": settings.ar.hand_ray_color,
                "handRayWidth": settings.ar.hand_ray_width,
                "gestureSsmoothing": settings.ar.gesture_smoothing
            },
            "environment": {
                "enableLightEstimation": settings.ar.enable_light_estimation,
                "enablePlaneDetection": settings.ar.enable_plane_detection,
                "enableSceneUnderstanding": settings.ar.enable_scene_understanding,
                "planeColor": settings.ar.plane_color,
                "planeOpacity": settings.ar.plane_opacity,
                "showPlaneOverlay": settings.ar.show_plane_overlay,
                "snapToFloor": settings.ar.snap_to_floor
            },
            "passthrough": true,
            "haptics": settings.ar.enable_haptics
        }
    });

    Ok(HttpResponse::Ok().json(client_settings))
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(json!({
        "visualization": {
            "nodes": settings.nodes,
            "edges": settings.edges,
            "physics": settings.physics,
            "labels": settings.labels
        }
    })))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(get_settings))
    ).service(
        web::resource("/graph")
            .route(web::get().to(get_graph_settings))
    );
} 