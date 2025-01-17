use actix_web::{web, HttpResponse, Result};
use crate::AppState;
use serde_json::json;
use crate::utils::case_conversion::to_camel_case;

pub async fn get_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    
    // Transform settings into client-expected structure matching defaultSettings.ts
    let client_settings = json!({
        "visualization": {
            "animations": {
                "enableMotionBlur": settings.animations.enable_motion_blur,
                "enableNodeAnimations": settings.animations.enable_node_animations,
                "motionBlurStrength": settings.animations.motion_blur_strength,
                "selectionWaveEnabled": settings.animations.selection_wave_enabled,
                "pulseEnabled": settings.animations.pulse_enabled,
                "rippleEnabled": settings.animations.ripple_enabled,
                "edgeAnimationEnabled": settings.animations.edge_animation_enabled,
                "flowParticlesEnabled": settings.animations.flow_particles_enabled,
                "pulseSpeed": settings.animations.pulse_speed,
                "pulseStrength": settings.animations.pulse_strength,
                "waveSpeed": settings.animations.wave_speed
            },
            "bloom": {
                "enabled": settings.bloom.enabled,
                "strength": settings.bloom.strength,
                "radius": settings.bloom.radius,
                "edgeBloomStrength": settings.bloom.edge_bloom_strength,
                "nodeBloomStrength": settings.bloom.node_bloom_strength,
                "environmentBloomStrength": settings.bloom.environment_bloom_strength
            },
            "edges": {
                "arrowSize": settings.edges.arrow_size,
                "baseWidth": settings.edges.base_width,
                "color": settings.edges.color,
                "enableArrows": settings.edges.enable_arrows,
                "opacity": settings.edges.opacity,
                "widthRange": settings.edges.width_range
            },
            "labels": {
                "desktopFontSize": settings.labels.desktop_font_size,
                "enableLabels": settings.labels.enable_labels,
                "textColor": settings.labels.text_color
            },
            "nodes": settings.nodes,
            "physics": settings.physics,
            "rendering": settings.rendering,
            "hologram": {
                "ringCount": settings.hologram.ring_count,
                "ringColor": settings.hologram.ring_color,
                "ringOpacity": settings.hologram.ring_opacity,
                "ringSizes": settings.hologram.ring_sizes,
                "ringRotationSpeed": settings.hologram.ring_rotation_speed,
                "enableBuckminster": settings.hologram.enable_buckminster,
                "buckminsterScale": settings.hologram.buckminster_scale,
                "buckminsterOpacity": settings.hologram.buckminster_opacity,
                "enableGeodesic": settings.hologram.enable_geodesic,
                "geodesicScale": settings.hologram.geodesic_scale,
                "geodesicOpacity": settings.hologram.geodesic_opacity,
                "enableTriangleSphere": settings.hologram.enable_triangle_sphere,
                "triangleSphereScale": settings.hologram.triangle_sphere_scale,
                "triangleSphereOpacity": settings.hologram.triangle_sphere_opacity,
                "globalRotationSpeed": settings.hologram.global_rotation_speed
            }
        },
        "system": {
            "network": {
                "bindAddress": settings.network.bind_address,
                "domain": settings.network.domain,
                "port": settings.network.port,
                "enableHttp2": settings.network.enable_http2,
                "enableTls": settings.network.enable_tls,
                "minTlsVersion": settings.network.min_tls_version,
                "maxRequestSize": settings.network.max_request_size,
                "enableRateLimiting": settings.network.enable_rate_limiting,
                "rateLimitRequests": settings.network.rate_limit_requests,
                "rateLimitWindow": settings.network.rate_limit_window,
                "tunnelId": settings.network.tunnel_id
            },
            "websocket": {
                "binaryChunkSize": settings.websocket.binary_chunk_size,
                "binaryUpdateRate": settings.websocket.binary_update_rate,
                "binaryMessageVersion": settings.websocket.binary_message_version,
                "compressionEnabled": settings.websocket.compression_enabled,
                "compressionThreshold": settings.websocket.compression_threshold,
                "heartbeatInterval": settings.websocket.heartbeat_interval,
                "heartbeatTimeout": settings.websocket.heartbeat_timeout,
                "maxConnections": settings.websocket.max_connections,
                "maxMessageSize": settings.websocket.max_message_size,
                "reconnectAttempts": settings.websocket.reconnect_attempts,
                "reconnectDelay": settings.websocket.reconnect_delay,
                "updateRate": settings.websocket.update_rate
            },
            "security": settings.security,
            "debug": settings.client_debug
        },
        "xr": {
            "mode": "ar",
            "roomScale": settings.ar.room_scale,
            "spaceType": "local",
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
            "passthrough": false,
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