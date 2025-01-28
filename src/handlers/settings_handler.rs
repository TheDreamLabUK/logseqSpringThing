use crate::app_state::AppState;
use crate::utils::case_conversion::to_camel_case;
use actix_web::{web, Error, HttpResponse};
use serde_json::{json, Value};

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/settings")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings)),
    );
}

async fn get_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = state.settings.read().await;
    let settings_json = convert_struct_to_camel_case(&*settings_guard);
    Ok(HttpResponse::Ok().json(settings_json))
}

async fn update_settings(
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    if let Err(e) = validate_settings(&payload) {
        return Ok(HttpResponse::BadRequest().body(format!("Invalid settings: {}", e)));
    }

    let mut settings_guard = state.settings.write().await;

    if let Err(e) = settings_guard.merge(payload.into_inner()) {
        return Ok(HttpResponse::BadRequest().body(format!("Failed to merge settings: {}", e)));
    }

    if let Err(e) = settings_guard.save() {
        return Ok(
            HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e))
        );
    }

    let settings_json = convert_struct_to_camel_case(&*settings_guard);
    Ok(HttpResponse::Ok().json(settings_json))
}

fn validate_settings(settings: &Value) -> Result<(), String> {
    if !settings.is_object() {
        return Err("Settings must be an object".to_string());
    }

    let obj = settings.as_object().unwrap();

    // Validate visualization settings
    if let Some(visualization) = obj.get("visualization") {
        validate_visualization_settings(visualization)?;
    }

    // Validate system settings
    if let Some(system) = obj.get("system") {
        validate_system_settings(system)?;
    }

    // Validate XR settings
    if let Some(xr) = obj.get("xr") {
        validate_xr_settings(xr)?;
    }

    Ok(())
}

fn validate_visualization_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("Visualization settings must be an object")?;

    // Validate bloom settings
    if let Some(bloom) = obj.get("bloom") {
        validate_bloom_settings(bloom)?;
    }

    // Validate hologram settings
    if let Some(hologram) = obj.get("hologram") {
        validate_hologram_settings(hologram)?;
    }

    // Validate physics settings
    if let Some(physics) = obj.get("physics") {
        validate_physics_settings(physics)?;
    }

    // Validate node settings
    if let Some(nodes) = obj.get("nodes") {
        validate_node_settings(nodes)?;
    }

    Ok(())
}

fn validate_bloom_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("Bloom settings must be an object")?;

    // If bloom is enabled, validate required settings
    if let Some(enabled) = obj.get("enabled") {
        if enabled.as_bool().unwrap_or(false) {
            // Validate strength (0.0 - 2.0)
            if let Some(strength) = obj.get("strength") {
                let strength_val = strength.as_f64().ok_or("Bloom strength must be a number")?;
                if !(0.0..=2.0).contains(&strength_val) {
                    return Err("Bloom strength must be between 0.0 and 2.0".to_string());
                }
            }

            // Validate radius (0.0 - 1.0)
            if let Some(radius) = obj.get("radius") {
                let radius_val = radius.as_f64().ok_or("Bloom radius must be a number")?;
                if !(0.0..=1.0).contains(&radius_val) {
                    return Err("Bloom radius must be between 0.0 and 1.0".to_string());
                }
            }
        }
    }

    Ok(())
}

fn validate_hologram_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("Hologram settings must be an object")?;

    // If hologram features are enabled, validate their settings
    if let Some(enable_buckminster) = obj.get("enableBuckminster") {
        if enable_buckminster.as_bool().unwrap_or(false) {
            validate_range_f64(obj, "buckminsterScale", 0.1, 50.0, "Buckminster scale")?;
            validate_range_f64(obj, "buckminsterOpacity", 0.0, 1.0, "Buckminster opacity")?;
        }
    }

    if let Some(enable_geodesic) = obj.get("enableGeodesic") {
        if enable_geodesic.as_bool().unwrap_or(false) {
            validate_range_f64(obj, "geodesicScale", 0.1, 50.0, "Geodesic scale")?;
            validate_range_f64(obj, "geodesicOpacity", 0.0, 1.0, "Geodesic opacity")?;
        }
    }

    Ok(())
}

fn validate_physics_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("Physics settings must be an object")?;

    if let Some(enabled) = obj.get("enabled") {
        if enabled.as_bool().unwrap_or(false) {
            // Validate required physics parameters when enabled
            validate_range_f64(obj, "attractionStrength", 0.0, 1.0, "Attraction strength")?;
            validate_range_f64(obj, "repulsionStrength", 0.0, 5000.0, "Repulsion strength")?;
            validate_range_f64(obj, "springStrength", 0.0, 1.0, "Spring strength")?;
            validate_range_f64(obj, "damping", 0.0, 1.0, "Damping")?;

            if let Some(iterations) = obj.get("iterations") {
                let iter_val = iterations
                    .as_u64()
                    .ok_or("Iterations must be a positive integer")?;
                if !(100..=1000).contains(&iter_val) {
                    return Err("Iterations must be between 100 and 1000".to_string());
                }
            }
        }
    }

    Ok(())
}

fn validate_node_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("Node settings must be an object")?;

    // Validate quality enum
    if let Some(quality) = obj.get("quality") {
        let quality_str = quality.as_str().ok_or("Quality must be a string")?;
        if !["low", "medium", "high"].contains(&quality_str) {
            return Err("Quality must be one of: low, medium, high".to_string());
        }
    }

    // Validate numeric ranges
    validate_range_f64(obj, "baseSize", 0.1, 10.0, "Base size")?;
    validate_range_f64(obj, "opacity", 0.0, 1.0, "Opacity")?;
    validate_range_f64(obj, "metalness", 0.0, 1.0, "Metalness")?;
    validate_range_f64(obj, "roughness", 0.0, 1.0, "Roughness")?;

    Ok(())
}

fn validate_system_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("System settings must be an object")?;

    // Validate network settings
    if let Some(network) = obj.get("network") {
        validate_network_settings(network)?;
    }

    // Validate websocket settings
    if let Some(websocket) = obj.get("websocket") {
        validate_websocket_settings(websocket)?;
    }

    Ok(())
}

fn validate_network_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("Network settings must be an object")?;

    // Validate port range
    if let Some(port) = obj.get("port") {
        let port_val = port.as_u64().ok_or("Port must be a positive integer")?;
        if !(1..=65535).contains(&port_val) {
            return Err("Port must be between 1 and 65535".to_string());
        }
    }

    // Validate rate limiting settings
    if let Some(enable_rate_limiting) = obj.get("enableRateLimiting") {
        if enable_rate_limiting.as_bool().unwrap_or(false) {
            if let Some(rate_limit_requests) = obj.get("rateLimitRequests") {
                let requests = rate_limit_requests
                    .as_u64()
                    .ok_or("Rate limit requests must be a positive integer")?;
                if requests == 0 {
                    return Err("Rate limit requests must be greater than 0".to_string());
                }
            }
        }
    }

    Ok(())
}

fn validate_websocket_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("WebSocket settings must be an object")?;

    // Validate update rate
    if let Some(update_rate) = obj.get("updateRate") {
        let rate = update_rate
            .as_u64()
            .ok_or("Update rate must be a positive integer")?;
        if !(1..=120).contains(&rate) {
            return Err("Update rate must be between 1 and 120".to_string());
        }
    }

    // Validate message size
    if let Some(max_message_size) = obj.get("maxMessageSize") {
        let size = max_message_size
            .as_u64()
            .ok_or("Max message size must be a positive integer")?;
        if size > 100 * 1024 * 1024 {
            // 100MB limit
            return Err("Max message size cannot exceed 100MB".to_string());
        }
    }

    Ok(())
}

fn validate_xr_settings(settings: &Value) -> Result<(), String> {
    let obj = settings
        .as_object()
        .ok_or("XR settings must be an object")?;

    // Validate mode enum
    if let Some(mode) = obj.get("mode") {
        let mode_str = mode.as_str().ok_or("XR mode must be a string")?;
        if !["immersive-ar", "immersive-vr"].contains(&mode_str) {
            return Err("XR mode must be one of: immersive-ar, immersive-vr".to_string());
        }
    }

    // Validate space type enum
    if let Some(space_type) = obj.get("spaceType") {
        let space_str = space_type.as_str().ok_or("Space type must be a string")?;
        if ![
            "viewer",
            "local",
            "local-floor",
            "bounded-floor",
            "unbounded",
        ]
        .contains(&space_str)
        {
            return Err("Invalid space type".to_string());
        }
    }

    // Validate numeric ranges
    validate_range_f64(obj, "handMeshOpacity", 0.0, 1.0, "Hand mesh opacity")?;
    validate_range_f64(obj, "handPointSize", 0.1, 20.0, "Hand point size")?;
    validate_range_f64(obj, "handRayWidth", 0.1, 10.0, "Hand ray width")?;
    validate_range_f64(obj, "hapticIntensity", 0.0, 1.0, "Haptic intensity")?;

    Ok(())
}

fn validate_range_f64(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    min: f64,
    max: f64,
    name: &str,
) -> Result<(), String> {
    if let Some(value) = obj.get(key) {
        let val = value.as_f64().ok_or(format!("{} must be a number", name))?;
        if !(min..=max).contains(&val) {
            return Err(format!("{} must be between {} and {}", name, min, max));
        }
    }
    Ok(())
}

fn convert_struct_to_camel_case<T: serde::Serialize>(value: &T) -> serde_json::Value {
    let json_value = serde_json::to_value(value).unwrap_or_default();

    match json_value {
        serde_json::Value::Object(map) => {
            let converted: serde_json::Map<String, serde_json::Value> = map
                .into_iter()
                .map(|(k, v)| (to_camel_case(&k), convert_struct_to_camel_case_value(&v)))
                .collect();
            serde_json::Value::Object(converted)
        }
        _ => json_value,
    }
}

fn convert_struct_to_camel_case_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let converted: serde_json::Map<String, serde_json::Value> = map
                .into_iter()
                .map(|(k, v)| (to_camel_case(&k), convert_struct_to_camel_case_value(v)))
                .collect();
            serde_json::Value::Object(converted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(convert_struct_to_camel_case_value).collect())
        }
        _ => value.clone(),
    }
}

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(json!({
        "visualization": {
            "nodes": settings.visualization.nodes,
            "edges": settings.visualization.edges,
            "physics": settings.visualization.physics,
            "labels": settings.visualization.labels
        }
    })))
}
