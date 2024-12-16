use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs;
use std::path::PathBuf;
use toml;
use log::{error, info};
use serde::{Deserialize, Serialize};
use crate::utils::case_conversion::{to_snake_case};

// Request/Response structures for individual settings
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeSettingValue<T> {
    pub value: T,
}

// GET /api/visualization/settings - Get all settings
pub async fn get_all_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let settings_guard = settings.read().await;
    HttpResponse::Ok().json(&*settings_guard)
}

// PUT /api/visualization/settings - Update all settings
pub async fn update_all_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<Settings>,
) -> HttpResponse {
    let mut settings_guard = settings.write().await;
    *settings_guard = new_settings.into_inner();
    
    // Save settings to file
    if let Err(e) = save_settings_to_file(&settings_guard) {
        error!("Failed to save settings to file: {}", e);
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": "Failed to save settings to file"
        }));
    }
    
    HttpResponse::Ok().json(&*settings_guard)
}

// GET /api/visualization/nodes/{setting}
pub async fn get_node_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting_name = path.into_inner();
    let settings_guard = settings.read().await;
    
    match setting_name.as_str() {
        "size" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.base_size 
        }),
        "color" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.nodes.base_color 
        }),
        "opacity" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.opacity 
        }),
        "metalness" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.metalness 
        }),
        "roughness" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.roughness 
        }),
        "clearcoat" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.clearcoat 
        }),
        "enableInstancing" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.enable_instancing 
        }),
        "materialType" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.nodes.material_type 
        }),
        "sizeRange" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.nodes.size_range 
        }),
        "sizeByConnections" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.size_by_connections 
        }),
        "highlightColor" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.nodes.highlight_color 
        }),
        "highlightDuration" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.highlight_duration 
        }),
        "enableHoverEffect" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.enable_hover_effect 
        }),
        "hoverScale" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.nodes.hover_scale 
        }),
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown node setting: {}", setting_name)
        }))
    }
}

// PUT /api/visualization/nodes/{setting}
pub async fn update_node_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<NodeSettingValue<serde_json::Value>>,
) -> HttpResponse {
    update_setting(
        &settings,
        "nodes",
        &path.into_inner(),
        value.value.clone(),
    ).await
}

async fn update_setting<T: Serialize>(
    settings: &web::Data<Arc<RwLock<Settings>>>,
    category: &str,
    setting: &str,
    value: T,
) -> HttpResponse {
    let mut settings_guard = settings.write().await;
    let snake_setting = to_snake_case(setting);
    
    if let Err(e) = update_setting_value(&mut settings_guard, category, &snake_setting, &value) {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": format!("Failed to update setting: {}", e)
        }));
    }
    
    // Save settings to file after update
    if let Err(e) = save_settings_to_file(&settings_guard) {
        error!("Failed to save settings to file: {}", e);
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": "Failed to save settings to file"
        }));
    }
    
    HttpResponse::Ok().json(serde_json::json!({ "value": value }))
}

// GET /api/visualization/edges/{setting}
pub async fn get_edge_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting_name = path.into_inner();
    let settings_guard = settings.read().await;
    
    match setting_name.as_str() {
        "width" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.edges.base_width 
        }),
        "color" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.edges.color 
        }),
        "opacity" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.edges.opacity 
        }),
        "widthRange" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.edges.width_range 
        }),
        "enableArrows" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.edges.enable_arrows 
        }),
        "arrowSize" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.edges.arrow_size 
        }),
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown edge setting: {}", setting_name)
        }))
    }
}

// PUT /api/visualization/edges/{setting}
pub async fn update_edge_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<NodeSettingValue<serde_json::Value>>,
) -> HttpResponse {
    update_setting(
        &settings,
        "edges",
        &path.into_inner(),
        value.value.clone(),
    ).await
}

// GET /api/visualization/physics/{setting}
pub async fn get_physics_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting_name = path.into_inner();
    let settings_guard = settings.read().await;
    
    match setting_name.as_str() {
        "enabled" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.enabled 
        }),
        "attractionStrength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.attraction_strength 
        }),
        "repulsionStrength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.repulsion_strength 
        }),
        "springStrength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.spring_strength 
        }),
        "damping" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.damping 
        }),
        "maxVelocity" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.max_velocity 
        }),
        "collisionRadius" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.collision_radius 
        }),
        "boundsSize" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.bounds_size 
        }),
        "enableBounds" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.enable_bounds 
        }),
        "iterations" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.physics.iterations 
        }),
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown physics setting: {}", setting_name)
        }))
    }
}

// PUT /api/visualization/physics/{setting}
pub async fn update_physics_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<NodeSettingValue<serde_json::Value>>,
) -> HttpResponse {
    update_setting(
        &settings,
        "physics",
        &path.into_inner(),
        value.value.clone(),
    ).await
}

// GET /api/visualization/rendering/{setting}
pub async fn get_rendering_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting_name = path.into_inner();
    let settings_guard = settings.read().await;
    
    match setting_name.as_str() {
        "ambientLightIntensity" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.rendering.ambient_light_intensity 
        }),
        "directionalLightIntensity" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.rendering.directional_light_intensity 
        }),
        "environmentIntensity" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.rendering.environment_intensity 
        }),
        "enableAmbientOcclusion" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.rendering.enable_ambient_occlusion 
        }),
        "enableAntialiasing" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.rendering.enable_antialiasing 
        }),
        "enableShadows" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.rendering.enable_shadows 
        }),
        "backgroundColor" => HttpResponse::Ok().json(NodeSettingValue { 
            value: &settings_guard.rendering.background_color 
        }),
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown rendering setting: {}", setting_name)
        }))
    }
}

// PUT /api/visualization/rendering/{setting}
pub async fn update_rendering_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<NodeSettingValue<serde_json::Value>>,
) -> HttpResponse {
    update_setting(
        &settings,
        "rendering",
        &path.into_inner(),
        value.value.clone(),
    ).await
}

// GET /api/visualization/bloom/{setting}
pub async fn get_bloom_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
) -> HttpResponse {
    let setting_name = path.into_inner();
    let settings_guard = settings.read().await;
    
    match setting_name.as_str() {
        "enabled" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.bloom.enabled 
        }),
        "strength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.bloom.strength 
        }),
        "radius" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.bloom.radius 
        }),
        "nodeBloomStrength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.bloom.node_bloom_strength 
        }),
        "edgeBloomStrength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.bloom.edge_bloom_strength 
        }),
        "environmentBloomStrength" => HttpResponse::Ok().json(NodeSettingValue { 
            value: settings_guard.bloom.environment_bloom_strength 
        }),
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown bloom setting: {}", setting_name)
        }))
    }
}

// PUT /api/visualization/bloom/{setting}
pub async fn update_bloom_setting(
    settings: web::Data<Arc<RwLock<Settings>>>,
    path: web::Path<String>,
    value: web::Json<NodeSettingValue<serde_json::Value>>,
) -> HttpResponse {
    update_setting(
        &settings,
        "bloom",
        &path.into_inner(),
        value.value.clone(),
    ).await
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.route("/settings", web::get().to(get_all_settings))
       .route("/settings", web::put().to(update_all_settings))
       .route("/nodes/{setting}", web::get().to(get_node_setting))
       .route("/nodes/{setting}", web::put().to(update_node_setting))
       .route("/edges/{setting}", web::get().to(get_edge_setting))
       .route("/edges/{setting}", web::put().to(update_edge_setting))
       .route("/physics/{setting}", web::get().to(get_physics_setting))
       .route("/physics/{setting}", web::put().to(update_physics_setting))
       .route("/rendering/{setting}", web::get().to(get_rendering_setting))
       .route("/rendering/{setting}", web::put().to(update_rendering_setting))
       .route("/bloom/{setting}", web::get().to(get_bloom_setting))
       .route("/bloom/{setting}", web::put().to(update_bloom_setting));
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    // Convert settings to TOML
    let toml_string = match toml::to_string_pretty(&settings) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to serialize settings to TOML: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };
    
    // Use absolute path from environment or default to /app/settings.toml
    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.toml"));
    
    info!("Attempting to save settings to: {:?}", settings_path);
    
    // Ensure parent directory exists and is writable
    if let Some(parent) = settings_path.parent() {
        match fs::create_dir_all(parent) {
            Ok(_) => info!("Created parent directories: {:?}", parent),
            Err(e) => {
                error!("Failed to create parent directories: {}", e);
                return Err(e);
            }
        }
    }
    
    // Check if file exists and is writable
    if settings_path.exists() {
        match fs::metadata(&settings_path) {
            Ok(metadata) => {
                if metadata.permissions().readonly() {
                    error!("Settings file is read-only: {:?}", settings_path);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Settings file is read-only"
                    ));
                }
            }
            Err(e) => {
                error!("Failed to check settings file permissions: {}", e);
                return Err(e);
            }
        }
    }
    
    // Write to settings.toml
    match fs::write(&settings_path, toml_string) {
        Ok(_) => {
            info!("Settings saved successfully to: {:?}", settings_path);
            Ok(())
        }
        Err(e) => {
            error!("Failed to write settings file: {}", e);
            Err(e)
        }
    }
}

fn update_setting_value<T: Serialize>(
    settings: &mut Settings,
    category: &str,
    setting: &str,
    value: &T,
) -> Result<(), String> {
    match category {
        "nodes" => {
            match setting {
                "base_size" => settings.nodes.base_size = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "base_color" => settings.nodes.base_color = serde_json::to_value(value).unwrap().as_str().unwrap().to_string(),
                "opacity" => settings.nodes.opacity = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "metalness" => settings.nodes.metalness = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "roughness" => settings.nodes.roughness = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "clearcoat" => settings.nodes.clearcoat = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "enable_instancing" => settings.nodes.enable_instancing = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "material_type" => settings.nodes.material_type = serde_json::to_value(value).unwrap().as_str().unwrap().to_string(),
                "size_range" => settings.nodes.size_range = serde_json::to_value(value).unwrap().as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect(),
                "size_by_connections" => settings.nodes.size_by_connections = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "highlight_color" => settings.nodes.highlight_color = serde_json::to_value(value).unwrap().as_str().unwrap().to_string(),
                "highlight_duration" => settings.nodes.highlight_duration = serde_json::to_value(value).unwrap().as_u64().unwrap() as u32,
                "enable_hover_effect" => settings.nodes.enable_hover_effect = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "hover_scale" => settings.nodes.hover_scale = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                _ => return Err(format!("Unknown node setting: {}", setting)),
            }
        },
        "edges" => {
            match setting {
                "base_width" => settings.edges.base_width = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "color" => settings.edges.color = serde_json::to_value(value).unwrap().as_str().unwrap().to_string(),
                "opacity" => settings.edges.opacity = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "width_range" => settings.edges.width_range = serde_json::to_value(value).unwrap().as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect(),
                "enable_arrows" => settings.edges.enable_arrows = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "arrow_size" => settings.edges.arrow_size = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                _ => return Err(format!("Unknown edge setting: {}", setting)),
            }
        },
        "physics" => {
            match setting {
                "enabled" => settings.physics.enabled = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "attraction_strength" => settings.physics.attraction_strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "repulsion_strength" => settings.physics.repulsion_strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "spring_strength" => settings.physics.spring_strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "damping" => settings.physics.damping = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "max_velocity" => settings.physics.max_velocity = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "collision_radius" => settings.physics.collision_radius = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "bounds_size" => settings.physics.bounds_size = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "enable_bounds" => settings.physics.enable_bounds = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "iterations" => settings.physics.iterations = serde_json::to_value(value).unwrap().as_u64().unwrap() as u32,
                _ => return Err(format!("Unknown physics setting: {}", setting)),
            }
        },
        "rendering" => {
            match setting {
                "ambient_light_intensity" => settings.rendering.ambient_light_intensity = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "directional_light_intensity" => settings.rendering.directional_light_intensity = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "environment_intensity" => settings.rendering.environment_intensity = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "enable_ambient_occlusion" => settings.rendering.enable_ambient_occlusion = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "enable_antialiasing" => settings.rendering.enable_antialiasing = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "enable_shadows" => settings.rendering.enable_shadows = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "background_color" => settings.rendering.background_color = serde_json::to_value(value).unwrap().as_str().unwrap().to_string(),
                _ => return Err(format!("Unknown rendering setting: {}", setting)),
            }
        },
        "bloom" => {
            match setting {
                "enabled" => settings.bloom.enabled = serde_json::to_value(value).unwrap().as_bool().unwrap(),
                "strength" => settings.bloom.strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "radius" => settings.bloom.radius = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "node_bloom_strength" => settings.bloom.node_bloom_strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "edge_bloom_strength" => settings.bloom.edge_bloom_strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                "environment_bloom_strength" => settings.bloom.environment_bloom_strength = serde_json::to_value(value).unwrap().as_f64().unwrap() as f32,
                _ => return Err(format!("Unknown bloom setting: {}", setting)),
            }
        },
        _ => return Err(format!("Unknown category: {}", category)),
    }
    Ok(())
}
