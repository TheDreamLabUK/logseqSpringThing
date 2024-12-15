use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs;
use std::path::PathBuf;
use toml;
use log::{error, info};
use serde::{Deserialize, Serialize};

// Request/Response structures for individual settings
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeSettingValue<T> {
    pub value: T,
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
    let setting_name = path.into_inner();
    let mut settings_guard = settings.write().await;
    
    match setting_name.as_str() {
        "size" => {
            if let Some(size) = value.value.as_f64() {
                settings_guard.nodes.base_size = size as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: size })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node size"
                }))
            }
        },
        "color" => {
            if let Some(color) = value.value.as_str() {
                settings_guard.nodes.base_color = color.to_string();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.nodes.base_color 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node color"
                }))
            }
        },
        "opacity" => {
            if let Some(opacity) = value.value.as_f64() {
                settings_guard.nodes.opacity = opacity as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: opacity })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node opacity"
                }))
            }
        },
        "metalness" => {
            if let Some(metalness) = value.value.as_f64() {
                settings_guard.nodes.metalness = metalness as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: metalness })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node metalness"
                }))
            }
        },
        "roughness" => {
            if let Some(roughness) = value.value.as_f64() {
                settings_guard.nodes.roughness = roughness as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: roughness })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node roughness"
                }))
            }
        },
        "clearcoat" => {
            if let Some(clearcoat) = value.value.as_f64() {
                settings_guard.nodes.clearcoat = clearcoat as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: clearcoat })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node clearcoat"
                }))
            }
        },
        "enableInstancing" => {
            if let Some(enable_instancing) = value.value.as_bool() {
                settings_guard.nodes.enable_instancing = enable_instancing;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_instancing })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node enableInstancing"
                }))
            }
        },
        "materialType" => {
            if let Some(material_type) = value.value.as_str() {
                settings_guard.nodes.material_type = material_type.to_string();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.nodes.material_type 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node materialType"
                }))
            }
        },
        "sizeRange" => {
            if let Some(size_range) = value.value.as_array() {
                settings_guard.nodes.size_range = size_range.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.nodes.size_range 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node sizeRange"
                }))
            }
        },
        "sizeByConnections" => {
            if let Some(size_by_connections) = value.value.as_bool() {
                settings_guard.nodes.size_by_connections = size_by_connections;
                HttpResponse::Ok().json(NodeSettingValue { value: size_by_connections })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node sizeByConnections"
                }))
            }
        },
        "highlightColor" => {
            if let Some(highlight_color) = value.value.as_str() {
                settings_guard.nodes.highlight_color = highlight_color.to_string();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.nodes.highlight_color 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node highlightColor"
                }))
            }
        },
        "highlightDuration" => {
            if let Some(highlight_duration) = value.value.as_u64() {
                settings_guard.nodes.highlight_duration = highlight_duration as u32;
                HttpResponse::Ok().json(NodeSettingValue { value: highlight_duration })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node highlightDuration"
                }))
            }
        },
        "enableHoverEffect" => {
            if let Some(enable_hover_effect) = value.value.as_bool() {
                settings_guard.nodes.enable_hover_effect = enable_hover_effect;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_hover_effect })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node enableHoverEffect"
                }))
            }
        },
        "hoverScale" => {
            if let Some(hover_scale) = value.value.as_f64() {
                settings_guard.nodes.hover_scale = hover_scale as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: hover_scale })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for node hoverScale"
                }))
            }
        },
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown node setting: {}", setting_name)
        }))
    }
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
    let setting_name = path.into_inner();
    let mut settings_guard = settings.write().await;
    
    match setting_name.as_str() {
        "width" => {
            if let Some(width) = value.value.as_f64() {
                settings_guard.edges.base_width = width as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: width })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for edge width"
                }))
            }
        },
        "color" => {
            if let Some(color) = value.value.as_str() {
                settings_guard.edges.color = color.to_string();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.edges.color 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for edge color"
                }))
            }
        },
        "opacity" => {
            if let Some(opacity) = value.value.as_f64() {
                settings_guard.edges.opacity = opacity as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: opacity })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for edge opacity"
                }))
            }
        },
        "widthRange" => {
            if let Some(width_range) = value.value.as_array() {
                settings_guard.edges.width_range = width_range.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.edges.width_range 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for edge widthRange"
                }))
            }
        },
        "enableArrows" => {
            if let Some(enable_arrows) = value.value.as_bool() {
                settings_guard.edges.enable_arrows = enable_arrows;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_arrows })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for edge enableArrows"
                }))
            }
        },
        "arrowSize" => {
            if let Some(arrow_size) = value.value.as_f64() {
                settings_guard.edges.arrow_size = arrow_size as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: arrow_size })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for edge arrowSize"
                }))
            }
        },
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown edge setting: {}", setting_name)
        }))
    }
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
    let setting_name = path.into_inner();
    let mut settings_guard = settings.write().await;
    
    match setting_name.as_str() {
        "enabled" => {
            if let Some(enabled) = value.value.as_bool() {
                settings_guard.physics.enabled = enabled;
                HttpResponse::Ok().json(NodeSettingValue { value: enabled })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics enabled"
                }))
            }
        },
        "attractionStrength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.physics.attraction_strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics attractionStrength"
                }))
            }
        },
        "repulsionStrength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.physics.repulsion_strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics repulsionStrength"
                }))
            }
        },
        "springStrength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.physics.spring_strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics springStrength"
                }))
            }
        },
        "damping" => {
            if let Some(damping) = value.value.as_f64() {
                settings_guard.physics.damping = damping as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: damping })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics damping"
                }))
            }
        },
        "maxVelocity" => {
            if let Some(velocity) = value.value.as_f64() {
                settings_guard.physics.max_velocity = velocity as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: velocity })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics maxVelocity"
                }))
            }
        },
        "collisionRadius" => {
            if let Some(radius) = value.value.as_f64() {
                settings_guard.physics.collision_radius = radius as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: radius })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics collisionRadius"
                }))
            }
        },
        "boundsSize" => {
            if let Some(size) = value.value.as_f64() {
                settings_guard.physics.bounds_size = size as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: size })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics boundsSize"
                }))
            }
        },
        "enableBounds" => {
            if let Some(enable_bounds) = value.value.as_bool() {
                settings_guard.physics.enable_bounds = enable_bounds;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_bounds })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics enableBounds"
                }))
            }
        },
        "iterations" => {
            if let Some(iterations) = value.value.as_u64() {
                settings_guard.physics.iterations = iterations as u32;
                HttpResponse::Ok().json(NodeSettingValue { value: iterations })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for physics iterations"
                }))
            }
        },
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown physics setting: {}", setting_name)
        }))
    }
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
    let setting_name = path.into_inner();
    let mut settings_guard = settings.write().await;
    
    match setting_name.as_str() {
        "ambientLightIntensity" => {
            if let Some(intensity) = value.value.as_f64() {
                settings_guard.rendering.ambient_light_intensity = intensity as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: intensity })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering ambientLightIntensity"
                }))
            }
        },
        "directionalLightIntensity" => {
            if let Some(intensity) = value.value.as_f64() {
                settings_guard.rendering.directional_light_intensity = intensity as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: intensity })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering directionalLightIntensity"
                }))
            }
        },
        "environmentIntensity" => {
            if let Some(intensity) = value.value.as_f64() {
                settings_guard.rendering.environment_intensity = intensity as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: intensity })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering environmentIntensity"
                }))
            }
        },
        "enableAmbientOcclusion" => {
            if let Some(enable_ambient_occlusion) = value.value.as_bool() {
                settings_guard.rendering.enable_ambient_occlusion = enable_ambient_occlusion;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_ambient_occlusion })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering enableAmbientOcclusion"
                }))
            }
        },
        "enableAntialiasing" => {
            if let Some(enable_antialiasing) = value.value.as_bool() {
                settings_guard.rendering.enable_antialiasing = enable_antialiasing;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_antialiasing })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering enableAntialiasing"
                }))
            }
        },
        "enableShadows" => {
            if let Some(enable_shadows) = value.value.as_bool() {
                settings_guard.rendering.enable_shadows = enable_shadows;
                HttpResponse::Ok().json(NodeSettingValue { value: enable_shadows })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering enableShadows"
                }))
            }
        },
        "backgroundColor" => {
            if let Some(color) = value.value.as_str() {
                settings_guard.rendering.background_color = color.to_string();
                HttpResponse::Ok().json(NodeSettingValue { 
                    value: &settings_guard.rendering.background_color 
                })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for rendering backgroundColor"
                }))
            }
        },
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown rendering setting: {}", setting_name)
        }))
    }
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
    let setting_name = path.into_inner();
    let mut settings_guard = settings.write().await;
    
    match setting_name.as_str() {
        "enabled" => {
            if let Some(enabled) = value.value.as_bool() {
                settings_guard.bloom.enabled = enabled;
                HttpResponse::Ok().json(NodeSettingValue { value: enabled })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for bloom enabled"
                }))
            }
        },
        "strength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.bloom.strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for bloom strength"
                }))
            }
        },
        "radius" => {
            if let Some(radius) = value.value.as_f64() {
                settings_guard.bloom.radius = radius as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: radius })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for bloom radius"
                }))
            }
        },
        "nodeBloomStrength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.bloom.node_bloom_strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for bloom nodeBloomStrength"
                }))
            }
        },
        "edgeBloomStrength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.bloom.edge_bloom_strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for bloom edgeBloomStrength"
                }))
            }
        },
        "environmentBloomStrength" => {
            if let Some(strength) = value.value.as_f64() {
                settings_guard.bloom.environment_bloom_strength = strength as f32;
                HttpResponse::Ok().json(NodeSettingValue { value: strength })
            } else {
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid value type for bloom environmentBloomStrength"
                }))
            }
        },
        _ => HttpResponse::NotFound().json(serde_json::json!({
            "error": format!("Unknown bloom setting: {}", setting_name)
        }))
    }
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.route("/nodes/{setting}", web::get().to(get_node_setting))
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
