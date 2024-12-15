use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs;
use std::path::PathBuf;
use toml;
use log::{error, info};

// GET /api/visualization/settings
pub async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    info!("Attempting to read visualization settings");
    
    let settings_guard = match settings.read().await {
        Ok(guard) => guard,
        Err(e) => {
            error!("Failed to acquire read lock on settings: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to read settings"
            }));
        }
    };

    info!("Successfully acquired settings read lock");
    
    match serde_json::to_value({
        let json = serde_json::json!({
            // Node settings
            "nodeSize": settings_guard.nodes.base_size,
            "nodeColor": settings_guard.nodes.base_color,
            "nodeOpacity": settings_guard.nodes.opacity,
            "metalness": settings_guard.nodes.metalness,
            "roughness": settings_guard.nodes.roughness,
            "clearcoat": settings_guard.nodes.clearcoat,
            "enableInstancing": settings_guard.nodes.enable_instancing,
            "materialType": settings_guard.nodes.material_type,
            "sizeRange": settings_guard.nodes.size_range,
            "sizeByConnections": settings_guard.nodes.size_by_connections,
            "highlightColor": settings_guard.nodes.highlight_color,
            "highlightDuration": settings_guard.nodes.highlight_duration,
            "enableHoverEffect": settings_guard.nodes.enable_hover_effect,
            "hoverScale": settings_guard.nodes.hover_scale,

            // Edge settings
            "edgeWidth": settings_guard.edges.base_width,
            "edgeColor": settings_guard.edges.color,
            "edgeOpacity": settings_guard.edges.opacity,
            "edgeWidthRange": settings_guard.edges.width_range,
            "enableArrows": settings_guard.edges.enable_arrows,
            "arrowSize": settings_guard.edges.arrow_size,

            // Physics settings
            "physicsEnabled": settings_guard.physics.enabled,
            "attractionStrength": settings_guard.physics.attraction_strength,
            "repulsionStrength": settings_guard.physics.repulsion_strength,
            "springStrength": settings_guard.physics.spring_strength,
            "damping": settings_guard.physics.damping,
            "maxVelocity": settings_guard.physics.max_velocity,
            "collisionRadius": settings_guard.physics.collision_radius,
            "boundsSize": settings_guard.physics.bounds_size,
            "enableBounds": settings_guard.physics.enable_bounds,
            "iterations": settings_guard.physics.iterations,

            // Rendering settings
            "ambientLightIntensity": settings_guard.rendering.ambient_light_intensity,
            "directionalLightIntensity": settings_guard.rendering.directional_light_intensity,
            "environmentIntensity": settings_guard.rendering.environment_intensity,
            "enableAmbientOcclusion": settings_guard.rendering.enable_ambient_occlusion,
            "enableAntialiasing": settings_guard.rendering.enable_antialiasing,
            "enableShadows": settings_guard.rendering.enable_shadows,
            "backgroundColor": settings_guard.rendering.background_color,

            // Bloom settings
            "enableBloom": settings_guard.bloom.enabled,
            "bloomIntensity": settings_guard.bloom.strength,
            "bloomRadius": settings_guard.bloom.radius,
            "nodeBloomStrength": settings_guard.bloom.node_bloom_strength,
            "edgeBloomStrength": settings_guard.bloom.edge_bloom_strength,
            "environmentBloomStrength": settings_guard.bloom.environment_bloom_strength,

            // Animation settings
            "enableNodeAnimations": settings_guard.animations.enable_node_animations,
            "enableMotionBlur": settings_guard.animations.enable_motion_blur,
            "motionBlurStrength": settings_guard.animations.motion_blur_strength,
            "selectionWaveEnabled": settings_guard.animations.selection_wave_enabled,
            "pulseEnabled": settings_guard.animations.pulse_enabled,
            "rippleEnabled": settings_guard.animations.ripple_enabled,
            "edgeAnimationEnabled": settings_guard.animations.edge_animation_enabled,
            "flowParticlesEnabled": settings_guard.animations.flow_particles_enabled,

            // Label settings
            "showLabels": settings_guard.labels.enable_labels,
            "labelSize": settings_guard.labels.desktop_font_size as f32 / 48.0,
            "labelColor": settings_guard.labels.text_color,

            // AR settings
            "enablePlaneDetection": settings_guard.ar.enable_plane_detection,
            "enableHandTracking": settings_guard.ar.enable_hand_tracking,
            "enableHaptics": settings_guard.ar.enable_haptics,
            "showPlaneOverlay": settings_guard.ar.show_plane_overlay,
            "planeOpacity": settings_guard.ar.plane_opacity,
            "planeColor": settings_guard.ar.plane_color
        });
        json
    }) {
        Ok(settings_json) => {
            info!("Successfully serialized settings to JSON");
            HttpResponse::Ok().json(settings_json)
        }
        Err(e) => {
            error!("Failed to serialize settings to JSON: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Failed to serialize settings: {}", e)
            }))
        }
    }
}

// PUT /api/visualization/settings
pub async fn update_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<serde_json::Value>,
) -> HttpResponse {
    let mut settings_guard = settings.write().await;
    
    if let Some(obj) = new_settings.as_object() {
        info!("Received settings update: {:?}", obj);
        
        // Update node settings
        if let Some(size) = obj.get("nodeSize").and_then(|v| v.as_f64()) {
            settings_guard.nodes.base_size = size as f32;
        }
        if let Some(color) = obj.get("nodeColor").and_then(|v| v.as_str()) {
            settings_guard.nodes.base_color = color.to_string();
        }
        if let Some(opacity) = obj.get("nodeOpacity").and_then(|v| v.as_f64()) {
            settings_guard.nodes.opacity = opacity as f32;
        }
        if let Some(metalness) = obj.get("metalness").and_then(|v| v.as_f64()) {
            settings_guard.nodes.metalness = metalness as f32;
        }
        if let Some(roughness) = obj.get("roughness").and_then(|v| v.as_f64()) {
            settings_guard.nodes.roughness = roughness as f32;
        }
        if let Some(clearcoat) = obj.get("clearcoat").and_then(|v| v.as_f64()) {
            settings_guard.nodes.clearcoat = clearcoat as f32;
        }
        if let Some(enable_instancing) = obj.get("enableInstancing").and_then(|v| v.as_bool()) {
            settings_guard.nodes.enable_instancing = enable_instancing;
        }
        if let Some(material_type) = obj.get("materialType").and_then(|v| v.as_str()) {
            settings_guard.nodes.material_type = material_type.to_string();
        }
        if let Some(size_range) = obj.get("sizeRange").and_then(|v| v.as_array()) {
            settings_guard.nodes.size_range = size_range.iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect();
        }
        if let Some(size_by_connections) = obj.get("sizeByConnections").and_then(|v| v.as_bool()) {
            settings_guard.nodes.size_by_connections = size_by_connections;
        }
        if let Some(highlight_color) = obj.get("highlightColor").and_then(|v| v.as_str()) {
            settings_guard.nodes.highlight_color = highlight_color.to_string();
        }
        if let Some(highlight_duration) = obj.get("highlightDuration").and_then(|v| v.as_u64()) {
            settings_guard.nodes.highlight_duration = highlight_duration as u32;
        }
        if let Some(enable_hover_effect) = obj.get("enableHoverEffect").and_then(|v| v.as_bool()) {
            settings_guard.nodes.enable_hover_effect = enable_hover_effect;
        }
        if let Some(hover_scale) = obj.get("hoverScale").and_then(|v| v.as_f64()) {
            settings_guard.nodes.hover_scale = hover_scale as f32;
        }

        // Update edge settings
        if let Some(base_width) = obj.get("edgeWidth").and_then(|v| v.as_f64()) {
            settings_guard.edges.base_width = base_width as f32;
        }
        if let Some(color) = obj.get("edgeColor").and_then(|v| v.as_str()) {
            settings_guard.edges.color = color.to_string();
        }
        if let Some(opacity) = obj.get("edgeOpacity").and_then(|v| v.as_f64()) {
            settings_guard.edges.opacity = opacity as f32;
        }
        if let Some(width_range) = obj.get("edgeWidthRange").and_then(|v| v.as_array()) {
            settings_guard.edges.width_range = width_range.iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect();
        }
        if let Some(enable_arrows) = obj.get("enableArrows").and_then(|v| v.as_bool()) {
            settings_guard.edges.enable_arrows = enable_arrows;
        }
        if let Some(arrow_size) = obj.get("arrowSize").and_then(|v| v.as_f64()) {
            settings_guard.edges.arrow_size = arrow_size as f32;
        }

        // Update animation settings
        if let Some(enable_node_animations) = obj.get("enableNodeAnimations").and_then(|v| v.as_bool()) {
            settings_guard.animations.enable_node_animations = enable_node_animations;
        }
        if let Some(enable_motion_blur) = obj.get("enableMotionBlur").and_then(|v| v.as_bool()) {
            settings_guard.animations.enable_motion_blur = enable_motion_blur;
        }
        if let Some(motion_blur_strength) = obj.get("motionBlurStrength").and_then(|v| v.as_f64()) {
            settings_guard.animations.motion_blur_strength = motion_blur_strength as f32;
        }
        if let Some(selection_wave_enabled) = obj.get("selectionWaveEnabled").and_then(|v| v.as_bool()) {
            settings_guard.animations.selection_wave_enabled = selection_wave_enabled;
        }
        if let Some(pulse_enabled) = obj.get("pulseEnabled").and_then(|v| v.as_bool()) {
            settings_guard.animations.pulse_enabled = pulse_enabled;
        }
        if let Some(ripple_enabled) = obj.get("rippleEnabled").and_then(|v| v.as_bool()) {
            settings_guard.animations.ripple_enabled = ripple_enabled;
        }
        if let Some(edge_animation_enabled) = obj.get("edgeAnimationEnabled").and_then(|v| v.as_bool()) {
            settings_guard.animations.edge_animation_enabled = edge_animation_enabled;
        }
        if let Some(flow_particles_enabled) = obj.get("flowParticlesEnabled").and_then(|v| v.as_bool()) {
            settings_guard.animations.flow_particles_enabled = flow_particles_enabled;
        }

        // Save settings to file
        match save_settings_to_file(&settings_guard) {
            Ok(_) => {
                info!("Settings saved successfully");
                HttpResponse::Ok().json(serde_json::json!({
                    "message": "Settings updated successfully"
                }))
            }
            Err(e) => {
                error!("Failed to save settings: {}", e);
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": format!("Failed to save settings: {}", e)
                }))
            }
        }
    } else {
        error!("Invalid settings format received: {:?}", new_settings);
        HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Invalid settings format"
        }))
    }
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

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.route("/settings", web::get().to(get_visualization_settings))
       .route("/settings", web::put().to(update_visualization_settings));
}
