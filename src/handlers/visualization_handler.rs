use crate::config::Settings;
use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use toml;

// GET /api/visualization/settings
pub async fn get_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
) -> HttpResponse {
    let settings_guard = settings.read().await;
    
    // Break down JSON construction into parts
    let node_settings = serde_json::json!({
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
        "enableHoverEffect": settings_guard.nodes.enable_hover_effect
    });

    let edge_settings = serde_json::json!({
        "edgeWidth": settings_guard.edges.base_width,
        "edgeColor": settings_guard.edges.color,
        "edgeOpacity": settings_guard.edges.opacity,
        "edgeWidthRange": settings_guard.edges.width_range
    });

    let physics_settings = serde_json::json!({
        "physicsEnabled": settings_guard.physics.enabled,
        "attractionStrength": settings_guard.physics.attraction_strength,
        "repulsionStrength": settings_guard.physics.repulsion_strength,
        "springStrength": settings_guard.physics.spring_strength,
        "damping": settings_guard.physics.damping,
        "maxVelocity": settings_guard.physics.max_velocity,
        "collisionRadius": settings_guard.physics.collision_radius,
        "boundsSize": settings_guard.physics.bounds_size,
        "enableBounds": settings_guard.physics.enable_bounds,
        "iterations": settings_guard.physics.iterations
    });

    let rendering_settings = serde_json::json!({
        "ambientLightIntensity": settings_guard.rendering.ambient_light_intensity,
        "directionalLightIntensity": settings_guard.rendering.directional_light_intensity,
        "environmentIntensity": settings_guard.rendering.environment_intensity,
        "enableAmbientOcclusion": settings_guard.rendering.enable_ambient_occlusion,
        "enableAntialiasing": settings_guard.rendering.enable_antialiasing,
        "enableShadows": settings_guard.rendering.enable_shadows,
        "backgroundColor": settings_guard.rendering.background_color
    });

    let bloom_settings = serde_json::json!({
        "bloomEnabled": settings_guard.bloom.enabled,
        "nodeBloomStrength": settings_guard.bloom.node_bloom_strength,
        "edgeBloomStrength": settings_guard.bloom.edge_bloom_strength,
        "environmentBloomStrength": settings_guard.bloom.environment_bloom_strength
    });

    let animation_settings = serde_json::json!({
        "enableNodeAnimations": settings_guard.animations.enable_node_animations,
        "selectionWaveEnabled": settings_guard.animations.selection_wave_enabled,
        "pulseEnabled": settings_guard.animations.pulse_enabled,
        "rippleEnabled": settings_guard.animations.ripple_enabled,
        "edgeAnimationEnabled": settings_guard.animations.edge_animation_enabled,
        "flowParticlesEnabled": settings_guard.animations.flow_particles_enabled
    });

    let label_settings = serde_json::json!({
        "enableLabels": settings_guard.labels.enable_labels,
        "textColor": settings_guard.labels.text_color
    });

    let ar_settings = serde_json::json!({
        "enablePlaneDetection": settings_guard.ar.enable_plane_detection,
        "enableHandTracking": settings_guard.ar.enable_hand_tracking,
        "enableHaptics": settings_guard.ar.enable_haptics
    });

    // Combine all settings
    let mut settings_map = serde_json::Map::new();
    if let Some(obj) = node_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = edge_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = physics_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = rendering_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = bloom_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = animation_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = label_settings.as_object() {
        settings_map.extend(obj.clone());
    }
    if let Some(obj) = ar_settings.as_object() {
        settings_map.extend(obj.clone());
    }

    HttpResponse::Ok().json(serde_json::Value::Object(settings_map))
}

// PUT /api/visualization/settings
pub async fn update_visualization_settings(
    settings: web::Data<Arc<RwLock<Settings>>>,
    new_settings: web::Json<serde_json::Value>,
) -> HttpResponse {
    // Clone the settings Data before borrowing
    let settings_clone = settings.clone();
    let mut settings_guard = settings.write().await;
    
    if let Some(obj) = new_settings.as_object() {
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
        if let Some(highlight_duration) = obj.get("highlightDuration").and_then(|v| v.as_f64()) {
            settings_guard.nodes.highlight_duration = highlight_duration as u32;
        }
        if let Some(enable_hover_effect) = obj.get("enableHoverEffect").and_then(|v| v.as_bool()) {
            settings_guard.nodes.enable_hover_effect = enable_hover_effect;
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

        // Update physics settings
        if let Some(enabled) = obj.get("physicsEnabled").and_then(|v| v.as_bool()) {
            settings_guard.physics.enabled = enabled;
        }
        if let Some(attraction_strength) = obj.get("attractionStrength").and_then(|v| v.as_f64()) {
            settings_guard.physics.attraction_strength = attraction_strength as f32;
        }
        if let Some(repulsion_strength) = obj.get("repulsionStrength").and_then(|v| v.as_f64()) {
            settings_guard.physics.repulsion_strength = repulsion_strength as f32;
        }
        if let Some(spring_strength) = obj.get("springStrength").and_then(|v| v.as_f64()) {
            settings_guard.physics.spring_strength = spring_strength as f32;
        }
        if let Some(damping) = obj.get("damping").and_then(|v| v.as_f64()) {
            settings_guard.physics.damping = damping as f32;
        }
        if let Some(max_velocity) = obj.get("maxVelocity").and_then(|v| v.as_f64()) {
            settings_guard.physics.max_velocity = max_velocity as f32;
        }
        if let Some(collision_radius) = obj.get("collisionRadius").and_then(|v| v.as_f64()) {
            settings_guard.physics.collision_radius = collision_radius as f32;
        }
        if let Some(bounds_size) = obj.get("boundsSize").and_then(|v| v.as_f64()) {
            settings_guard.physics.bounds_size = bounds_size as f32;
        }
        if let Some(enable_bounds) = obj.get("enableBounds").and_then(|v| v.as_bool()) {
            settings_guard.physics.enable_bounds = enable_bounds;
        }
        if let Some(iterations) = obj.get("iterations").and_then(|v| v.as_f64()) {
            settings_guard.physics.iterations = iterations as u32;
        }

        // Update rendering settings
        if let Some(ambient_light_intensity) = obj.get("ambientLightIntensity").and_then(|v| v.as_f64()) {
            settings_guard.rendering.ambient_light_intensity = ambient_light_intensity as f32;
        }
        if let Some(directional_light_intensity) = obj.get("directionalLightIntensity").and_then(|v| v.as_f64()) {
            settings_guard.rendering.directional_light_intensity = directional_light_intensity as f32;
        }
        if let Some(environment_intensity) = obj.get("environmentIntensity").and_then(|v| v.as_f64()) {
            settings_guard.rendering.environment_intensity = environment_intensity as f32;
        }
        if let Some(enable_ambient_occlusion) = obj.get("enableAmbientOcclusion").and_then(|v| v.as_bool()) {
            settings_guard.rendering.enable_ambient_occlusion = enable_ambient_occlusion;
        }
        if let Some(enable_antialiasing) = obj.get("enableAntialiasing").and_then(|v| v.as_bool()) {
            settings_guard.rendering.enable_antialiasing = enable_antialiasing;
        }
        if let Some(enable_shadows) = obj.get("enableShadows").and_then(|v| v.as_bool()) {
            settings_guard.rendering.enable_shadows = enable_shadows;
        }
        if let Some(background_color) = obj.get("backgroundColor").and_then(|v| v.as_str()) {
            settings_guard.rendering.background_color = background_color.to_string();
        }

        // Update bloom settings
        if let Some(enabled) = obj.get("bloomEnabled").and_then(|v| v.as_bool()) {
            settings_guard.bloom.enabled = enabled;
        }
        if let Some(node_bloom_strength) = obj.get("nodeBloomStrength").and_then(|v| v.as_f64()) {
            settings_guard.bloom.node_bloom_strength = node_bloom_strength as f32;
        }
        if let Some(edge_bloom_strength) = obj.get("edgeBloomStrength").and_then(|v| v.as_f64()) {
            settings_guard.bloom.edge_bloom_strength = edge_bloom_strength as f32;
        }
        if let Some(environment_bloom_strength) = obj.get("environmentBloomStrength").and_then(|v| v.as_f64()) {
            settings_guard.bloom.environment_bloom_strength = environment_bloom_strength as f32;
        }

        // Update animation settings
        if let Some(enable_node_animations) = obj.get("enableNodeAnimations").and_then(|v| v.as_bool()) {
            settings_guard.animations.enable_node_animations = enable_node_animations;
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

        // Update label settings
        if let Some(enable_labels) = obj.get("enableLabels").and_then(|v| v.as_bool()) {
            settings_guard.labels.enable_labels = enable_labels;
        }
        if let Some(text_color) = obj.get("textColor").and_then(|v| v.as_str()) {
            settings_guard.labels.text_color = text_color.to_string();
        }

        // Update AR settings
        if let Some(enable_plane_detection) = obj.get("enablePlaneDetection").and_then(|v| v.as_bool()) {
            settings_guard.ar.enable_plane_detection = enable_plane_detection;
        }
        if let Some(enable_hand_tracking) = obj.get("enableHandTracking").and_then(|v| v.as_bool()) {
            settings_guard.ar.enable_hand_tracking = enable_hand_tracking;
        }
        if let Some(enable_haptics) = obj.get("enableHaptics").and_then(|v| v.as_bool()) {
            settings_guard.ar.enable_haptics = enable_haptics;
        }

        // Save settings to file
        if let Err(e) = save_settings_to_file(&settings_guard) {
            eprintln!("Failed to save settings to file: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to save settings to file"
            }));
        }

        // Drop the write guard before returning the current settings
        drop(settings_guard);
        
        // Return the current settings using the cloned Data
        get_visualization_settings(settings_clone).await
    } else {
        HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Invalid settings format"
        }))
    }
}

fn update_settings(settings: &mut Settings, new_settings: Value) {
    if let Some(obj) = new_settings.as_object() {
        for (section, values) in obj {
            match section.as_str() {
                "rendering" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.rendering, v);
                },
                "nodes" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.nodes, v);
                },
                "edges" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.edges, v);
                },
                "labels" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.labels, v);
                },
                "bloom" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.bloom, v);
                },
                "ar" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.ar, v);
                },
                "physics" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.physics, v);
                },
                "animations" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.animations, v);
                },
                "audio" => if let Some(v) = values.as_object() {
                    update_section(&mut settings.audio, v);
                },
                _ => {}
            }
        }
    }
}

fn update_section<T: serde::de::DeserializeOwned + serde::Serialize>(
    section: &mut T,
    values: &serde_json::Map<String, Value>
) {
    // Create a reference to section to avoid moving it
    let current = serde_json::to_value(&*section).unwrap_or(Value::Null);
    if let Value::Object(mut current_map) = current {
        // Update only provided values
        for (key, value) in values {
            current_map.insert(key.clone(), value.clone());
        }
        // Convert back to the original type
        if let Ok(updated) = serde_json::from_value(Value::Object(current_map)) {
            *section = updated;
        }
    }
}

fn save_settings_to_file(settings: &Settings) -> std::io::Result<()> {
    // Convert settings to TOML
    let toml_string = toml::to_string_pretty(&settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    // Get the absolute path to settings.toml
    let settings_path = PathBuf::from(std::env::current_dir()?)
        .join("settings.toml");
    
    // Write to settings.toml with absolute path
    fs::write(&settings_path, toml_string)?;
    
    log::info!("Settings saved to: {:?}", settings_path);
    
    Ok(())
}

// Register the handlers with the Actix web app
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.route("/visualization/settings", web::get().to(get_visualization_settings))
       .route("/visualization/settings", web::put().to(update_visualization_settings));
}
