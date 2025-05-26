use crate::app_state::AppState;
use crate::models::{UISettings, UserSettings};
use crate::config::AppFullSettings; // Removed ClientFacingSettings alias
use crate::models::client_settings_payload::*; // Import all DTOs
// use crate::handlers::socket_flow_handler::ClientManager;
use actix_web::{web, Error, HttpResponse, HttpRequest};
use chrono::Utc;
use serde_json::json;
use crate::config::feature_access::FeatureAccess;
use log::{info, error, warn, debug};
use std::time::Instant;

// Helper function to convert AppFullSettings to UISettings (requires From impl update)
// This assumes the From impl is updated in models/ui_settings.rs
fn convert_to_ui_settings(full_settings: &AppFullSettings) -> UISettings {
    UISettings::from(full_settings) // Rely on the From trait implementation
}

// --- Helper Macros for Merging Settings ---

// Helper macro for merging Option fields
// Assigns source value (wrapped in Some) to target if source is Some.
macro_rules! merge_copy_option {
    ($target:expr, $source:expr) => {
        if let Some(value_ref) = $source.as_ref() {
            $target = *value_ref; // For Copy types
        }
    };
}

macro_rules! merge_clone_option {
    ($target:expr, $source:expr) => {
        if let Some(value_ref) = $source.as_ref() {
            $target = value_ref.clone(); // For Clone types
        }
    };
}

// --- Cache Clearing Endpoints (Unaffected by Settings struct changes) ---

async fn clear_user_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted to clear settings cache without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }
    UserSettings::clear_cache(&pubkey);
    info!("Cleared settings cache for user {}", pubkey);
    Ok(HttpResponse::Ok().json(json!({ "status": "success", "message": "Settings cache cleared" })))
}

async fn clear_all_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    if !feature_access.is_power_user(&pubkey) {
        warn!("Non-power user {} attempted to clear all settings caches", pubkey);
        return Ok(HttpResponse::Forbidden().body("Only power users can clear all settings caches"));
    }
    UserSettings::clear_all_cache();
    info!("Power user {} cleared all settings caches", pubkey);
    Ok(HttpResponse::Ok().json(json!({ "status": "success", "message": "All settings caches cleared" })))
}

// --- Configuration ---

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/user-settings")
            .route(web::get().to(get_public_settings))
            .route(web::post().to(update_settings))
    ).service(
        web::resource("/user-settings/sync")
            .route(web::get().to(get_user_settings))
            .route(web::post().to(update_user_settings)) // This now points to the updated function
    ).service(
        web::resource("/user-settings/clear-cache")
            .route(web::post().to(clear_user_settings_cache))
    ).service(
        web::resource("/admin/settings/clear-all-cache")
            .route(web::post().to(clear_all_settings_cache))
    );
}

// --- GET Endpoints ---

pub async fn get_public_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = state.settings.read().await;
    let ui_settings = convert_to_ui_settings(&*settings_guard);
    Ok(HttpResponse::Ok().json(&ui_settings))
}

async fn get_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let start_time = Instant::now();
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Missing Nostr pubkey in request headers for get_user_settings");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };
    debug!("Processing get_user_settings request for user: {}", pubkey);

    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted get_user_settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    let is_power_user = feature_access.is_power_user(&pubkey);
    let result;

    if is_power_user {
        let settings_guard = state.settings.read().await;
        let ui_settings = convert_to_ui_settings(&*settings_guard);
        debug!("Returning global UI settings for power user {}", pubkey);
        result = Ok(HttpResponse::Ok().json(ui_settings));
    } else {
        let user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {} with default settings", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });
        result = Ok(HttpResponse::Ok().json(&user_settings.settings));
    }

    let elapsed = start_time.elapsed();
    debug!("Settings request for {} processed in {:?}", pubkey, elapsed);
    result
}

// --- POST Endpoints ---

// Handles updates from the main settings UI (/user-settings/sync)
async fn update_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    payload: web::Json<ClientSettingsPayload>, // Use the new DTO
) -> Result<HttpResponse, Error> {
    let _start_time = Instant::now(); // Prefixed with underscore
    let client_payload = payload.into_inner();

    debug!("Received client settings payload: {:?}", client_payload);

    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Update settings request received without Nostr pubkey.");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey for settings update"));
        }
    };
    debug!("Processing update_user_settings for user: {}", pubkey);

    if !feature_access.can_sync_settings(&pubkey) {
        warn!("User {} attempted update_user_settings without permission", pubkey);
        return Ok(HttpResponse::Forbidden().body("Settings sync not enabled for this user"));
    }

    let is_power_user = feature_access.is_power_user(&pubkey);

    if is_power_user {
        let mut settings_guard = state.settings.write().await; // AppFullSettings

        // --- Merge from ClientSettingsPayload DTO into AppFullSettings ---
        // Macros are now defined at module level

        if let Some(vis_dto) = client_payload.visualisation {
            let target_vis = &mut settings_guard.visualisation;
            if let Some(nodes_dto) = vis_dto.nodes {
                merge_clone_option!(target_vis.nodes.base_color, nodes_dto.base_color);
                merge_copy_option!(target_vis.nodes.metalness, nodes_dto.metalness);
                merge_copy_option!(target_vis.nodes.opacity, nodes_dto.opacity);
                merge_copy_option!(target_vis.nodes.roughness, nodes_dto.roughness);
                merge_copy_option!(target_vis.nodes.node_size, nodes_dto.node_size); // Changed from size_range, and to merge_copy_option
                merge_clone_option!(target_vis.nodes.quality, nodes_dto.quality);
                merge_copy_option!(target_vis.nodes.enable_instancing, nodes_dto.enable_instancing);
                merge_copy_option!(target_vis.nodes.enable_hologram, nodes_dto.enable_hologram);
                merge_copy_option!(target_vis.nodes.enable_metadata_shape, nodes_dto.enable_metadata_shape);
                merge_copy_option!(target_vis.nodes.enable_metadata_visualisation, nodes_dto.enable_metadata_visualisation);
            }
            if let Some(edges_dto) = vis_dto.edges {
                let target_edges = &mut target_vis.edges;
                merge_copy_option!(target_edges.arrow_size, edges_dto.arrow_size);
                merge_copy_option!(target_edges.base_width, edges_dto.base_width);
                merge_clone_option!(target_edges.color, edges_dto.color);
                merge_copy_option!(target_edges.enable_arrows, edges_dto.enable_arrows);
                merge_copy_option!(target_edges.opacity, edges_dto.opacity);
                merge_clone_option!(target_edges.width_range, edges_dto.width_range);
                merge_clone_option!(target_edges.quality, edges_dto.quality);
                // Note: ClientEdgeSettings DTO has extra fields not in server's EdgeSettings.
                // They are ignored here as AppFullSettings.visualisation.edges doesn't have them.
            }
            if let Some(physics_dto) = vis_dto.physics { // physics_dto is ClientPhysicsSettings
                let target_physics = &mut target_vis.physics; // Type: config::PhysicsSettings
                merge_copy_option!(target_physics.attraction_strength, physics_dto.attraction_strength);
                merge_copy_option!(target_physics.bounds_size, physics_dto.bounds_size);
                merge_copy_option!(target_physics.collision_radius, physics_dto.collision_radius);
                merge_copy_option!(target_physics.damping, physics_dto.damping);
                merge_copy_option!(target_physics.enable_bounds, physics_dto.enable_bounds);
                merge_copy_option!(target_physics.enabled, physics_dto.enabled);
                merge_copy_option!(target_physics.iterations, physics_dto.iterations);
                merge_copy_option!(target_physics.max_velocity, physics_dto.max_velocity);
                merge_copy_option!(target_physics.repulsion_strength, physics_dto.repulsion_strength);
                merge_copy_option!(target_physics.spring_strength, physics_dto.spring_strength);
                merge_copy_option!(target_physics.repulsion_distance, physics_dto.repulsion_distance);
                merge_copy_option!(target_physics.mass_scale, physics_dto.mass_scale);
                merge_copy_option!(target_physics.boundary_damping, physics_dto.boundary_damping);
            }
             if let Some(rendering_dto) = vis_dto.rendering { // rendering_dto is ClientRenderingSettings
                let target_rendering = &mut target_vis.rendering; // Type: config::RenderingSettings
                merge_copy_option!(target_rendering.ambient_light_intensity, rendering_dto.ambient_light_intensity);
                merge_clone_option!(target_rendering.background_color, rendering_dto.background_color);
                merge_copy_option!(target_rendering.directional_light_intensity, rendering_dto.directional_light_intensity);
                merge_copy_option!(target_rendering.enable_ambient_occlusion, rendering_dto.enable_ambient_occlusion);
                merge_copy_option!(target_rendering.enable_antialiasing, rendering_dto.enable_antialiasing);
                merge_copy_option!(target_rendering.enable_shadows, rendering_dto.enable_shadows);
                merge_copy_option!(target_rendering.environment_intensity, rendering_dto.environment_intensity);
                // Client DTO has shadow_map_size, shadow_bias, context - not in server RenderingSettings
            }
             if let Some(anim_dto) = vis_dto.animations { // anim_dto is ClientAnimationSettings
                let target_anim = &mut target_vis.animations; // Type: config::AnimationSettings
                merge_copy_option!(target_anim.enable_motion_blur, anim_dto.enable_motion_blur);
                merge_copy_option!(target_anim.enable_node_animations, anim_dto.enable_node_animations);
                merge_copy_option!(target_anim.motion_blur_strength, anim_dto.motion_blur_strength);
                merge_copy_option!(target_anim.selection_wave_enabled, anim_dto.selection_wave_enabled);
                merge_copy_option!(target_anim.pulse_enabled, anim_dto.pulse_enabled);
                merge_copy_option!(target_anim.pulse_speed, anim_dto.pulse_speed);
                merge_copy_option!(target_anim.pulse_strength, anim_dto.pulse_strength);
                merge_copy_option!(target_anim.wave_speed, anim_dto.wave_speed);
            }
             if let Some(labels_dto) = vis_dto.labels { // labels_dto is ClientLabelSettings
                let target_labels = &mut target_vis.labels; // Type: config::LabelSettings
                merge_copy_option!(target_labels.desktop_font_size, labels_dto.desktop_font_size);
                merge_copy_option!(target_labels.enable_labels, labels_dto.enable_labels);
                merge_clone_option!(target_labels.text_color, labels_dto.text_color);
                merge_clone_option!(target_labels.text_outline_color, labels_dto.text_outline_color);
                merge_copy_option!(target_labels.text_outline_width, labels_dto.text_outline_width);
                merge_copy_option!(target_labels.text_resolution, labels_dto.text_resolution);
                merge_copy_option!(target_labels.text_padding, labels_dto.text_padding);
                merge_clone_option!(target_labels.billboard_mode, labels_dto.billboard_mode);
            }
             if let Some(bloom_dto) = vis_dto.bloom { // bloom_dto is ClientBloomSettings
                let target_bloom = &mut target_vis.bloom; // Type: config::BloomSettings
                merge_copy_option!(target_bloom.edge_bloom_strength, bloom_dto.edge_bloom_strength);
                merge_copy_option!(target_bloom.enabled, bloom_dto.enabled);
                merge_copy_option!(target_bloom.environment_bloom_strength, bloom_dto.environment_bloom_strength);
                merge_copy_option!(target_bloom.node_bloom_strength, bloom_dto.node_bloom_strength);
                merge_copy_option!(target_bloom.radius, bloom_dto.radius);
                merge_copy_option!(target_bloom.strength, bloom_dto.strength);
                // Client DTO has threshold - not in server BloomSettings
            }
             if let Some(hologram_dto) = vis_dto.hologram { // hologram_dto is ClientHologramSettings
                let target_hologram = &mut target_vis.hologram; // Type: config::HologramSettings
                merge_copy_option!(target_hologram.ring_count, hologram_dto.ring_count);
                merge_clone_option!(target_hologram.ring_color, hologram_dto.ring_color);
                merge_copy_option!(target_hologram.ring_opacity, hologram_dto.ring_opacity);
                merge_clone_option!(target_hologram.sphere_sizes, hologram_dto.sphere_sizes);
                merge_copy_option!(target_hologram.ring_rotation_speed, hologram_dto.ring_rotation_speed);
                merge_copy_option!(target_hologram.enable_buckminster, hologram_dto.enable_buckminster);
                merge_copy_option!(target_hologram.buckminster_size, hologram_dto.buckminster_size);
                merge_copy_option!(target_hologram.buckminster_opacity, hologram_dto.buckminster_opacity);
                merge_copy_option!(target_hologram.enable_geodesic, hologram_dto.enable_geodesic);
                merge_copy_option!(target_hologram.geodesic_size, hologram_dto.geodesic_size);
                merge_copy_option!(target_hologram.geodesic_opacity, hologram_dto.geodesic_opacity);
                merge_copy_option!(target_hologram.enable_triangle_sphere, hologram_dto.enable_triangle_sphere);
                merge_copy_option!(target_hologram.triangle_sphere_size, hologram_dto.triangle_sphere_size);
                merge_copy_option!(target_hologram.triangle_sphere_opacity, hologram_dto.triangle_sphere_opacity);
                merge_copy_option!(target_hologram.global_rotation_speed, hologram_dto.global_rotation_speed);
            }
            // Camera settings are not part of AppFullSettings.visualisation, so ignore vis_dto.camera
        }

        if let Some(xr_dto) = client_payload.xr {
            let target_xr = &mut settings_guard.xr;
            merge_clone_option!(target_xr.mode, xr_dto.mode);
            merge_copy_option!(target_xr.room_scale, xr_dto.room_scale);
            merge_clone_option!(target_xr.space_type, xr_dto.space_type);
            merge_clone_option!(target_xr.quality, xr_dto.quality);
            merge_copy_option!(target_xr.enable_hand_tracking, xr_dto.enable_hand_tracking);
            merge_copy_option!(target_xr.hand_mesh_enabled, xr_dto.hand_mesh_enabled);
            merge_clone_option!(target_xr.hand_mesh_color, xr_dto.hand_mesh_color);
            merge_copy_option!(target_xr.hand_mesh_opacity, xr_dto.hand_mesh_opacity);
            merge_copy_option!(target_xr.hand_point_size, xr_dto.hand_point_size);
            merge_copy_option!(target_xr.hand_ray_enabled, xr_dto.hand_ray_enabled);
            merge_clone_option!(target_xr.hand_ray_color, xr_dto.hand_ray_color);
            merge_copy_option!(target_xr.hand_ray_width, xr_dto.hand_ray_width);
            merge_copy_option!(target_xr.gesture_smoothing, xr_dto.gesture_smoothing);
            merge_copy_option!(target_xr.enable_haptics, xr_dto.enable_haptics);
            merge_copy_option!(target_xr.drag_threshold, xr_dto.drag_threshold);
            merge_copy_option!(target_xr.pinch_threshold, xr_dto.pinch_threshold);
            merge_copy_option!(target_xr.rotation_threshold, xr_dto.rotation_threshold);
            merge_copy_option!(target_xr.interaction_radius, xr_dto.interaction_radius);
            merge_copy_option!(target_xr.movement_speed, xr_dto.movement_speed);
            merge_copy_option!(target_xr.dead_zone, xr_dto.dead_zone);
            if let Some(axes_dto) = xr_dto.movement_axes {
                merge_copy_option!(target_xr.movement_axes.horizontal, axes_dto.horizontal);
                merge_copy_option!(target_xr.movement_axes.vertical, axes_dto.vertical);
            }
            merge_copy_option!(target_xr.enable_light_estimation, xr_dto.enable_light_estimation);
            merge_copy_option!(target_xr.enable_plane_detection, xr_dto.enable_plane_detection);
            merge_copy_option!(target_xr.enable_scene_understanding, xr_dto.enable_scene_understanding);
            merge_clone_option!(target_xr.plane_color, xr_dto.plane_color);
            merge_copy_option!(target_xr.plane_opacity, xr_dto.plane_opacity);
            merge_copy_option!(target_xr.plane_detection_distance, xr_dto.plane_detection_distance);
            merge_copy_option!(target_xr.show_plane_overlay, xr_dto.show_plane_overlay);
            merge_copy_option!(target_xr.snap_to_floor, xr_dto.snap_to_floor);
            merge_copy_option!(target_xr.enable_passthrough_portal, xr_dto.enable_passthrough_portal);
            merge_copy_option!(target_xr.passthrough_opacity, xr_dto.passthrough_opacity);
            merge_copy_option!(target_xr.passthrough_brightness, xr_dto.passthrough_brightness);
            merge_copy_option!(target_xr.passthrough_contrast, xr_dto.passthrough_contrast);
            merge_copy_option!(target_xr.portal_size, xr_dto.portal_size);
            merge_clone_option!(target_xr.portal_edge_color, xr_dto.portal_edge_color);
            merge_copy_option!(target_xr.portal_edge_width, xr_dto.portal_edge_width);
            
            // Manual merge for Option<T> fields in XRSettings, as target_xr fields are Option<T>
            if xr_dto.enabled.is_some() { target_xr.enabled = xr_dto.enabled; }
            if xr_dto.controller_model.is_some() { target_xr.controller_model = xr_dto.controller_model.clone(); }
            if xr_dto.render_scale.is_some() { target_xr.render_scale = xr_dto.render_scale; }
            if xr_dto.locomotion_method.is_some() { target_xr.locomotion_method = xr_dto.locomotion_method.clone(); }
            if xr_dto.teleport_ray_color.is_some() { target_xr.teleport_ray_color = xr_dto.teleport_ray_color.clone(); }
            // xr_dto.mode (Option<String>) maps to target_xr.display_mode (Option<String>)
            if xr_dto.mode.is_some() { target_xr.display_mode = xr_dto.mode.clone(); }
            if xr_dto.controller_ray_color.is_some() { target_xr.controller_ray_color = xr_dto.controller_ray_color.clone(); }
        }

        if let Some(auth_dto) = client_payload.auth {
            let target_auth = &mut settings_guard.auth;
            merge_copy_option!(target_auth.enabled, auth_dto.enabled);
            merge_clone_option!(target_auth.provider, auth_dto.provider);
            merge_copy_option!(target_auth.required, auth_dto.required);
        }

        if let Some(sys_dto) = client_payload.system {
            let target_sys_config = &mut settings_guard.system; // ServerSystemConfigFromFile
            if let Some(ws_dto) = sys_dto.websocket {
                let target_ws = &mut target_sys_config.websocket; // ServerFullWebSocketSettings
                merge_copy_option!(target_ws.reconnect_attempts, ws_dto.reconnect_attempts);
                merge_copy_option!(target_ws.reconnect_delay, ws_dto.reconnect_delay);
                merge_copy_option!(target_ws.binary_chunk_size, ws_dto.binary_chunk_size);
                merge_copy_option!(target_ws.compression_enabled, ws_dto.compression_enabled);
                merge_copy_option!(target_ws.compression_threshold, ws_dto.compression_threshold);
                merge_copy_option!(target_ws.update_rate, ws_dto.update_rate);
            }
            if let Some(debug_dto) = sys_dto.debug {
                let target_debug = &mut target_sys_config.debug; // DebugSettings (server version)
                merge_copy_option!(target_debug.enabled, debug_dto.enabled);
                merge_copy_option!(target_debug.enable_data_debug, debug_dto.enable_data_debug);
                merge_copy_option!(target_debug.enable_websocket_debug, debug_dto.enable_websocket_debug);
                merge_copy_option!(target_debug.log_binary_headers, debug_dto.log_binary_headers);
                merge_copy_option!(target_debug.log_full_json, debug_dto.log_full_json);
                // Server DebugSettings has log_level, log_format not in ClientPayloadDebugSettings
            }
            merge_copy_option!(target_sys_config.persist_settings, sys_dto.persist_settings);
            // custom_backend_url is client-only, not in ServerSystemConfigFromFile
        }
        
        // AI settings merge (all are Option<Struct> on AppFullSettings)
        if client_payload.ragflow.is_some() { settings_guard.ragflow = client_payload.ragflow.map(|dto| crate::config::RagFlowSettings {
            api_key: dto.api_key, agent_id: dto.agent_id, api_base_url: dto.api_base_url,
            timeout: dto.timeout, max_retries: dto.max_retries, chat_id: dto.chat_id,
        })};
        if client_payload.perplexity.is_some() { settings_guard.perplexity = client_payload.perplexity.map(|dto| crate::config::PerplexitySettings {
            api_key: dto.api_key, model: dto.model, api_url: dto.api_url, max_tokens: dto.max_tokens,
            temperature: dto.temperature, top_p: dto.top_p, presence_penalty: dto.presence_penalty,
            frequency_penalty: dto.frequency_penalty, timeout: dto.timeout, rate_limit: dto.rate_limit,
        })};
        if client_payload.openai.is_some() { settings_guard.openai = client_payload.openai.map(|dto| crate::config::OpenAISettings {
            api_key: dto.api_key, base_url: dto.base_url, timeout: dto.timeout, rate_limit: dto.rate_limit,
        })};
        if client_payload.kokoro.is_some() { settings_guard.kokoro = client_payload.kokoro.map(|dto| crate::config::KokoroSettings {
            api_url: dto.api_url, default_voice: dto.default_voice, default_format: dto.default_format,
            default_speed: dto.default_speed, timeout: dto.timeout, stream: dto.stream,
            return_timestamps: dto.return_timestamps, sample_rate: dto.sample_rate,
        })};
        // --- End Merge ---

        if let Err(e) = settings_guard.save() {
            error!("Failed to save global AppFullSettings after update from {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
        }

        info!("Power user {} updated global settings", pubkey);
        let updated_ui_settings = convert_to_ui_settings(&*settings_guard);
        Ok(HttpResponse::Ok().json(updated_ui_settings))

    } else { // Regular user - updates their UserSettings file (which stores UISettings)
        let mut user_settings = UserSettings::load(&pubkey).unwrap_or_else(|| {
            debug!("Creating new user settings for {}", pubkey);
            UserSettings::new(&pubkey, UISettings::default())
        });

        // Merge relevant parts of ClientSettingsPayload into user_settings.settings (UISettings)
        let target_ui_settings = &mut user_settings.settings;

        if let Some(vis_dto) = client_payload.visualisation { // vis_dto is ClientVisualisationSettings
            let target_vis = &mut target_ui_settings.visualisation; // Type: config::VisualisationSettings
            if let Some(nodes_dto) = vis_dto.nodes { // nodes_dto is ClientNodeSettings
                let target_nodes = &mut target_vis.nodes;
                merge_clone_option!(target_nodes.base_color, nodes_dto.base_color);
                merge_copy_option!(target_nodes.metalness, nodes_dto.metalness);
                merge_copy_option!(target_nodes.opacity, nodes_dto.opacity);
                merge_copy_option!(target_nodes.roughness, nodes_dto.roughness);
                merge_copy_option!(target_nodes.node_size, nodes_dto.node_size); // Changed from size_range, and to merge_copy_option
                merge_clone_option!(target_nodes.quality, nodes_dto.quality);
                merge_copy_option!(target_nodes.enable_instancing, nodes_dto.enable_instancing);
                merge_copy_option!(target_nodes.enable_hologram, nodes_dto.enable_hologram);
                merge_copy_option!(target_nodes.enable_metadata_shape, nodes_dto.enable_metadata_shape);
                merge_copy_option!(target_nodes.enable_metadata_visualisation, nodes_dto.enable_metadata_visualisation);
            }
            if let Some(edges_dto) = vis_dto.edges { // edges_dto is ClientEdgeSettings
                let target_edges = &mut target_vis.edges;
                merge_copy_option!(target_edges.arrow_size, edges_dto.arrow_size);
                merge_copy_option!(target_edges.base_width, edges_dto.base_width);
                merge_clone_option!(target_edges.color, edges_dto.color);
                merge_copy_option!(target_edges.enable_arrows, edges_dto.enable_arrows);
                merge_copy_option!(target_edges.opacity, edges_dto.opacity);
                merge_clone_option!(target_edges.width_range, edges_dto.width_range);
                merge_clone_option!(target_edges.quality, edges_dto.quality);
                // Extra fields in ClientEdgeSettings DTO (enable_flow_effect etc.) are ignored as they are not in config::EdgeSettings
            }
            if let Some(physics_dto) = vis_dto.physics { // physics_dto is ClientPhysicsSettings
                let target_physics = &mut target_vis.physics; // Type: config::PhysicsSettings
                merge_copy_option!(target_physics.attraction_strength, physics_dto.attraction_strength);
                merge_copy_option!(target_physics.bounds_size, physics_dto.bounds_size);
                merge_copy_option!(target_physics.collision_radius, physics_dto.collision_radius);
                merge_copy_option!(target_physics.damping, physics_dto.damping);
                merge_copy_option!(target_physics.enable_bounds, physics_dto.enable_bounds);
                merge_copy_option!(target_physics.enabled, physics_dto.enabled);
                merge_copy_option!(target_physics.iterations, physics_dto.iterations);
                merge_copy_option!(target_physics.max_velocity, physics_dto.max_velocity);
                merge_copy_option!(target_physics.repulsion_strength, physics_dto.repulsion_strength);
                merge_copy_option!(target_physics.spring_strength, physics_dto.spring_strength);
                merge_copy_option!(target_physics.repulsion_distance, physics_dto.repulsion_distance);
                merge_copy_option!(target_physics.mass_scale, physics_dto.mass_scale);
                merge_copy_option!(target_physics.boundary_damping, physics_dto.boundary_damping);
            }
             if let Some(rendering_dto) = vis_dto.rendering { // rendering_dto is ClientRenderingSettings
                let target_rendering = &mut target_vis.rendering; // Type: config::RenderingSettings
                merge_copy_option!(target_rendering.ambient_light_intensity, rendering_dto.ambient_light_intensity);
                merge_clone_option!(target_rendering.background_color, rendering_dto.background_color);
                merge_copy_option!(target_rendering.directional_light_intensity, rendering_dto.directional_light_intensity);
                merge_copy_option!(target_rendering.enable_ambient_occlusion, rendering_dto.enable_ambient_occlusion);
                merge_copy_option!(target_rendering.enable_antialiasing, rendering_dto.enable_antialiasing);
                merge_copy_option!(target_rendering.enable_shadows, rendering_dto.enable_shadows);
                merge_copy_option!(target_rendering.environment_intensity, rendering_dto.environment_intensity);
                 // Extra fields in ClientRenderingSettings DTO (shadow_map_size etc.) are ignored.
            }
             if let Some(anim_dto) = vis_dto.animations { // anim_dto is ClientAnimationSettings
                let target_anim = &mut target_vis.animations; // Type: config::AnimationSettings
                merge_copy_option!(target_anim.enable_motion_blur, anim_dto.enable_motion_blur);
                merge_copy_option!(target_anim.enable_node_animations, anim_dto.enable_node_animations);
                merge_copy_option!(target_anim.motion_blur_strength, anim_dto.motion_blur_strength);
                merge_copy_option!(target_anim.selection_wave_enabled, anim_dto.selection_wave_enabled);
                merge_copy_option!(target_anim.pulse_enabled, anim_dto.pulse_enabled);
                merge_copy_option!(target_anim.pulse_speed, anim_dto.pulse_speed);
                merge_copy_option!(target_anim.pulse_strength, anim_dto.pulse_strength);
                merge_copy_option!(target_anim.wave_speed, anim_dto.wave_speed);
            }
             if let Some(labels_dto) = vis_dto.labels { // labels_dto is ClientLabelSettings
                let target_labels = &mut target_vis.labels; // Type: config::LabelSettings
                merge_copy_option!(target_labels.desktop_font_size, labels_dto.desktop_font_size);
                merge_copy_option!(target_labels.enable_labels, labels_dto.enable_labels);
                merge_clone_option!(target_labels.text_color, labels_dto.text_color);
                merge_clone_option!(target_labels.text_outline_color, labels_dto.text_outline_color);
                merge_copy_option!(target_labels.text_outline_width, labels_dto.text_outline_width);
                merge_copy_option!(target_labels.text_resolution, labels_dto.text_resolution);
                merge_copy_option!(target_labels.text_padding, labels_dto.text_padding);
                merge_clone_option!(target_labels.billboard_mode, labels_dto.billboard_mode);
            }
             if let Some(bloom_dto) = vis_dto.bloom { // bloom_dto is ClientBloomSettings
                let target_bloom = &mut target_vis.bloom; // Type: config::BloomSettings
                merge_copy_option!(target_bloom.edge_bloom_strength, bloom_dto.edge_bloom_strength);
                merge_copy_option!(target_bloom.enabled, bloom_dto.enabled);
                merge_copy_option!(target_bloom.environment_bloom_strength, bloom_dto.environment_bloom_strength);
                merge_copy_option!(target_bloom.node_bloom_strength, bloom_dto.node_bloom_strength);
                merge_copy_option!(target_bloom.radius, bloom_dto.radius);
                merge_copy_option!(target_bloom.strength, bloom_dto.strength);
                 // Extra field 'threshold' in ClientBloomSettings DTO is ignored.
            }
             if let Some(hologram_dto) = vis_dto.hologram { // hologram_dto is ClientHologramSettings
                let target_hologram = &mut target_vis.hologram; // Type: config::HologramSettings
                merge_copy_option!(target_hologram.ring_count, hologram_dto.ring_count);
                merge_clone_option!(target_hologram.ring_color, hologram_dto.ring_color);
                merge_copy_option!(target_hologram.ring_opacity, hologram_dto.ring_opacity);
                merge_clone_option!(target_hologram.sphere_sizes, hologram_dto.sphere_sizes);
                merge_copy_option!(target_hologram.ring_rotation_speed, hologram_dto.ring_rotation_speed);
                merge_copy_option!(target_hologram.enable_buckminster, hologram_dto.enable_buckminster);
                merge_copy_option!(target_hologram.buckminster_size, hologram_dto.buckminster_size);
                merge_copy_option!(target_hologram.buckminster_opacity, hologram_dto.buckminster_opacity);
                merge_copy_option!(target_hologram.enable_geodesic, hologram_dto.enable_geodesic);
                merge_copy_option!(target_hologram.geodesic_size, hologram_dto.geodesic_size);
                merge_copy_option!(target_hologram.geodesic_opacity, hologram_dto.geodesic_opacity);
                merge_copy_option!(target_hologram.enable_triangle_sphere, hologram_dto.enable_triangle_sphere);
                merge_copy_option!(target_hologram.triangle_sphere_size, hologram_dto.triangle_sphere_size);
                merge_copy_option!(target_hologram.triangle_sphere_opacity, hologram_dto.triangle_sphere_opacity);
                merge_copy_option!(target_hologram.global_rotation_speed, hologram_dto.global_rotation_speed);
            }
            // ClientVisualisationSettings DTO has 'camera' but UISettings.visualisation (config::VisualisationSettings) does not.
        }

        if let Some(xr_dto) = client_payload.xr { // xr_dto is ClientXRSettings
            let target_xr = &mut target_ui_settings.xr; // Type: config::XRSettings
            merge_clone_option!(target_xr.mode, xr_dto.mode);
            merge_copy_option!(target_xr.room_scale, xr_dto.room_scale);
            merge_clone_option!(target_xr.space_type, xr_dto.space_type);
            merge_clone_option!(target_xr.quality, xr_dto.quality);
            merge_copy_option!(target_xr.enable_hand_tracking, xr_dto.enable_hand_tracking);
            merge_copy_option!(target_xr.hand_mesh_enabled, xr_dto.hand_mesh_enabled);
            merge_clone_option!(target_xr.hand_mesh_color, xr_dto.hand_mesh_color);
            merge_copy_option!(target_xr.hand_mesh_opacity, xr_dto.hand_mesh_opacity);
            merge_copy_option!(target_xr.hand_point_size, xr_dto.hand_point_size);
            merge_copy_option!(target_xr.hand_ray_enabled, xr_dto.hand_ray_enabled);
            merge_clone_option!(target_xr.hand_ray_color, xr_dto.hand_ray_color);
            merge_copy_option!(target_xr.hand_ray_width, xr_dto.hand_ray_width);
            merge_copy_option!(target_xr.gesture_smoothing, xr_dto.gesture_smoothing);
            merge_copy_option!(target_xr.enable_haptics, xr_dto.enable_haptics);
            merge_copy_option!(target_xr.drag_threshold, xr_dto.drag_threshold);
            merge_copy_option!(target_xr.pinch_threshold, xr_dto.pinch_threshold);
            merge_copy_option!(target_xr.rotation_threshold, xr_dto.rotation_threshold);
            merge_copy_option!(target_xr.interaction_radius, xr_dto.interaction_radius);
            merge_copy_option!(target_xr.movement_speed, xr_dto.movement_speed);
            merge_copy_option!(target_xr.dead_zone, xr_dto.dead_zone);
            if let Some(axes_dto) = xr_dto.movement_axes { // axes_dto is ClientMovementAxes
                merge_copy_option!(target_xr.movement_axes.horizontal, axes_dto.horizontal);
                merge_copy_option!(target_xr.movement_axes.vertical, axes_dto.vertical);
            }
            merge_copy_option!(target_xr.enable_light_estimation, xr_dto.enable_light_estimation);
            merge_copy_option!(target_xr.enable_plane_detection, xr_dto.enable_plane_detection);
            merge_copy_option!(target_xr.enable_scene_understanding, xr_dto.enable_scene_understanding);
            merge_clone_option!(target_xr.plane_color, xr_dto.plane_color);
            merge_copy_option!(target_xr.plane_opacity, xr_dto.plane_opacity);
            merge_copy_option!(target_xr.plane_detection_distance, xr_dto.plane_detection_distance);
            merge_copy_option!(target_xr.show_plane_overlay, xr_dto.show_plane_overlay);
            merge_copy_option!(target_xr.snap_to_floor, xr_dto.snap_to_floor);
            merge_copy_option!(target_xr.enable_passthrough_portal, xr_dto.enable_passthrough_portal);
            merge_copy_option!(target_xr.passthrough_opacity, xr_dto.passthrough_opacity);
            merge_copy_option!(target_xr.passthrough_brightness, xr_dto.passthrough_brightness);
            merge_copy_option!(target_xr.passthrough_contrast, xr_dto.passthrough_contrast);
            merge_copy_option!(target_xr.portal_size, xr_dto.portal_size);
            merge_clone_option!(target_xr.portal_edge_color, xr_dto.portal_edge_color);
            merge_copy_option!(target_xr.portal_edge_width, xr_dto.portal_edge_width);
            
            // Manual merge for Option<T> fields in XRSettings
            if xr_dto.enabled.is_some() { target_xr.enabled = xr_dto.enabled; }
            if xr_dto.controller_model.is_some() { target_xr.controller_model = xr_dto.controller_model.clone(); }
            if xr_dto.render_scale.is_some() { target_xr.render_scale = xr_dto.render_scale; }
            if xr_dto.locomotion_method.is_some() { target_xr.locomotion_method = xr_dto.locomotion_method.clone(); }
            if xr_dto.teleport_ray_color.is_some() { target_xr.teleport_ray_color = xr_dto.teleport_ray_color.clone(); }
            if xr_dto.mode.is_some() { target_xr.display_mode = xr_dto.mode.clone(); } // xr_dto.mode maps to target_xr.display_mode (Option<String>)
            if xr_dto.controller_ray_color.is_some() { target_xr.controller_ray_color = xr_dto.controller_ray_color.clone(); }
        }

        if let Some(sys_dto) = client_payload.system { // sys_dto is ClientSystemSettings DTO
            // target_ui_settings.system is UISystemSettings
            if let Some(ws_dto) = sys_dto.websocket { // ws_dto is ClientPayloadWebSocketSettings DTO
                let target_ws = &mut target_ui_settings.system.websocket; // Type: config::ClientWebSocketSettings
                merge_copy_option!(target_ws.reconnect_attempts, ws_dto.reconnect_attempts);
                merge_copy_option!(target_ws.reconnect_delay, ws_dto.reconnect_delay);
                merge_copy_option!(target_ws.binary_chunk_size, ws_dto.binary_chunk_size);
                merge_copy_option!(target_ws.compression_enabled, ws_dto.compression_enabled);
                merge_copy_option!(target_ws.compression_threshold, ws_dto.compression_threshold);
                merge_copy_option!(target_ws.update_rate, ws_dto.update_rate);
            }
            if let Some(debug_dto) = sys_dto.debug { // debug_dto is ClientPayloadDebugSettings DTO
                let target_debug = &mut target_ui_settings.system.debug; // Type: config::DebugSettings
                merge_copy_option!(target_debug.enabled, debug_dto.enabled);
                merge_copy_option!(target_debug.enable_data_debug, debug_dto.enable_data_debug);
                merge_copy_option!(target_debug.enable_websocket_debug, debug_dto.enable_websocket_debug);
                merge_copy_option!(target_debug.log_binary_headers, debug_dto.log_binary_headers);
                merge_copy_option!(target_debug.log_full_json, debug_dto.log_full_json);
                // Extra fields in ClientPayloadDebugSettings DTO are ignored.
                // log_level, log_format in config::DebugSettings are not settable by regular users via this DTO.
            }
            // persist_settings and custom_backend_url from ClientSystemSettings DTO are not part of UISystemSettings.
        }
        // Auth and AI settings are not part of UISettings for regular users and are not mapped.

        user_settings.last_modified = Utc::now().timestamp();

        if let Err(e) = user_settings.save() {
            error!("Failed to save user settings for {}: {}", pubkey, e);
            return Ok(HttpResponse::InternalServerError().body(format!("Failed to save user settings: {}", e)));
        }

        debug!("User {} updated their settings", pubkey);
        Ok(HttpResponse::Ok().json(&user_settings.settings))
    }
}

// Handles updates from the older /user-settings endpoint (needs review/deprecation?)
async fn update_settings( // This is the deprecated endpoint
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    payload: web::Json<ClientSettingsPayload>, // Should also use DTO if kept
) -> Result<HttpResponse, Error> {
    warn!("Received settings update via deprecated /user-settings endpoint. Use /user-settings/sync instead.");
    // For now, let's assume this endpoint might not need the full DTO merge logic
    // or should be fully deprecated. If it needs to work like /sync, copy merge logic.
    // For simplicity in this step, we'll keep its old behavior but with the DTO.
    // This will likely fail if ClientSettingsPayload is not directly convertible to AppFullSettings parts.
    // TODO: Refactor or remove this deprecated endpoint properly.
    // The following is a placeholder and likely incorrect without proper mapping.
    let client_payload = payload.into_inner();
    debug!("Deserialized payload via deprecated /user-settings: {:?}", client_payload);

    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            warn!("Attempt to update settings via /user-settings without authentication");
            return Ok(HttpResponse::BadRequest().body("Missing Nostr pubkey"));
        }
    };

    if !feature_access.is_power_user(&pubkey) {
        warn!("Non-power user {} attempted to modify global settings via /user-settings", pubkey);
        return Ok(HttpResponse::Forbidden().body("Only power users can modify global settings"));
    }

    // Perform the same careful merge as in update_user_settings
    let mut settings_guard = state.settings.write().await; // Locks AppFullSettings

    // --- Merge from ClientSettingsPayload DTO into AppFullSettings ---
    // Macros are defined at module level

    if let Some(vis_dto) = client_payload.visualisation {
        let target_vis = &mut settings_guard.visualisation;
        if let Some(nodes_dto) = vis_dto.nodes {
            merge_clone_option!(target_vis.nodes.base_color, nodes_dto.base_color);
            merge_copy_option!(target_vis.nodes.metalness, nodes_dto.metalness);
            merge_copy_option!(target_vis.nodes.opacity, nodes_dto.opacity);
            merge_copy_option!(target_vis.nodes.roughness, nodes_dto.roughness);
            merge_copy_option!(target_vis.nodes.node_size, nodes_dto.node_size); // Changed from size_range, and to merge_copy_option
            merge_clone_option!(target_vis.nodes.quality, nodes_dto.quality);
            merge_copy_option!(target_vis.nodes.enable_instancing, nodes_dto.enable_instancing);
            merge_copy_option!(target_vis.nodes.enable_hologram, nodes_dto.enable_hologram);
            merge_copy_option!(target_vis.nodes.enable_metadata_shape, nodes_dto.enable_metadata_shape);
            merge_copy_option!(target_vis.nodes.enable_metadata_visualisation, nodes_dto.enable_metadata_visualisation);
        }
        if let Some(edges_dto) = vis_dto.edges {
            let target_edges = &mut target_vis.edges;
            merge_copy_option!(target_edges.arrow_size, edges_dto.arrow_size);
            merge_copy_option!(target_edges.base_width, edges_dto.base_width);
            merge_clone_option!(target_edges.color, edges_dto.color);
            merge_copy_option!(target_edges.enable_arrows, edges_dto.enable_arrows);
            merge_copy_option!(target_edges.opacity, edges_dto.opacity);
            merge_clone_option!(target_edges.width_range, edges_dto.width_range);
            merge_clone_option!(target_edges.quality, edges_dto.quality);
        }
        if let Some(physics_dto) = vis_dto.physics { // physics_dto is ClientPhysicsSettings
            let target_physics = &mut target_vis.physics; // Type: config::PhysicsSettings
            merge_copy_option!(target_physics.attraction_strength, physics_dto.attraction_strength);
            merge_copy_option!(target_physics.bounds_size, physics_dto.bounds_size);
            merge_copy_option!(target_physics.collision_radius, physics_dto.collision_radius);
            merge_copy_option!(target_physics.damping, physics_dto.damping);
            merge_copy_option!(target_physics.enable_bounds, physics_dto.enable_bounds);
            merge_copy_option!(target_physics.enabled, physics_dto.enabled);
            merge_copy_option!(target_physics.iterations, physics_dto.iterations);
            merge_copy_option!(target_physics.max_velocity, physics_dto.max_velocity);
            merge_copy_option!(target_physics.repulsion_strength, physics_dto.repulsion_strength);
            merge_copy_option!(target_physics.spring_strength, physics_dto.spring_strength);
            merge_copy_option!(target_physics.repulsion_distance, physics_dto.repulsion_distance);
            merge_copy_option!(target_physics.mass_scale, physics_dto.mass_scale);
            merge_copy_option!(target_physics.boundary_damping, physics_dto.boundary_damping);
        }
         if let Some(rendering_dto) = vis_dto.rendering { // rendering_dto is ClientRenderingSettings
            let target_rendering = &mut target_vis.rendering; // Type: config::RenderingSettings
            merge_copy_option!(target_rendering.ambient_light_intensity, rendering_dto.ambient_light_intensity);
            merge_clone_option!(target_rendering.background_color, rendering_dto.background_color);
            merge_copy_option!(target_rendering.directional_light_intensity, rendering_dto.directional_light_intensity);
            merge_copy_option!(target_rendering.enable_ambient_occlusion, rendering_dto.enable_ambient_occlusion);
            merge_copy_option!(target_rendering.enable_antialiasing, rendering_dto.enable_antialiasing);
            merge_copy_option!(target_rendering.enable_shadows, rendering_dto.enable_shadows);
            merge_copy_option!(target_rendering.environment_intensity, rendering_dto.environment_intensity);
        }
         if let Some(anim_dto) = vis_dto.animations { // anim_dto is ClientAnimationSettings
            let target_anim = &mut target_vis.animations; // Type: config::AnimationSettings
            merge_copy_option!(target_anim.enable_motion_blur, anim_dto.enable_motion_blur);
            merge_copy_option!(target_anim.enable_node_animations, anim_dto.enable_node_animations);
            merge_copy_option!(target_anim.motion_blur_strength, anim_dto.motion_blur_strength);
            merge_copy_option!(target_anim.selection_wave_enabled, anim_dto.selection_wave_enabled);
            merge_copy_option!(target_anim.pulse_enabled, anim_dto.pulse_enabled);
            merge_copy_option!(target_anim.pulse_speed, anim_dto.pulse_speed);
            merge_copy_option!(target_anim.pulse_strength, anim_dto.pulse_strength);
            merge_copy_option!(target_anim.wave_speed, anim_dto.wave_speed);
        }
         if let Some(labels_dto) = vis_dto.labels { // labels_dto is ClientLabelSettings
            let target_labels = &mut target_vis.labels; // Type: config::LabelSettings
            merge_copy_option!(target_labels.desktop_font_size, labels_dto.desktop_font_size);
            merge_copy_option!(target_labels.enable_labels, labels_dto.enable_labels);
            merge_clone_option!(target_labels.text_color, labels_dto.text_color);
            merge_clone_option!(target_labels.text_outline_color, labels_dto.text_outline_color);
            merge_copy_option!(target_labels.text_outline_width, labels_dto.text_outline_width);
            merge_copy_option!(target_labels.text_resolution, labels_dto.text_resolution);
            merge_copy_option!(target_labels.text_padding, labels_dto.text_padding);
            merge_clone_option!(target_labels.billboard_mode, labels_dto.billboard_mode);
        }
         if let Some(bloom_dto) = vis_dto.bloom { // bloom_dto is ClientBloomSettings
            let target_bloom = &mut target_vis.bloom; // Type: config::BloomSettings
            merge_copy_option!(target_bloom.edge_bloom_strength, bloom_dto.edge_bloom_strength);
            merge_copy_option!(target_bloom.enabled, bloom_dto.enabled);
            merge_copy_option!(target_bloom.environment_bloom_strength, bloom_dto.environment_bloom_strength);
            merge_copy_option!(target_bloom.node_bloom_strength, bloom_dto.node_bloom_strength);
            merge_copy_option!(target_bloom.radius, bloom_dto.radius);
            merge_copy_option!(target_bloom.strength, bloom_dto.strength);
        }
         if let Some(hologram_dto) = vis_dto.hologram { // hologram_dto is ClientHologramSettings
            let target_hologram = &mut target_vis.hologram; // Type: config::HologramSettings
            merge_copy_option!(target_hologram.ring_count, hologram_dto.ring_count);
            merge_clone_option!(target_hologram.ring_color, hologram_dto.ring_color);
            merge_copy_option!(target_hologram.ring_opacity, hologram_dto.ring_opacity);
            merge_clone_option!(target_hologram.sphere_sizes, hologram_dto.sphere_sizes);
            merge_copy_option!(target_hologram.ring_rotation_speed, hologram_dto.ring_rotation_speed);
            merge_copy_option!(target_hologram.enable_buckminster, hologram_dto.enable_buckminster);
            merge_copy_option!(target_hologram.buckminster_size, hologram_dto.buckminster_size);
            merge_copy_option!(target_hologram.buckminster_opacity, hologram_dto.buckminster_opacity);
            merge_copy_option!(target_hologram.enable_geodesic, hologram_dto.enable_geodesic);
            merge_copy_option!(target_hologram.geodesic_size, hologram_dto.geodesic_size);
            merge_copy_option!(target_hologram.geodesic_opacity, hologram_dto.geodesic_opacity);
            merge_copy_option!(target_hologram.enable_triangle_sphere, hologram_dto.enable_triangle_sphere);
            merge_copy_option!(target_hologram.triangle_sphere_size, hologram_dto.triangle_sphere_size);
            merge_copy_option!(target_hologram.triangle_sphere_opacity, hologram_dto.triangle_sphere_opacity);
            merge_copy_option!(target_hologram.global_rotation_speed, hologram_dto.global_rotation_speed);
        }
    }

    if let Some(xr_dto) = client_payload.xr {
        let target_xr = &mut settings_guard.xr;
        merge_clone_option!(target_xr.mode, xr_dto.mode);
        merge_copy_option!(target_xr.room_scale, xr_dto.room_scale);
        merge_clone_option!(target_xr.space_type, xr_dto.space_type);
        merge_clone_option!(target_xr.quality, xr_dto.quality);
        merge_copy_option!(target_xr.enable_hand_tracking, xr_dto.enable_hand_tracking);
        merge_copy_option!(target_xr.hand_mesh_enabled, xr_dto.hand_mesh_enabled);
        merge_clone_option!(target_xr.hand_mesh_color, xr_dto.hand_mesh_color);
        merge_copy_option!(target_xr.hand_mesh_opacity, xr_dto.hand_mesh_opacity);
        merge_copy_option!(target_xr.hand_point_size, xr_dto.hand_point_size);
        merge_copy_option!(target_xr.hand_ray_enabled, xr_dto.hand_ray_enabled);
        merge_clone_option!(target_xr.hand_ray_color, xr_dto.hand_ray_color);
        merge_copy_option!(target_xr.hand_ray_width, xr_dto.hand_ray_width);
        merge_copy_option!(target_xr.gesture_smoothing, xr_dto.gesture_smoothing);
        merge_copy_option!(target_xr.enable_haptics, xr_dto.enable_haptics);
        merge_copy_option!(target_xr.drag_threshold, xr_dto.drag_threshold);
        merge_copy_option!(target_xr.pinch_threshold, xr_dto.pinch_threshold);
        merge_copy_option!(target_xr.rotation_threshold, xr_dto.rotation_threshold);
        merge_copy_option!(target_xr.interaction_radius, xr_dto.interaction_radius);
        merge_copy_option!(target_xr.movement_speed, xr_dto.movement_speed);
        merge_copy_option!(target_xr.dead_zone, xr_dto.dead_zone);
        if let Some(axes_dto) = xr_dto.movement_axes {
            merge_copy_option!(target_xr.movement_axes.horizontal, axes_dto.horizontal);
            merge_copy_option!(target_xr.movement_axes.vertical, axes_dto.vertical);
        }
        merge_copy_option!(target_xr.enable_light_estimation, xr_dto.enable_light_estimation);
        merge_copy_option!(target_xr.enable_plane_detection, xr_dto.enable_plane_detection);
        merge_copy_option!(target_xr.enable_scene_understanding, xr_dto.enable_scene_understanding);
        merge_clone_option!(target_xr.plane_color, xr_dto.plane_color);
        merge_copy_option!(target_xr.plane_opacity, xr_dto.plane_opacity);
        merge_copy_option!(target_xr.plane_detection_distance, xr_dto.plane_detection_distance);
        merge_copy_option!(target_xr.show_plane_overlay, xr_dto.show_plane_overlay);
        merge_copy_option!(target_xr.snap_to_floor, xr_dto.snap_to_floor);
        merge_copy_option!(target_xr.enable_passthrough_portal, xr_dto.enable_passthrough_portal);
        merge_copy_option!(target_xr.passthrough_opacity, xr_dto.passthrough_opacity);
        merge_copy_option!(target_xr.passthrough_brightness, xr_dto.passthrough_brightness);
        merge_copy_option!(target_xr.passthrough_contrast, xr_dto.passthrough_contrast);
        merge_copy_option!(target_xr.portal_size, xr_dto.portal_size);
        merge_clone_option!(target_xr.portal_edge_color, xr_dto.portal_edge_color);
        merge_copy_option!(target_xr.portal_edge_width, xr_dto.portal_edge_width);
        
        // Manual merge for Option<T> fields in XRSettings
        if xr_dto.enabled.is_some() { target_xr.enabled = xr_dto.enabled; }
        if xr_dto.controller_model.is_some() { target_xr.controller_model = xr_dto.controller_model.clone(); }
        if xr_dto.render_scale.is_some() { target_xr.render_scale = xr_dto.render_scale; }
        if xr_dto.locomotion_method.is_some() { target_xr.locomotion_method = xr_dto.locomotion_method.clone(); }
        if xr_dto.teleport_ray_color.is_some() { target_xr.teleport_ray_color = xr_dto.teleport_ray_color.clone(); }
        if xr_dto.mode.is_some() { target_xr.display_mode = xr_dto.mode.clone(); } // xr_dto.mode maps to target_xr.display_mode (Option<String>)
        if xr_dto.controller_ray_color.is_some() { target_xr.controller_ray_color = xr_dto.controller_ray_color.clone(); }
    }

if let Some(auth_dto) = client_payload.auth {
        let target_auth = &mut settings_guard.auth;
        merge_copy_option!(target_auth.enabled, auth_dto.enabled);
        merge_clone_option!(target_auth.provider, auth_dto.provider);
        merge_copy_option!(target_auth.required, auth_dto.required);
    }

    if let Some(sys_dto) = client_payload.system {
        let target_sys_config = &mut settings_guard.system; // ServerSystemConfigFromFile
        if let Some(ws_dto) = sys_dto.websocket {
            let target_ws = &mut target_sys_config.websocket; // ServerFullWebSocketSettings
            merge_copy_option!(target_ws.reconnect_attempts, ws_dto.reconnect_attempts);
            merge_copy_option!(target_ws.reconnect_delay, ws_dto.reconnect_delay);
            merge_copy_option!(target_ws.binary_chunk_size, ws_dto.binary_chunk_size);
            merge_copy_option!(target_ws.compression_enabled, ws_dto.compression_enabled);
            merge_copy_option!(target_ws.compression_threshold, ws_dto.compression_threshold);
            merge_copy_option!(target_ws.update_rate, ws_dto.update_rate);
        }
        if let Some(debug_dto) = sys_dto.debug {
            let target_debug = &mut target_sys_config.debug; // DebugSettings (server version)
            merge_copy_option!(target_debug.enabled, debug_dto.enabled);
            merge_copy_option!(target_debug.enable_data_debug, debug_dto.enable_data_debug);
            merge_copy_option!(target_debug.enable_websocket_debug, debug_dto.enable_websocket_debug);
            merge_copy_option!(target_debug.log_binary_headers, debug_dto.log_binary_headers);
            merge_copy_option!(target_debug.log_full_json, debug_dto.log_full_json);
        }
        merge_copy_option!(target_sys_config.persist_settings, sys_dto.persist_settings);
    }
    
    // AI settings merge (all are Option<Struct> on AppFullSettings)
    if client_payload.ragflow.is_some() { settings_guard.ragflow = client_payload.ragflow.map(|dto| crate::config::RagFlowSettings {
        api_key: dto.api_key, agent_id: dto.agent_id, api_base_url: dto.api_base_url,
        timeout: dto.timeout, max_retries: dto.max_retries, chat_id: dto.chat_id,
    })};
    if client_payload.perplexity.is_some() { settings_guard.perplexity = client_payload.perplexity.map(|dto| crate::config::PerplexitySettings {
        api_key: dto.api_key, model: dto.model, api_url: dto.api_url, max_tokens: dto.max_tokens,
        temperature: dto.temperature, top_p: dto.top_p, presence_penalty: dto.presence_penalty,
        frequency_penalty: dto.frequency_penalty, timeout: dto.timeout, rate_limit: dto.rate_limit,
    })};
    if client_payload.openai.is_some() { settings_guard.openai = client_payload.openai.map(|dto| crate::config::OpenAISettings {
        api_key: dto.api_key, base_url: dto.base_url, timeout: dto.timeout, rate_limit: dto.rate_limit,
    })};
    if client_payload.kokoro.is_some() { settings_guard.kokoro = client_payload.kokoro.map(|dto| crate::config::KokoroSettings {
        api_url: dto.api_url, default_voice: dto.default_voice, default_format: dto.default_format,
        default_speed: dto.default_speed, timeout: dto.timeout, stream: dto.stream,
        return_timestamps: dto.return_timestamps, sample_rate: dto.sample_rate,
    })};
    // --- End Merge ---

    if let Err(e) = settings_guard.save() {
        error!("Failed to save global AppFullSettings after update from {}: {}", pubkey, e);
        return Ok(HttpResponse::InternalServerError().body(format!("Failed to save settings: {}", e)));
    }

    info!("Power user {} updated global settings via /user-settings endpoint", pubkey);
    let updated_ui_settings = convert_to_ui_settings(&*settings_guard);
    // Consider broadcasting here too if this endpoint remains active
    Ok(HttpResponse::Ok().json(updated_ui_settings))
}

// --- GET Graph Specific Settings ---

pub async fn get_graph_settings(app_state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    let settings_guard = app_state.settings.read().await; // Reads AppFullSettings
    Ok(HttpResponse::Ok().json(&settings_guard.visualisation))
}
