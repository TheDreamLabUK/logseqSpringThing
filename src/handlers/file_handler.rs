// src/handlers/file_handler.rs

use actix_web::{web, HttpResponse};
use crate::AppState;
use crate::services::file_service::FileService;
use crate::services::graph_service::GraphService;
use log::{info, error, debug};
use serde_json::json;

/// Handler for fetching and processing Markdown files from GitHub.
///
/// This function performs the following steps:
/// 1. Fetches Markdown files from a private GitHub repository using the GitHubService.
/// 2. Processes each file using the PerplexityService to enhance content.
/// 3. Updates the file cache with the processed content.
/// 4. Builds or refreshes the graph data structure based on the processed files.
/// 5. Returns a JSON response containing the names of the processed files.
///
/// # Arguments
///
/// * `state` - Shared application state.
///
/// # Returns
///
/// An HTTP response indicating success or failure.
pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    info!("Initiating file fetch and processing");
    
    // Step 1: Fetch files from GitHub
    match FileService::fetch_and_process_files(&*state.github_service, &state.settings).await {
        Ok(processed_files) => {
            // Generate unique names for processed files
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} files", processed_files.len());

            // Step 2: Update the file cache with processed content
            {
                let mut file_cache = state.file_cache.write().await;
                for processed_file in &processed_files {
                    file_cache.insert(processed_file.file_name.clone(), processed_file.content.clone());
                    debug!("Updated file cache with: {}", processed_file.file_name);
                }
            }

            // Step 3: Update graph data structure based on processed files
            match GraphService::build_graph(&state).await {
                Ok(graph_data) => {
                    let mut graph = state.graph_data.write().await;
                    *graph = graph_data.clone();
                    info!("Graph data structure updated successfully");

                    // Broadcast the updated graph to connected WebSocket clients
                    state.websocket_manager.broadcast_message(&json!({
                        "type": "graphUpdate",
                        "data": graph_data,
                    }).to_string());
                },
                Err(e) => {
                    error!("Failed to build graph data: {}", e);
                    return HttpResponse::InternalServerError().json(format!("Failed to build graph data: {}", e));
                }
            }

            // Step 4: Respond with the list of processed file names
            HttpResponse::Ok().json(json!({
                "status": "success",
                "processed_files": file_names
            }))
        },
        Err(e) => {
            error!("Error processing files: {:?}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Error processing files: {:?}", e)
            }))
        }
    }
}
