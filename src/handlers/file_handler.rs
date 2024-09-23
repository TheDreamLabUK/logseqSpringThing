use actix_web::{web, HttpResponse};
use crate::AppState;
use crate::services::file_service::FileService;
use dotenv::dotenv;
use env_logger::Env;
use log::{info, error, debug};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    match FileService::fetch_and_process_files(&*state.github_service, &state.settings).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .enumerate()
                .map(|(index, _)| format!("processed_file_{}", index))
                .collect();

            log::info!("Successfully processed {} files", processed_files.len());

            // Update the file cache
            let mut file_cache = state.file_cache.write().await;
            for (name, file) in file_names.iter().zip(processed_files.iter()) {
                file_cache.insert(name.clone(), file.content.clone());
                log::debug!("Updated file cache with: {}", name);
            }

            // Here you would typically update your graph data structure
            // This is a placeholder for future graph update logic
            log::info!("Graph data structure should be updated here");

            HttpResponse::Ok().json(file_names)
        },
        Err(e) => {
            log::error!("Error processing files: {:?}", e);
            HttpResponse::InternalServerError().json(format!("Error processing files: {:?}", e))
        }
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Load environment variables
    dotenv().ok();

    // Initialize logger
    env_logger::init_from_env(Env::default().default_filter_or("debug"));

    info!("Starting file retrieval test");

    // Create AppState
    let settings = Settings::new().expect("Failed to load settings");
    let file_cache = Arc::new(RwLock::new(HashMap::new()));
    let graph_data = Arc::new(RwLock::new(GraphData::default()));
    let github_service = Arc::new(RealGitHubService::new());

    let app_state = web::Data::new(AppState {
        settings,
        file_cache,
        graph_data,
        github_service,
    });

    // Call fetch_and_process_files
    info!("Calling fetch_and_process_files");
    let response = fetch_and_process_files(app_state.clone()).await;

    match response.status() {
        actix_web::http::StatusCode::OK => {
            info!("File retrieval and processing successful");
            let body = response.into_body();
            match body.try_into_bytes() {
                Ok(bytes) => {
                    if let Ok(file_names) = serde_json::from_slice::<Vec<String>>(&bytes) {
                        info!("Processed files: {:?}", file_names);
                        
                        // Print file cache contents
                        let file_cache = app_state.file_cache.read().await;
                        for (name, content) in file_cache.iter() {
                            debug!("File '{}' content preview: {}", name, &content[..content.len().min(50)]);
                        }
                    } else {
                        error!("Failed to parse response body as JSON");
                    }
                },
                Err(e) => error!("Failed to extract response body: {:?}", e),
            }
        },
        _ => {
            error!("File retrieval and processing failed with status: {}", response.status());
            let body = response.into_body();
            match body.try_into_bytes() {
                Ok(bytes) => {
                    if let Ok(error_message) = String::from_utf8(bytes.to_vec()) {
                        error!("Error message: {}", error_message);
                    } else {
                        error!("Failed to parse error message");
                    }
                },
                Err(e) => error!("Failed to extract error message: {:?}", e),
            }
        }
    }

    info!("File retrieval test completed");

    Ok(())
}
}
