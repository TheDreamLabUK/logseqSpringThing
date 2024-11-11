use actix_web::{web, HttpResponse};
use log::{info, error};
use chrono::Utc;
use std::collections::HashMap;
use crate::app_state::AppState;
use crate::services::perplexity_service::{ApiClientImpl, PerplexityService};
use crate::services::file_service::ProcessedFile;
use crate::models::metadata::Metadata;

pub async fn process_files(app_state: web::Data<AppState>) -> HttpResponse {
    info!("Starting Perplexity processing for all files");
    
    let settings = app_state.settings.read().await;
    let api_client = ApiClientImpl::new();
    let file_cache = app_state.file_cache.read().await;
    let graph_data = app_state.graph_data.read().await;
    
    let mut processed_count = 0;
    let mut error_count = 0;
    let mut pr_urls = Vec::new();

    for (file_name, content) in file_cache.iter() {
        let metadata = graph_data.metadata.get(file_name).cloned().unwrap_or_else(|| {
            error!("No metadata found for file: {}", file_name);
            Metadata {
                file_name: file_name.clone(),
                last_modified: Utc::now(),
                topic_counts: HashMap::new(),
                ..Default::default()
            }
        });

        let processed_file = ProcessedFile {
            file_name: file_name.clone(),
            content: content.clone(),
            is_public: true,
            metadata: metadata.clone(),
        };

        match app_state.perplexity_service.process_file(processed_file, &settings, &api_client).await {
            Ok(processed) => {
                // Update file cache with processed content
                let mut file_cache = app_state.file_cache.write().await;
                file_cache.insert(file_name.clone(), processed.content.clone());
                
                // Create GitHub PR for the processed file
                match app_state.github_pr_service.create_pull_request(
                    file_name,
                    &processed.content,
                    &metadata.sha1,
                ).await {
                    Ok(pr_url) => {
                        info!("Created PR for {}: {}", file_name, pr_url);
                        pr_urls.push((file_name.clone(), pr_url));
                    }
                    Err(e) => {
                        error!("Failed to create PR for {}: {}", file_name, e);
                    }
                }
                
                processed_count += 1;
                info!("Successfully processed file: {}", file_name);
            }
            Err(e) => {
                error!("Error processing file {}: {}", file_name, e);
                error_count += 1;
            }
        }
    }

    HttpResponse::Ok().json(serde_json::json!({
        "status": "completed",
        "processed_files": processed_count,
        "errors": error_count,
        "pull_requests": pr_urls.into_iter().collect::<HashMap<_, _>>()
    }))
}
