use actix_web::{web, HttpResponse};
use crate::AppState;
use crate::services::file_service::{FileService, RealGitHubService};
use crate::services::perplexity_service::{PerplexityService, RealPerplexityService};

pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    match FileService::fetch_files_from_github::<RealGitHubService>().await {
        Ok(github_files) => {
            match FileService::compare_and_identify_updates(github_files.clone()) {
                Ok(updated_files) => {
                    let mut processed_files = Vec::new();

                    for file_name in &updated_files {
                        if let Some(file) = github_files.iter().find(|f| f.name == *file_name) {
                            match RealPerplexityService::process_file(file.content.clone()).await {
                                Ok(processed_file) => {
                                    let metadata = crate::models::metadata::Metadata {
                                        file_name: file.name.clone(),
                                        last_modified: chrono::Utc::now(),
                                        processed_file: processed_file.content,
                                        original_file: file.content.clone(),
                                    };

                                    if let Err(_) = FileService::save_file_metadata(metadata) {
                                        eprintln!("Failed to save metadata for file: {}", file.name);
                                    }

                                    let mut file_cache = state.file_cache.write().await;
                                    file_cache.insert(file.name.clone(), file.content.clone());

                                    processed_files.push(file.name.clone());
                                },
                                Err(_) => eprintln!("Failed to process file: {}", file.name),
                            }
                        }
                    }

                    HttpResponse::Ok().json(processed_files)
                },
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}
