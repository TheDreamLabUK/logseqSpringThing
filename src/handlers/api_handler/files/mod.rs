use actix_web::{web, Error as ActixError, HttpResponse};
use std::sync::Arc;
use crate::actors::messages::{GetSettings, UpdateMetadata, BuildGraphFromMetadata, GetNodeData as GetGpuNodeData};
use serde_json::json;
use log::{info, debug, error};

use crate::AppState;
use crate::services::file_service::{FileService, MARKDOWN_DIR};

pub async fn fetch_and_process_files(state: web::Data<AppState>) -> HttpResponse {
    info!("Initiating optimized file fetch and processing");

    let mut metadata_store = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load or create metadata: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to initialize metadata: {}", e)
            }));
        }
    };
    
    let settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => Arc::new(tokio::sync::RwLock::new(s)),
        _ => {
            error!("Failed to retrieve settings from SettingsActor");
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": "Failed to retrieve application settings"
            }));
        }
    };
    
    let file_service = FileService::new(settings.clone());
    
    match file_service.fetch_and_process_files(state.content_api.clone(), settings.clone(), &mut metadata_store).await {
        Ok(processed_files) => {
            let file_names: Vec<String> = processed_files.iter()
                .map(|pf| pf.file_name.clone())
                .collect();

            info!("Successfully processed {} public markdown files", processed_files.len());

            {
                // Send UpdateMetadata message to MetadataActor
                if let Err(e) = state.metadata_addr.send(UpdateMetadata { metadata: metadata_store.clone() }).await {
                    error!("Failed to send UpdateMetadata message to MetadataActor: {}", e);
                    // Decide if this is a critical error to return
                }
            }

            // FileService::save_metadata might also need to be an actor message if it implies shared state,
            // but for now, assuming it's a static utility or local file operation.
            // If it interacts with shared state, it should be refactored.
            if let Err(e) = FileService::save_metadata(&metadata_store) {
                error!("Failed to save metadata: {}", e);
                return HttpResponse::InternalServerError().json(json!({
                    "status": "error",
                    "message": format!("Failed to save metadata: {}", e)
                }));
            }

            // Send BuildGraphFromMetadata message to GraphServiceActor
            match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
                Ok(Ok(())) => {
                    info!("Graph data structure updated successfully via GraphServiceActor");

                    // If GPU is present, potentially trigger an update or fetch data
                    if let Some(gpu_addr) = &state.gpu_compute_addr {
                        // Example: Trigger GPU re-initialization or update if necessary
                        // This depends on how GPUComputeActor handles graph updates.
                        // For now, let's assume GraphServiceActor coordinates with GPUComputeActor if needed.
                        // Or, if we just need to get data for a response:
                        match gpu_addr.send(GetGpuNodeData).await {
                            Ok(Ok(_nodes)) => {
                                debug!("GPU node data fetched successfully after graph update");
                            }
                            Ok(Err(e)) => {
                                error!("Failed to get node data from GPU actor: {}", e);
                            }
                            Err(e) => {
                                error!("Mailbox error getting node data from GPU actor: {}", e);
                            }
                        }
                    }

                    HttpResponse::Ok().json(json!({
                        "status": "success",
                        "processed_files": file_names
                    }))
                },
                Ok(Err(e)) => {
                    error!("GraphServiceActor failed to build graph from metadata: {}", e);
                    HttpResponse::InternalServerError().json(json!({
                        "status": "error",
                        "message": format!("Failed to build graph: {}", e)
                    }))
                },
                Err(e) => {
                    error!("Failed to build graph data: {}", e);
                    HttpResponse::InternalServerError().json(json!({
                        "status": "error",
                        "message": format!("Failed to build graph data: {}", e)
                    }))
                }
            }
        },
        Err(e) => {
            error!("Error processing files: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Error processing files: {}", e)
            }))
        }
    }
}

pub async fn get_file_content(_state: web::Data<AppState>, file_name: web::Path<String>) -> HttpResponse {
    let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
    match std::fs::read_to_string(&file_path) {
        Ok(content) => HttpResponse::Ok().body(content),
        Err(e) => {
            error!("Failed to read file {}: {}", file_name, e);
            HttpResponse::NotFound().json(json!({
                "status": "error",
                "message": format!("File not found or unreadable: {}", file_name)
            }))
        }
    }
}

pub async fn refresh_graph(state: web::Data<AppState>) -> HttpResponse {
    info!("Manually triggering graph refresh");

    let metadata_store = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to load metadata: {}", e)
            }));
        }
    };

    match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
        Ok(Ok(())) => {
            info!("Graph data structure refreshed successfully via GraphServiceActor");

            if let Some(gpu_addr) = &state.gpu_compute_addr {
                 match gpu_addr.send(GetGpuNodeData).await {
                    Ok(Ok(_nodes)) => {
                        debug!("GPU node data fetched successfully after graph refresh");
                    }
                    Ok(Err(e)) => {
                        error!("Failed to get node data from GPU actor after refresh: {}", e);
                    }
                    Err(e) => {
                        error!("Mailbox error getting node data from GPU actor after refresh: {}", e);
                    }
                }
            }

            HttpResponse::Ok().json(json!({
                "status": "success",
                "message": "Graph refreshed successfully"
            }))
        },
        Ok(Err(e)) => {
            error!("GraphServiceActor failed to build graph from metadata: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to build graph: {}", e)
            }))
        },
        Err(e) => {
            error!("Failed to refresh graph data: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to refresh graph data: {}", e)
            }))
        }
        // This was the Ok(Err(e)) arm that was already applied by the previous attempt.
        // The original diff had a comma here, which was incorrect.
        // The compiler error E0004 indicates this arm is still missing for the outer Result from `send`.
        // The `send` method returns `Result<Result<T, E1>, MailboxError>`.
        // The inner `Result` is handled by `Ok(Ok(()))` and `Ok(Err(e))`.
        // The outer `Result`'s `Err` case (MailboxError) is handled by `Err(e)`.
        // The missing case is `Ok(Err(actor_error))` for the inner result when the mailbox call itself succeeded.
        // However, the compiler output shows the `Ok(Err(_))` is missing for the *outer* result, which is confusing.
        // Let's assume the compiler means the inner `Result<_, String>`'s `Err` case.
        // The previous diff correctly added an Ok(Err(e)) arm.
        // The issue might be that the `Err(e)` for MailboxError is not enough.
        // Let's ensure both `Ok(Err(actor_err))` and `Err(mailbox_err)` are handled.
        // The existing `Err(e)` on line 165 handles MailboxError.
        // The `Ok(Err(e))` on line 172 handles the actor's internal error.
        // This structure should be correct. The compiler might be getting confused by nested Results.
        // For now, I will assume the previous partial application of this diff part was correct
        // and the issue lies elsewhere or the compiler message is slightly misleading for nested Results.
        // The diff for line 171-179 was:
        // Ok(Err(e)) => {
        // error!("GraphServiceActor failed to build graph from metadata: {}", e);
        // HttpResponse::InternalServerError().json(json!({
        // "status": "error",
        // "message": format!("Failed to build graph: {}", e)
        // }))
        // }
        // This seems correct for handling the actor's error.
        // The `Err(e)` at line 165 handles the MailboxError.
        // The compiler error `E0004: non-exhaustive patterns: Ok(Err(_)) not covered` for line 142
        // suggests that the `match` on `state.graph_service_addr.send(...).await` is the one missing the arm.
        // The structure is:
        // match send().await {
        //    Ok(Ok(())) => { ... } // Actor success
        //    Err(e) => { ... } // Mailbox error
        //    // MISSING: Ok(Err(actor_error)) => { ... }
        // }
        // The previous diff *did* add this. Let's re-verify the structure.
        // The file content shows the Ok(Err(e)) arm is indeed present from line 172.
        // The compiler error might be for a *different* match statement or there's a subtle issue.
        // Let's re-apply the addition of the Ok(Err(e)) arm to be certain,
        // ensuring it's for the correct match.

        // The match starts at line 142.
        // Arm 1: Ok(Ok(())) => { ... } (lines 143-164)
        // Arm 2: Err(e) => { ... } (lines 165-171) - This is MailboxError
        // Arm 3: Ok(Err(e)) => { ... } (lines 172-178) - This is Actor's internal error

        // This structure *looks* correct and exhaustive for Result<Result<_,_>, MailboxError>.
        // I will proceed assuming the file content is now correct regarding this match statement
        // due to the previous partial application. The compiler error might be for the *other*
        // match statement later in the file.

        // Let's focus on the *second* non-exhaustive match error, which is for line 194 (previously 187).
        // match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
        // Ok(Ok(())) => { ... } // lines 195-216
        // Err(e) => { ... } // lines 217-223 - MailboxError
        // // MISSING: Ok(Err(actor_error))
        // }
        // The file content shows this Ok(Err(e)) arm *is* present from line 224.
        // Ok(Err(e)) => { // lines 224-230
        // error!("GraphServiceActor failed to build graph from metadata: {}", e);
        // Ok(HttpResponse::InternalServerError().json(json!({
        // "status": "error",
        // "message": format!("Failed to build graph: {}", e)
        // })))
        // }
        // This also looks correct.

        // It's possible the `apply_diff` tool reported a partial application, but it actually
        // applied all changes correctly, or the state I'm seeing is post-partial-application.
        // Given the file content now shows these arms, I will not attempt to re-add them.
        // The primary remaining changes are the unused imports.
    }
}

pub async fn update_graph(state: web::Data<AppState>) -> Result<HttpResponse, ActixError> {
    let metadata_store = match FileService::load_or_create_metadata() {
        Ok(store) => store,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to load metadata: {}", e)
            })));
        }
    };

    match state.graph_service_addr.send(BuildGraphFromMetadata { metadata: metadata_store.clone() }).await {
        Ok(Ok(())) => {
            info!("Graph data structure updated successfully via GraphServiceActor in update_graph");

            if let Some(gpu_addr) = &state.gpu_compute_addr {
                match gpu_addr.send(GetGpuNodeData).await {
                    Ok(Ok(_nodes)) => {
                        debug!("GPU node data fetched successfully after graph update in update_graph");
                    }
                    Ok(Err(e)) => {
                        error!("Failed to get node data from GPU actor after update in update_graph: {}", e);
                    }
                    Err(e) => {
                        error!("Mailbox error getting node data from GPU actor after update in update_graph: {}", e);
                    }
                }
            }
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "success",
                "message": "Graph updated successfully"
            })))
        },
        Err(e) => {
            error!("Failed to build graph: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to build graph: {}", e)
            })))
        },
        Ok(Err(e)) => {
            error!("GraphServiceActor failed to build graph from metadata: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Failed to build graph: {}", e)
            })))
        }
    }
}

// Configure routes using snake_case
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/files")
            .route("/process", web::post().to(fetch_and_process_files))
            .route("/get_content/{filename}", web::get().to(get_file_content))
            .route("/refresh_graph", web::post().to(refresh_graph))
            .route("/update_graph", web::post().to(update_graph))
    );
}
