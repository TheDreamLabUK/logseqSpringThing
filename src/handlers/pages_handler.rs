use actix_web::{web, HttpResponse, Result, Error};
use crate::AppState;
use crate::actors::messages::GetSettings;
use serde::Serialize;
use futures::future::join_all;
use crate::models::metadata::Metadata;
use crate::services::github::GitHubFileMetadata;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageInfo {
    id: String,
    title: String,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    modified: i64,
}

pub async fn get_pages(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings_addr.send(GetSettings).await
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Settings actor mailbox error: {}", e)))?
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    let debug_enabled = settings.system.debug.enabled;

    if debug_enabled {
        log::debug!("Starting pages retrieval");
    }

    let metadata = app_state.metadata.read().await;
    if debug_enabled {
        log::debug!("Found {} metadata entries to process", metadata.len());
    }

    let futures: Vec<_> = metadata.iter()
        .map(|(id, meta)| {
            let content_api = app_state.content_api.clone();
            let file_name = meta.file_name.clone();
            let id = id.clone();
            let meta = meta.clone();
            let debug_enabled = debug_enabled;

            async move {
                if debug_enabled {
                    log::debug!("Processing file: {} (ID: {})", file_name, id);
                }

                let github_meta = content_api
                    .list_markdown_files("")  // Empty string since base path is already configured
                    .await;

                match github_meta {
                    Ok(files) => {
                        if debug_enabled {
                            log::debug!("Found {} GitHub files for {}", files.len(), file_name);
                        }

                        let matching_file = files.into_iter()
                            .find(|f| f.name == file_name);

                        if debug_enabled {
                            if let Some(ref file) = matching_file {
                                log::debug!("Found matching GitHub file for {}: {:?}", file_name, file);
                            } else {
                                log::debug!("No matching GitHub file found for {}", file_name);
                            }
                        }

                        Ok((id, meta, matching_file))
                    },
                    Err(e) => {
                        log::error!("Failed to fetch GitHub metadata for {}: {}", file_name, e);
                        Ok((id, meta, None))
                    }
                }
            }
        })
        .collect();
    
    if debug_enabled {
        log::debug!("Created {} futures for parallel processing", futures.len());
    }

    let results = join_all(futures).await;
    
    let pages: Vec<PageInfo> = results.into_iter()
        .filter_map(|result: Result<(String, Metadata, Option<GitHubFileMetadata>), actix_web::Error>| {
            match result {
                Ok((id, meta, github_meta)) => {
                    if debug_enabled {
                        log::debug!("Building page info for {} (ID: {})", meta.file_name, id);
                    }

                    let modified = github_meta
                        .and_then(|gm| gm.last_modified)
                        .map(|dt| dt.timestamp())
                        .unwrap_or_else(|| {
                            if debug_enabled {
                                log::debug!("No modification time found for {}, using 0", meta.file_name);
                            }
                            0
                        });

                    Some(PageInfo {
                        id,
                        title: meta.file_name.clone(),
                        path: format!("/app/data/markdown/{}", meta.file_name),
                        parent: None,
                        modified,
                    })
                },
                Err(e) => {
                    log::error!("Failed to process page: {}", e);
                    None
                }
            }
        })
        .collect();

    if debug_enabled {
        log::debug!("Returning {} processed pages", pages.len());
    }

    Ok(HttpResponse::Ok().json(pages))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(get_pages))
    );
} 