use actix_web::{web, HttpResponse, Result};
use crate::AppState;
use serde::Serialize;
use futures::future::join_all;
use crate::services::github::ContentAPI;

#[derive(Serialize)]
pub struct PageInfo {
    id: String,
    title: String,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    modified: i64,
}

pub async fn get_pages(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let metadata = app_state.metadata.read().await;
    let futures: Vec<_> = metadata.iter()
        .map(|(id, meta)| async {
            let github_meta = app_state.content_api
                .list_markdown_files("")
                .await
                .ok()
                .and_then(|files| files.into_iter()
                    .find(|f| f.name == meta.file_name));
            
            Ok::<_, actix_web::Error>((id.clone(), meta.clone(), github_meta))
        })
        .collect();
    
    let results = join_all(futures).await;
    
    let pages: Vec<PageInfo> = results.into_iter()
        .filter_map(Result::ok)
        .map(|(id, meta, github_meta)| PageInfo {
            id,
            title: meta.file_name.clone(),
            path: format!("/app/data/markdown/{}", meta.file_name),
            parent: None,
            modified: github_meta
                .and_then(|gm| gm.last_modified)
                .map(|dt| dt.timestamp())
                .unwrap_or(0),
        })
        .collect();

    Ok(HttpResponse::Ok().json(pages))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(get_pages))
    );
} 