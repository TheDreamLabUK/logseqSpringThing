use actix_web::{web, HttpResponse, Result};
use crate::AppState;

pub async fn get_settings(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let settings = app_state.settings.read().await;
    Ok(HttpResponse::Ok().json(&*settings))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(get_settings))
    );
} 