use actix_web::{web, App, HttpServer};
use actix_web::middleware::Logger;
use handlers::{settings, graph_handler, websocket, metrics, gpu_compute, health};
use state::AppState;
use utils::metrics::MetricsCollector;
use middleware::rate_limit::RateLimiter;
use middleware::auth::{check_auth, cors_config};
use std::sync::Arc;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    let metrics = MetricsCollector::new();
    let rate_limiter = Arc::new(RateLimiter::new(60, 100));
    let state = Arc::new(AppState::new(
        std::env::var("BACKUP_DIR").unwrap_or_else(|_| "backups".to_string()).into(),
        metrics,
    ).await?);

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .wrap(cors_config())
            .app_data(web::Data::new(state.clone()))
            .app_data(web::Data::new(rate_limiter.clone()))
            .route("/health", web::get().to(health::health_check))
            .service(
                web::scope("/api")
                    .wrap_fn(|req, srv| {
                        let fut = srv.call(req);
                        async move {
                            let req = check_auth(fut.await??).await?;
                            Ok(req)
                        }
                    })
                    .service(
                        web::scope("/visualization/settings")
                            .route("/{category}/{setting}", web::get().to(settings::get_setting))
                            .route("/{category}/{setting}", web::put().to(settings::update_setting))
                            .route("/{category}", web::get().to(settings::get_category_settings))
                            .route("", web::get().to(settings::get_all_settings))
                            .route("", web::put().to(settings::update_settings))
                    )
                    .service(
                        web::scope("/graph")
                            .route("/data", web::get().to(graph_handler::get_graph_data))
                            .route("/data/paginated", web::get().to(graph_handler::get_paginated_data))
                            .route("/update", web::post().to(graph_handler::update_graph))
                    )
                    .service(
                        web::scope("/settings/websocket")
                            .route("", web::get().to(settings::get_websocket_settings))
                            .route("/{setting}", web::put().to(settings::update_websocket_setting))
                    )
                    .route("/metrics", web::get().to(metrics::get_metrics))
                    .service(
                        web::scope("/gpu")
                            .route("/status", web::get().to(gpu_compute::get_gpu_compute_status))
                            .route("/status", web::put().to(gpu_compute::set_gpu_compute_status))
                    )
                    .route("/ws", web::get().to(websocket::ws_route))
            )
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
} 